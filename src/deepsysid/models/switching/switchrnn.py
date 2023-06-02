import json
import logging
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from numpy.typing import NDArray

from ...networks import loss, rnn
from ...networks.switching import (
    StableIdentityOutputSwitchingLSTM,
    StableSwitchingLSTM,
    SwitchingBaseLSTM,
    UnconstrainedIdentityOutputSwitchingLSTM,
    UnconstrainedSwitchingLSTM,
)
from ...tracker.base import BaseEventTracker
from .. import base, utils
from ..datasets import RecurrentInitializerDataset, RecurrentPredictorDataset
from ..recurrent.separate_initialization import (
    SeparateInitializerRecurrentNetworkModelConfig,
)

logger = logging.getLogger(__name__)


class SwitchingLSTMBaseModelConfig(SeparateInitializerRecurrentNetworkModelConfig):
    switched_system_state_dim: Optional[int]


class SwitchingLSTMBaseModel(base.DynamicIdentificationModel):
    CONFIG = SwitchingLSTMBaseModelConfig

    def __init__(
        self, config: SwitchingLSTMBaseModelConfig, predictor: SwitchingBaseLSTM
    ):
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(config.device_name)

        self.control_dim = len(config.control_names)
        self.output_dim = len(config.state_names)

        self.recurrent_dim = config.recurrent_dim
        self.num_recurrent_layers = config.num_recurrent_layers
        self.dropout = config.dropout

        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs_initializer = config.epochs_initializer
        self.epochs_predictor = config.epochs_predictor

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        elif config.loss == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse" or "msge"')

        self.predictor = predictor.to(self.device)

        self.initializer = rnn.BasicLSTM(
            input_dim=self.control_dim + self.output_dim,
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=[self.output_dim],
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer_pred = optim.Adam(
            self.predictor.parameters(), lr=self.learning_rate
        )
        self.optimizer_init = optim.Adam(
            self.initializer.parameters(), lr=self.learning_rate
        )

        self.control_mean: Optional[NDArray[np.float64]] = None
        self.control_std: Optional[NDArray[np.float64]] = None
        self.output_mean: Optional[NDArray[np.float64]] = None
        self.output_std: Optional[NDArray[np.float64]] = None

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]] = None,
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> Dict[str, NDArray[np.float64]]:
        epoch_losses_initializer = []
        epoch_losses_predictor = []

        self.predictor.train()
        self.initializer.train()

        self.control_mean, self.control_std = utils.mean_stddev(control_seqs)
        self.output_mean, self.output_std = utils.mean_stddev(state_seqs)

        control_seqs = [
            utils.normalize(control, self.control_mean, self.control_std)
            for control in control_seqs
        ]
        state_seqs = [
            utils.normalize(state, self.output_mean, self.output_std)
            for state in state_seqs
        ]

        initializer_dataset = RecurrentInitializerDataset(
            control_seqs, state_seqs, self.sequence_length
        )

        time_start_init = time.time()
        for i in range(self.epochs_initializer):
            data_loader = data.DataLoader(
                initializer_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                self.initializer.zero_grad()
                y, _ = self.initializer.forward(batch['x'].float().to(self.device))
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_init.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_initializer} '
                f'- Epoch Loss (Initializer): {total_loss}'
            )
            epoch_losses_initializer.append([i, total_loss])
        time_end_init = time.time()

        predictor_dataset = RecurrentPredictorDataset(
            control_seqs, state_seqs, self.sequence_length
        )
        time_start_pred = time.time()
        for i in range(self.epochs_predictor):
            data_loader = data.DataLoader(
                predictor_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0
            for batch_idx, batch in enumerate(data_loader):
                self.predictor.zero_grad()
                # Initialize predictor with state of initializer network
                _, hx = self.initializer.forward(batch['x0'].float().to(self.device))
                # Predict and optimize
                output = self.predictor.forward(
                    control=batch['x'].float().to(self.device),
                    previous_output=batch['y0'].float().to(self.device),
                    hx=hx,
                )
                batch_loss = self.loss.forward(
                    output.outputs, batch['y'].float().to(self.device)
                )
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_pred.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_predictor} '
                f'- Epoch Loss (Predictor): {total_loss}'
            )
            epoch_losses_predictor.append([i, total_loss])
        time_end_pred = time.time()

        time_total_init = time_end_init - time_start_init
        time_total_pred = time_end_pred - time_start_pred
        logger.info(
            f'Training time for initializer {time_total_init}s '
            f'and for predictor {time_total_pred}s'
        )

        return dict(
            epoch_loss_initializer=np.array(epoch_losses_initializer, dtype=np.float64),
            epoch_loss_predictor=np.array(epoch_losses_predictor, dtype=np.float64),
            training_time_initializer=np.array([time_total_init], dtype=np.float64),
            training_time_predictor=np.array([time_total_pred], dtype=np.float64),
        )

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        x0: Optional[NDArray[np.float64]] = None,
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        if (
            self.output_mean is None
            or self.output_std is None
            or self.control_mean is None
            or self.control_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        system_matrices = []
        control_matrices = []

        self.initializer.eval()
        self.predictor.eval()

        initial_control = utils.normalize(
            initial_control, self.control_mean, self.control_std
        )
        initial_state = utils.normalize(
            initial_state, self.output_mean, self.output_std
        )
        control = utils.normalize(control, self.control_mean, self.control_std)

        with torch.no_grad():
            init_x = (
                torch.from_numpy(np.hstack((initial_control[1:], initial_state[:-1])))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            pred_x = torch.from_numpy(control).unsqueeze(0).float().to(self.device)

            y0 = (
                torch.from_numpy(initial_state[-1]).unsqueeze(0).float().to(self.device)
            )
            _, hx = self.initializer.forward(init_x)
            output = self.predictor.forward(control=pred_x, previous_output=y0, hx=hx)
            y_np = output.outputs.cpu().detach().squeeze().numpy()

            system_matrices.append(
                output.system_matrices.cpu().detach().squeeze().numpy()
            )
            control_matrices.append(
                output.control_matrices.cpu().detach().squeeze().numpy()
            )

        y_np = utils.denormalize(y_np, self.output_mean, self.output_std)

        return y_np.astype(np.float64), dict(
            system_matrices=np.array(system_matrices, dtype=np.float64),
            control_matrices=np.array(control_matrices, dtype=np.float64),
        )

    def save(
        self,
        file_path: Tuple[str, ...],
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> None:
        if (
            self.output_mean is None
            or self.output_std is None
            or self.control_mean is None
            or self.control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')

        torch.save(self.initializer.state_dict(), file_path[0])
        torch.save(self.predictor.state_dict(), file_path[1])
        with open(file_path[2], mode='w') as f:
            json.dump(
                {
                    'output_mean': self.output_mean.tolist(),
                    'output_std': self.output_std.tolist(),
                    'control_mean': self.control_mean.tolist(),
                    'control_std': self.control_std.tolist(),
                },
                f,
            )

    def load(self, file_path: Tuple[str, ...]) -> None:
        self.initializer.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        self.predictor.load_state_dict(
            torch.load(file_path[1], map_location=self.device_name)
        )
        with open(file_path[2], mode='r') as f:
            norm = json.load(f)
        self.output_mean = np.array(norm['output_mean'], dtype=np.float64)
        self.output_std = np.array(norm['output_std'], dtype=np.float64)
        self.control_mean = np.array(norm['control_mean'], dtype=np.float64)
        self.control_std = np.array(norm['control_std'], dtype=np.float64)

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'initializer.pth', 'predictor.pth', 'json'

    def get_parameter_count(self) -> int:
        init_count = sum(
            p.numel() for p in self.initializer.parameters() if p.requires_grad
        )
        predictor_count = sum(
            p.numel() for p in self.predictor.parameters() if p.requires_grad
        )
        return init_count + predictor_count


class UnconstrainedSwitchingLSTMModel(SwitchingLSTMBaseModel):
    def __init__(self, config: SwitchingLSTMBaseModelConfig):
        if config.switched_system_state_dim is None:
            state_dim = len(config.state_names)
        else:
            state_dim = config.switched_system_state_dim

        predictor = UnconstrainedSwitchingLSTM(
            control_dim=len(config.control_names),
            state_dim=state_dim,
            output_dim=len(config.state_names),
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            dropout=config.dropout,
        )
        super().__init__(config=config, predictor=predictor)


class UnconstrainedIdentityOutputSwitchingLSTMModel(SwitchingLSTMBaseModel):
    def __init__(self, config: SwitchingLSTMBaseModelConfig):
        predictor = UnconstrainedIdentityOutputSwitchingLSTM(
            control_dim=len(config.control_names),
            output_dim=len(config.state_names),
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            dropout=config.dropout,
        )
        super().__init__(config=config, predictor=predictor)


class StableSwitchingLSTMModel(SwitchingLSTMBaseModel):
    def __init__(self, config: SwitchingLSTMBaseModelConfig):
        if config.switched_system_state_dim is None:
            state_dim = len(config.state_names)
        else:
            state_dim = config.switched_system_state_dim

        predictor = StableSwitchingLSTM(
            control_dim=len(config.control_names),
            state_dim=state_dim,
            output_dim=len(config.state_names),
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            dropout=config.dropout,
        )
        super().__init__(config=config, predictor=predictor)


class StableIdentityOutputSwitchingLSTMModel(SwitchingLSTMBaseModel):
    def __init__(self, config: SwitchingLSTMBaseModelConfig):
        predictor = StableIdentityOutputSwitchingLSTM(
            control_dim=len(config.control_names),
            output_dim=len(config.state_names),
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            dropout=config.dropout,
        )
        super().__init__(config=config, predictor=predictor)
