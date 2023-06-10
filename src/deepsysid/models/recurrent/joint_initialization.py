import abc
import json
import logging
import time
from typing import Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn, optim
from torch.utils import data

from ...networks import loss, rnn
from ...networks.rnn import HiddenStateForwardModule
from ...tracker.base import BaseEventTracker
from ...tracker.event_data import TrackMetrics
from .. import base, utils
from ..base import DynamicIdentificationModelConfig
from ..datasets import RecurrentPredictorDataset

logger = logging.getLogger(__name__)


class JointInitializerRecurrentNetworkModelConfig(DynamicIdentificationModelConfig):
    recurrent_dim: int
    num_recurrent_layers: int
    bias: bool = True
    dropout: float
    sequence_length: int
    learning_rate: float
    batch_size: int
    epochs: int
    clip_gradient_norm: Optional[float] = None
    loss: Literal['mse', 'msge']


class JointInitializerRecurrentNetworkModel(
    base.NormalizedHiddenStateInitializerPredictorModel, metaclass=abc.ABCMeta
):
    CONFIG = JointInitializerRecurrentNetworkModelConfig

    def __init__(
        self,
        config: JointInitializerRecurrentNetworkModelConfig,
        initializer_rnn: HiddenStateForwardModule,
        predictor_rnn: HiddenStateForwardModule,
    ) -> None:
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)

        self.state_dim = len(config.state_names)

        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs = config.epochs

        if config.clip_gradient_norm is not None:
            self.clip_gradient_norm = config.clip_gradient_norm
        else:
            self.clip_gradient_norm = np.inf

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        elif config.loss == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse" or "msge"')

        self._initializer = initializer_rnn.to(self.device)
        self._predictor = predictor_rnn.to(self.device)

        self.optimizer_pred = optim.Adam(
            self._predictor.parameters(), lr=self.learning_rate
        )
        self.optimizer_init = optim.Adam(
            self._initializer.parameters(), lr=self.learning_rate
        )

    @property
    def initializer(self) -> HiddenStateForwardModule:
        return self._initializer

    @property
    def predictor(self) -> HiddenStateForwardModule:
        return self._predictor

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]] = None,
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> Optional[Dict[str, NDArray[np.float64]]]:
        epoch_losses_initializer = []
        epoch_losses_predictor = []

        self._predictor.train()
        self._initializer.train()

        self._control_mean, self._control_std = utils.mean_stddev(control_seqs)
        self._state_mean, self._state_std = utils.mean_stddev(state_seqs)

        control_seqs = [
            utils.normalize(control, self.control_mean, self.control_std)
            for control in control_seqs
        ]
        state_seqs = [
            utils.normalize(state, self.state_mean, self.state_std)
            for state in state_seqs
        ]

        predictor_dataset = RecurrentPredictorDataset(
            control_seqs, state_seqs, self.sequence_length
        )
        time_start = time.time()
        for i in range(self.epochs):
            data_loader = data.DataLoader(
                predictor_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_predictor_loss = 0.0
            total_initializer_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                self._predictor.zero_grad()
                # Initialize predictor with state of initializer network
                y_init, hx = self._initializer.forward(
                    batch['x0'].float().to(self.device)
                )
                # Predict and optimize
                y_pred, _ = self._predictor.forward(
                    batch['x'].float().to(self.device), hx=hx
                )
                y_true = batch['y'].float().to(self.device)
                batch_loss_predictor = self.loss.forward(y_pred, y_true)
                batch_loss_initializer = self.loss.forward(
                    y_init[:, -1, :], y_true[:, -1, :]
                )
                batch_loss = batch_loss_predictor + batch_loss_initializer
                total_predictor_loss += batch_loss_predictor.item()
                total_initializer_loss += batch_loss_initializer.item()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._predictor.parameters(), self.clip_gradient_norm
                )
                self.optimizer_pred.step()
            tracker(
                TrackMetrics(
                    f'Track loss step {i}', {'loss': float(total_predictor_loss)}
                )
            )
            logger.info(
                f'Epoch {i + 1}/{self.epochs} '
                f'- Epoch Loss (Initializer): {total_initializer_loss}'
            )
            logger.info(
                f'Epoch {i + 1}/{self.epochs} '
                f'- Epoch Loss (Predictor): {total_predictor_loss}'
            )
            epoch_losses_initializer.append([i, total_initializer_loss])
            epoch_losses_predictor.append([i, total_predictor_loss])

        time_end = time.time()
        time_total = time_end - time_start
        logger.info(f'Training time {time_total}s.')

        return dict(
            epoch_loss_initializer=np.array(epoch_losses_initializer, dtype=np.float64),
            epoch_loss_predictor=np.array(epoch_losses_predictor, dtype=np.float64),
            training_time=np.array([time_total], dtype=np.float64),
        )

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        x0: Optional[NDArray[np.float64]],
    ) -> Union[
        NDArray[np.float64], Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]
    ]:
        if (
            self.state_mean is None
            or self.state_std is None
            or self.control_mean is None
            or self.control_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        self._initializer.eval()
        self._predictor.eval()

        N, _ = control.shape

        initial_control = utils.normalize(
            initial_control, self.control_mean, self.control_std
        )
        initial_state = utils.normalize(initial_state, self.state_mean, self.state_std)
        control = utils.normalize(control, self.control_mean, self.control_std)

        with torch.no_grad():
            init_x = (
                torch.from_numpy(
                    np.hstack(
                        (
                            np.vstack((initial_control[1:], control[0][np.newaxis, :])),
                            initial_state,
                        )
                    )
                )
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            pred_x = torch.from_numpy(control).unsqueeze(0).float().to(self.device)

            _, hx = self._initializer.forward(init_x)
            y, _ = self._predictor.forward(pred_x, hx=hx)
            y_np: NDArray[np.float64] = (
                y.cpu()
                .detach()
                .reshape(shape=(N, self.state_dim))
                .numpy()
                .astype(np.float64)
            )

        y_np = utils.denormalize(y_np, self.state_mean, self.state_std)
        return y_np

    def save(
        self, file_path: Tuple[str, ...], tracker: BaseEventTracker = BaseEventTracker()
    ) -> None:
        if (
            self.state_mean is None
            or self.state_std is None
            or self.control_mean is None
            or self.control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')

        torch.save(self._initializer.state_dict(), file_path[0])
        torch.save(self._predictor.state_dict(), file_path[1])
        with open(file_path[2], mode='w') as f:
            json.dump(
                {
                    'state_mean': self.state_mean.tolist(),
                    'state_std': self.state_std.tolist(),
                    'control_mean': self.control_mean.tolist(),
                    'control_std': self.control_std.tolist(),
                },
                f,
            )

    def load(self, file_path: Tuple[str, ...]) -> None:
        self._initializer.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        self._predictor.load_state_dict(
            torch.load(file_path[1], map_location=self.device_name)
        )
        with open(file_path[2], mode='r') as f:
            norm = json.load(f)
        self._state_mean = np.array(norm['state_mean'], dtype=np.float64)
        self._state_std = np.array(norm['state_std'], dtype=np.float64)
        self._control_mean = np.array(norm['control_mean'], dtype=np.float64)
        self._control_std = np.array(norm['control_std'], dtype=np.float64)

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'initializer.pth', 'predictor.pth', 'json'

    def get_parameter_count(self) -> int:
        init_count = sum(
            p.numel() for p in self._initializer.parameters() if p.requires_grad
        )
        predictor_count = sum(
            p.numel() for p in self._predictor.parameters() if p.requires_grad
        )
        return init_count + predictor_count


class JointInitializerRNNModel(JointInitializerRecurrentNetworkModel):
    def __init__(self, config: JointInitializerRecurrentNetworkModelConfig) -> None:
        input_dim = len(config.control_names)
        output_dim = len(config.state_names)

        predictor = rnn.BasicRnn(
            input_dim=input_dim,
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            output_dim=output_dim,
            dropout=config.dropout,
            bias=config.bias,
        )

        initializer = rnn.BasicRnn(
            input_dim=input_dim + output_dim,
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            output_dim=output_dim,
            dropout=config.dropout,
            bias=config.bias,
        )

        super().__init__(config, initializer_rnn=initializer, predictor_rnn=predictor)


class JointInitializerGRUModel(JointInitializerRecurrentNetworkModel):
    def __init__(self, config: JointInitializerRecurrentNetworkModelConfig) -> None:
        input_dim = len(config.control_names)
        output_dim = len(config.state_names)

        predictor = rnn.BasicGRU(
            input_dim=input_dim,
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            output_dim=output_dim,
            dropout=config.dropout,
            bias=config.bias,
        )

        initializer = rnn.BasicGRU(
            input_dim=input_dim + output_dim,
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            output_dim=output_dim,
            dropout=config.dropout,
            bias=config.bias,
        )

        super().__init__(config, initializer_rnn=initializer, predictor_rnn=predictor)


class JointInitializerLSTMModel(JointInitializerRecurrentNetworkModel):
    def __init__(self, config: JointInitializerRecurrentNetworkModelConfig) -> None:
        input_dim = len(config.control_names)
        output_dim = len(config.state_names)

        predictor = rnn.BasicLSTM(
            input_dim=input_dim,
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            output_dim=output_dim,
            dropout=config.dropout,
            bias=config.bias,
        )

        initializer = rnn.BasicLSTM(
            input_dim=input_dim + output_dim,
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            output_dim=output_dim,
            dropout=config.dropout,
            bias=config.bias,
        )

        super().__init__(config, initializer_rnn=initializer, predictor_rnn=predictor)
