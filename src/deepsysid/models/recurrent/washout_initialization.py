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
from .. import base, utils
from ..base import DynamicIdentificationModelConfig
from ..datasets import RecurrentInitializerPredictorDataset

logger = logging.getLogger(__name__)


class WashoutInitializerRecurrentNetworkModelConfig(DynamicIdentificationModelConfig):
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


class WashoutInitializerRecurrentNetworkModel(base.NormalizedControlStateModel):
    CONFIG = WashoutInitializerRecurrentNetworkModelConfig

    def __init__(
        self,
        config: WashoutInitializerRecurrentNetworkModelConfig,
        network: HiddenStateForwardModule,
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

        self.network = network

        self.optimizer = optim.Adam(self.network.parameters(), lr=self.learning_rate)

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]] = None,
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> Optional[Dict[str, NDArray[np.float64]]]:
        epoch_losses = []

        self.network.train()

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

        dataset = RecurrentInitializerPredictorDataset(
            control_seqs=control_seqs,
            state_seqs=state_seqs,
            sequence_length=self.sequence_length,
        )

        time_start = time.time()
        for epoch_idx in range(self.epochs):
            data_loader = data.DataLoader(
                dataset, self.batch_size, shuffle=True, drop_last=True
            )
            epoch_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                self.network.zero_grad()
                _, hx = self.network.forward(
                    batch['control_window'].float().to(self.device)
                )
                y, _ = self.network.forward(
                    batch['control_horizon'].float().to(self.device)
                )
                batch_loss = self.loss.forward(
                    y, batch['state_horizon'].float().to(self.device)
                )
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.network.parameters(), self.clip_gradient_norm
                )
                self.optimizer.step()

            logger.info(
                f'Epoch {epoch_idx + 1}/{self.epochs} '
                f'- Epoch Loss (Predictor): {epoch_loss}'
            )
            epoch_losses.append([epoch_idx, epoch_loss])

        time_end = time.time()
        time_total = time_end - time_start
        logger.info(f'Training time is {time_total}s.')

        return dict(
            epoch_loss=np.array(epoch_losses, dtype=np.float64),
            training_time=np.array([time_total], dtype=np.float64),
        )

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        x0: Optional[NDArray[np.float64]] = None,
        initial_x0: Optional[NDArray[np.float64]] = None,
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

        self.network.eval()

        n_steps = control.shape[0]

        initial_control = utils.normalize(
            initial_control, self.control_mean, self.control_std
        )
        control = utils.normalize(control, self.control_mean, self.control_std)

        with torch.no_grad():
            x_init = (
                torch.from_numpy(initial_control).unsqueeze(0).float().to(self.device)
            )
            x_pred = torch.from_numpy(control).unsqueeze(0).float().to(self.device)

            _, hx = self.network.forward(x_init)
            y, _ = self.network.forward(x_pred, hx)
            y_np: NDArray[np.float64] = (
                y.cpu()
                .detach()
                .reshape((n_steps, self.state_dim))
                .numpy()
                .astype(np.float64)
            )

        y_np = utils.denormalize(y_np, self.state_mean, self.state_std)
        return y_np

    def save(
        self,
        file_path: Tuple[str, ...],
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> None:
        if (
            self.state_mean is None
            or self.state_std is None
            or self.control_mean is None
            or self.control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')

        torch.save(self.network.state_dict(), file_path[0])
        with open(file_path[1], mode='w') as f:
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
        self.network.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        with open(file_path[1], mode='r') as f:
            norm = json.load(f)
        self._state_mean = np.array(norm['state_mean'], dtype=np.float64)
        self._state_std = np.array(norm['state_std'], dtype=np.float64)
        self._control_mean = np.array(norm['control_mean'], dtype=np.float64)
        self._control_std = np.array(norm['control_std'], dtype=np.float64)

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'pth', 'json'

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.network.parameters() if p.requires_grad)


class WashoutInitializerRNNModel(WashoutInitializerRecurrentNetworkModel):
    def __init__(self, config: WashoutInitializerRecurrentNetworkModelConfig) -> None:
        input_dim = len(config.control_names)
        output_dim = len(config.state_names)

        network = rnn.BasicRnn(
            input_dim=input_dim,
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            output_dim=output_dim,
            dropout=config.dropout,
            bias=config.bias,
        )

        super().__init__(config=config, network=network)


class WashoutInitializerGRUModel(WashoutInitializerRecurrentNetworkModel):
    def __init__(self, config: WashoutInitializerRecurrentNetworkModelConfig) -> None:
        input_dim = len(config.control_names)
        output_dim = len(config.state_names)

        network = rnn.BasicGRU(
            input_dim=input_dim,
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            output_dim=output_dim,
            dropout=config.dropout,
            bias=config.bias,
        )

        super().__init__(config=config, network=network)


class WashoutInitializerLSTMModel(WashoutInitializerRecurrentNetworkModel):
    def __init__(self, config: WashoutInitializerRecurrentNetworkModelConfig) -> None:
        input_dim = len(config.control_names)
        output_dim = len(config.state_names)

        network = rnn.BasicLSTM(
            input_dim=input_dim,
            recurrent_dim=config.recurrent_dim,
            num_recurrent_layers=config.num_recurrent_layers,
            output_dim=output_dim,
            dropout=config.dropout,
            bias=config.bias,
        )

        super().__init__(config=config, network=network)
