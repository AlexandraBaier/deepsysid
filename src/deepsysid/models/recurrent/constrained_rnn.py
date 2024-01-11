import copy
import json
import logging
import os
import time
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from numpy.typing import NDArray
from scipy.io import savemat
from torch import nn as nn
from torch import optim as optim
from torch.utils import data as data

from deepsysid.models import base, utils
from deepsysid.models.base import (
    DynamicIdentificationModelConfig,
    track_model_parameters,
)
from deepsysid.models.datasets import (
    RecurrentInitializerDataset,
    RecurrentPredictorDataset,
    RecurrentPredictorInitializerInitialDataset,
)

from ...cli.interface import DATASET_DIR_ENV_VAR, MODELS_DIR_ENV_VAR
from ...networks import loss, rnn
from ...networks.rnn import HiddenStateForwardModule
from ...pipeline.data_io import load_simulation_data
from ...pipeline.testing.base import TestSimulation
from ...pipeline.testing.io import split_simulations
from ...pipeline.model_io import save_model
from ...tracker.base import BaseEventTracker
from ...tracker.event_data import TrackArtifacts, TrackFigures, TrackMetrics, TrackParameters


logger = logging.getLogger('deepsysid.pipeline.training')


class RnnInitFlexibleNonlinearityConfig(DynamicIdentificationModelConfig):
    recurrent_dim: int
    bias: bool
    sequence_length: int
    learning_rate: float
    batch_size: int
    epochs_initializer: int
    epochs_predictor: int
    clip_gradient_norm: float
    loss: Literal['mse', 'msge']
    nonlinearity: str
    num_recurrent_layers_init: int
    dropout_init: float


class RnnInitFlexibleNonlinearity(base.NormalizedHiddenStateInitializerPredictorModel):
    CONFIG = RnnInitFlexibleNonlinearityConfig

    def __init__(self, config: RnnInitFlexibleNonlinearityConfig):
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)

        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)

        self.recurrent_dim = config.recurrent_dim

        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs_initializer = config.epochs_initializer
        self.epochs_predictor = config.epochs_predictor
        self.num_recurrent_layers_init = config.num_recurrent_layers_init
        self.dropout_init = config.dropout_init

        self.clip_gradient_norm = config.clip_gradient_norm

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        elif config.loss == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse" or "msge"')

        self._predictor = rnn.FixedDepthRnnFlexibleNonlinearity(
            input_dim=self.control_dim,
            recurrent_dim=self.recurrent_dim,
            output_dim=self.state_dim,
            bias=config.bias,
            nonlinearity=config.nonlinearity,
        ).to(self.device)

        self._initializer = rnn.BasicLSTM(
            input_dim=self.control_dim + self.state_dim,
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers_init,
            output_dim=[self.state_dim],
            dropout=self.dropout_init,
        ).to(self.device)

        self.optimizer_pred = optim.Adam(
            self._predictor.parameters(), lr=self.learning_rate
        )
        self.optimizer_init = optim.Adam(
            self._initializer.parameters(), lr=self.learning_rate
        )

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]] = None,
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> Dict[str, NDArray[np.float64]]:
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
                self._initializer.zero_grad()
                y, _ = self._initializer.forward(batch['x'].double().to(self.device))
                batch_loss = self.loss.forward(y, batch['y'].double().to(self.device))
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
                self._predictor.zero_grad()
                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(batch['x0'].double().to(self.device))
                # Predict and optimize
                y, _ = self._predictor.forward(
                    batch['x'].double().to(self.device), hx=hx
                )
                y = y.to(self.device)
                batch_loss = self.loss.forward(y, batch['y'].double().to(self.device))
                total_loss += batch_loss.item()
                batch_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self._predictor.parameters(), self.clip_gradient_norm
                )
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
        initial_x0: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        if (
            self.state_mean is None
            or self.state_std is None
            or self.control_mean is None
            or self.control_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        self._initializer.eval()
        self._predictor.eval()

        initial_control = utils.normalize(
            initial_control, self.control_mean, self.control_std
        )
        initial_state = utils.normalize(initial_state, self.state_mean, self.state_std)
        control = utils.normalize(control, self.control_mean, self.control_std)

        with torch.no_grad():
            init_x = (
                torch.from_numpy(np.hstack((initial_control[1:], initial_state[:-1])))
                .unsqueeze(0)
                .double()
                .to(self.device)
            )
            pred_x = torch.from_numpy(control).unsqueeze(0).double().to(self.device)

            _, hx = self._initializer.forward(init_x)
            y, _ = self._predictor.forward(pred_x, hx=hx)
            y_np: NDArray[np.float64] = (
                y.cpu().detach().squeeze().numpy().astype(np.float64)
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
        # technically parameter counts of both networks are equal
        init_count = sum(
            p.numel() for p in self._initializer.parameters() if p.requires_grad
        )
        predictor_count = sum(
            p.numel() for p in self._predictor.parameters() if p.requires_grad
        )
        return init_count + predictor_count

    @property
    def initializer(self) -> HiddenStateForwardModule:
        return copy.deepcopy(self._initializer)

    @property
    def predictor(self) -> HiddenStateForwardModule:
        return copy.deepcopy(self._predictor)


class LtiRnnInitConfig(DynamicIdentificationModelConfig):
    nx: int
    recurrent_dim: int
    num_recurrent_layers_init: int
    dropout: float
    sequence_length: int
    learning_rate: float
    batch_size: int
    epochs_initializer: int
    epochs_predictor: int
    clip_gradient_norm: float
    loss: Literal['mse', 'msge']
    nonlinearity: str


class LtiRnnInit(base.NormalizedHiddenStateInitializerPredictorModel):
    CONFIG = LtiRnnInitConfig

    def __init__(self, config: LtiRnnInitConfig):
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)

        self.nx = config.nx
        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)
        self.recurrent_dim = config.recurrent_dim

        self.recurrent_dim = config.recurrent_dim
        self.num_recurrent_layers_init = config.num_recurrent_layers_init
        self.dropout = config.dropout

        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs_initializer = config.epochs_initializer
        self.epochs_predictor = config.epochs_predictor

        self.clip_gradient_norm = config.clip_gradient_norm

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        elif config.loss == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse" or "msge"')

        self._predictor = rnn.LtiRnn(
            nx=self.nx,
            nu=self.control_dim,
            ny=self.state_dim,
            nw=self.recurrent_dim,
            nonlinearity=config.nonlinearity,
        ).to(self.device)

        self._initializer = rnn.BasicLSTM(
            input_dim=self.control_dim + self.state_dim,
            recurrent_dim=self.nx,
            num_recurrent_layers=self.num_recurrent_layers_init,
            output_dim=[self.state_dim],
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer_pred = optim.Adam(
            self._predictor.parameters(), lr=self.learning_rate
        )
        self.optimizer_init = optim.Adam(
            self._initializer.parameters(), lr=self.learning_rate
        )

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]] = None,
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> Dict[str, NDArray[np.float64]]:
        us = control_seqs
        ys = state_seqs

        self._predictor.train()
        self._initializer.train()

        self._control_mean, self._control_std = utils.mean_stddev(us)
        self._state_mean, self._state_std = utils.mean_stddev(ys)

        us = [
            utils.normalize(control, self.control_mean, self.control_std)
            for control in us
        ]
        ys = [utils.normalize(state, self.state_mean, self.state_std) for state in ys]

        initializer_dataset = RecurrentInitializerDataset(us, ys, self.sequence_length)

        initializer_loss = []
        time_start_init = time.time()
        for i in range(self.epochs_initializer):
            data_loader = data.DataLoader(
                initializer_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                self._initializer.zero_grad()
                y, _ = self._initializer.forward(batch['x'].double().to(self.device))
                batch_loss = self.loss.forward(y, batch['y'].double().to(self.device))
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_init.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_initializer}\t'
                f'Epoch Loss (Initializer): {total_loss}'
            )
            initializer_loss.append(total_loss)
        time_end_init = time.time()

        predictor_dataset = RecurrentPredictorDataset(us, ys, self.sequence_length)

        time_start_pred = time.time()
        predictor_loss: List[np.float64] = []
        gradient_norm: List[np.float64] = []
        last_index = 0
        for i in range(self.epochs_predictor):
            data_loader = data.DataLoader(
                predictor_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0
            max_grad = 0
            for batch_idx, batch in enumerate(data_loader):
                self._predictor.zero_grad()
                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(batch['x0'].double().to(self.device))
                # Predict and optimize
                y, _ = self._predictor.forward(
                    batch['x'].double().to(self.device), hx=hx
                )
                y = y.to(self.device)
                batch_loss = self.loss.forward(y, batch['y'].double().to(self.device))
                total_loss += batch_loss.item()
                batch_loss.backward()

                # gradient infos
                grads_norm = [
                    torch.linalg.norm(p.grad) for p in self._predictor.parameters()
                ]
                max_grad += max(grads_norm)
                torch.nn.utils.clip_grad_norm_(
                    self._predictor.parameters(), self.clip_gradient_norm
                )
                self.optimizer_pred.step()

                last_index = i

            logger.info(
                f'Epoch {i + 1}/{self.epochs_predictor}\t'
                f'Total Loss (Predictor): {total_loss:1f} \t'
                f'Max accumulated gradient norm: {max_grad:1f}'
            )
            predictor_loss.append(np.float64(total_loss))
            gradient_norm.append(np.float64(max_grad))

        time_end_pred = time.time()
        time_total_init = time_end_init - time_start_init
        time_total_pred = time_end_pred - time_start_pred
        logger.info(
            f'Training time for initializer {time_total_init}s '
            f'and for predictor {time_total_pred}s'
        )

        return dict(
            index=np.asarray(last_index),
            epoch_loss_initializer=np.asarray(initializer_loss),
            epoch_loss_predictor=np.asarray(predictor_loss),
            gradient_norm=np.asarray(gradient_norm),
            training_time_initializer=np.asarray(time_total_init),
            training_time_predictor=np.asarray(time_total_pred),
        )

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        x0: Optional[NDArray[np.float64]] = None,
        initial_x0: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        if (
            self.control_mean is None
            or self.control_std is None
            or self.state_mean is None
            or self.state_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        self._initializer.eval()
        self._predictor.eval()

        initial_u = initial_control
        initial_y = initial_state
        u = control

        initial_u = utils.normalize(initial_u, self.control_mean, self.control_std)
        initial_y = utils.normalize(initial_y, self.state_mean, self.state_std)
        u = utils.normalize(u, self.control_mean, self.control_std)

        with torch.no_grad():
            init_x = (
                torch.from_numpy(np.hstack((initial_u[1:], initial_y[:-1])))
                .unsqueeze(0)
                .double()
                .to(self.device)
            )
            pred_x = torch.from_numpy(u).unsqueeze(0).double().to(self.device)

            _, hx = self._initializer.forward(init_x)
            y, _ = self._predictor.forward(pred_x, hx=hx)
            y_np: NDArray[np.float64] = (
                y.cpu().detach().squeeze().numpy().astype(np.float64)
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
        # technically parameter counts of both networks are equal
        init_count = sum(
            p.numel() for p in self._initializer.parameters() if p.requires_grad
        )
        predictor_count = sum(
            p.numel() for p in self._predictor.parameters() if p.requires_grad
        )
        return init_count + predictor_count

    @property
    def initializer(self) -> HiddenStateForwardModule:
        return copy.deepcopy(self._initializer)

    @property
    def predictor(self) -> HiddenStateForwardModule:
        return copy.deepcopy(self._predictor)


class ConstrainedRnnConfig(DynamicIdentificationModelConfig):
    nx: int
    recurrent_dim: int
    gamma: float
    beta: float
    initial_decay_parameter: float
    decay_rate: float
    epochs_with_const_decay: int
    num_recurrent_layers_init: int
    dropout: float
    sequence_length: int
    learning_rate: float
    batch_size: int
    epochs_initializer: int
    epochs_predictor: int
    loss: Literal['mse']
    bias: bool
    nonlinearity: str
    clip_gradient_norm: Optional[float] = False
    initial_window_size: Optional[int] = None


class ConstrainedRnn(base.NormalizedHiddenStateInitializerPredictorModel):
    CONFIG = ConstrainedRnnConfig

    def __init__(self, config: ConstrainedRnnConfig):
        super().__init__(config)

        torch.set_default_dtype(torch.float64)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)

        self.nx = config.nx
        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)
        self.recurrent_dim = config.recurrent_dim
        if config.initial_window_size is not None:
            self.initial_window_size = config.initial_window_size
        else:
            self.initial_window_size = config.sequence_length

        self.initial_decay_parameter = config.initial_decay_parameter
        self.decay_rate = config.decay_rate
        self.epochs_with_const_decay = config.epochs_with_const_decay

        self.recurrent_dim = config.recurrent_dim
        self.num_recurrent_layers_init = config.num_recurrent_layers_init
        self.dropout = config.dropout

        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs_initializer = config.epochs_initializer
        self.epochs_predictor = config.epochs_predictor

        self.state_names = config.state_names
        self.control_names = config.control_names
        self.initial_state_names = config.initial_state_names

        self.clip_gradient_norm = config.clip_gradient_norm
        self.gamma = config.gamma
        self.bias = config.bias

        self.nl = retrieve_nonlinearity_class(config.nonlinearity)
        self.nonlinearity = config.nonlinearity

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse"')

        self._predictor = rnn.LtiRnnConvConstr(
            nx=self.nx,
            nu=self.control_dim,
            ny=self.state_dim,
            nw=self.recurrent_dim,
            gamma=config.gamma,
            beta=config.beta,
            bias=config.bias,
            nonlinearity=self.nl,
            device=self.device,
        ).to(self.device)

        self._initializer = rnn.BasicLSTM(
            input_dim=self.control_dim + self.state_dim,
            recurrent_dim=self.nx,
            num_recurrent_layers=self.num_recurrent_layers_init,
            output_dim=[self.state_dim],
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer_pred = optim.Adam(
            self._predictor.parameters(), lr=self.learning_rate
        )
        # self.optimizer_pred = optim.RMSprop(
        #     self._predictor.parameters(), lr=self.learning_rate
        # )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer_pred,
            factor=0.25,
            patience=self.epochs_with_const_decay,
            verbose=True,
            threshold=1e-5,
        )
        self.optimizer_init = optim.Adam(
            self._initializer.parameters(), lr=self.learning_rate
        )

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]] = None,
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> Dict[str, NDArray[np.float64]]:
        us = control_seqs
        ys = state_seqs
        self._predictor.initialize_lmi()
        self._predictor.to(self.device)

        self._control_mean, self._control_std = utils.mean_stddev(us)
        self._state_mean, self._state_std = utils.mean_stddev(ys)

        track_model_parameters(self, tracker)

        us = [
            utils.normalize(control, self._control_mean, self._control_std)
            for control in us
        ]
        ys = [utils.normalize(state, self._state_mean, self._state_std) for state in ys]

        initializer_dataset = RecurrentInitializerDataset(
            us, ys, self.sequence_length, self.initial_window_size
        )

        initializer_loss = []
        time_start_init = time.time()
        for i in range(self.epochs_initializer):
            self._initializer.train()
            data_loader = data.DataLoader(
                initializer_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                self._initializer.zero_grad()
                y, _ = self._initializer.forward(batch['x'].double().to(self.device))
                batch_loss = self.loss.forward(y, batch['y'].double().to(self.device))
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_init.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_initializer}\t'
                f'Epoch Loss (Initializer): {total_loss}'
            )
            tracker(
                TrackMetrics(
                    'track initializer loss',
                    {'epoch loss initializer': float(total_loss)},
                    i,
                )
            )
            initializer_loss.append(total_loss)
        time_end_init = time.time()
        predictor_dataset = RecurrentPredictorDataset(us, ys, self.sequence_length, self.initial_window_size)

        time_start_pred = time.time()
        t = self.initial_decay_parameter
        predictor_loss: List[np.float64] = []
        barrier_value: List[np.float64] = []
        gradient_norm: List[np.float64] = []
        backtracking_iter: List[np.float64] = []
        validation_loss: np.float64 = np.float64(100.0)

        for i in range(self.epochs_predictor):
            self._predictor.train()
            self._initializer.train()
            data_loader = data.DataLoader(
                predictor_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0
            max_grad = 0
            for batch_idx, batch in enumerate(data_loader):
                self._predictor.zero_grad(set_to_none=True)
                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(batch['x0'].double().to(self.device))
                y, _ = self._predictor.forward(
                    batch['x'].double().to(self.device), hx=hx
                )
                y = y.to(self.device)
                barrier = self._predictor.get_barriers(t).to(self.device)

                batch_loss = self.loss.forward(y, batch['y'].double().to(self.device))
                total_loss += batch_loss.item()
                (batch_loss + barrier).backward()

                if self.clip_gradient_norm:
                    torch.nn.utils.clip_grad_norm_(
                        parameters=self._predictor.parameters(),
                        max_norm=self.clip_gradient_norm,
                    )

                # gradient infos
                grads_norm = [
                    torch.linalg.norm(p.grad)
                    for p in filter(
                        lambda p: p.grad is not None, self._predictor.parameters()
                    )
                ]
                max_grad += max(grads_norm)

                # save old parameter set
                old_pars = [
                    par.clone().detach() for par in self._predictor.parameters()
                ]

                self.optimizer_pred.step()

                new_pars = [
                    par.clone().detach() for par in self._predictor.parameters()
                ]

                # perform backtracking line search if constraints are not satisfied
                max_iter = 100
                alpha = 0.5
                bls_iter = 0
                while not self._predictor.check_constr():
                    new_pars = [
                        alpha * old_par.clone() + (1 - alpha) * new_par
                        for old_par, new_par in zip(old_pars, new_pars)
                    ]
                    self._predictor.write_parameters(new_pars)

                    if bls_iter > max_iter - 1:
                        M = self._predictor.get_constraints()
                        logger.warning(
                            f'Epoch {i+1}/{self.epochs_predictor}\t'
                            f'max real eigenvalue of M: '
                            f'{(torch.max(torch.real(torch.linalg.eig(M)[0]))):1f}\t'
                            f'Backtracking line search exceeded maximum iteration. \t'
                            f'Constraints satisfied? {self._predictor.check_constr()}'
                        )
                        time_end_pred = time.time()
                        time_total_init = time_end_init - time_start_init
                        time_total_pred = time_end_pred - time_start_pred

                        return dict(
                            index=np.asarray(i),
                            epoch_loss_initializer=np.asarray(initializer_loss),
                            epoch_loss_predictor=np.asarray(predictor_loss),
                            barrier_value=np.asarray(barrier_value),
                            backtracking_iter=np.asarray(backtracking_iter),
                            gradient_norm=np.asarray(gradient_norm),
                            training_time_initializer=np.asarray(time_total_init),
                            training_time_predictor=np.asarray(time_total_pred),
                        )
                    bls_iter += 1

                tracker(
                    TrackMetrics(
                        'track loss, validation loss and barrier',
                        {
                            'batch loss predictor': float(
                                batch_loss.cpu().detach().numpy()
                            ),
                            'batch barrier predictor': float(
                                barrier.cpu().detach().numpy()
                            ),
                            'batch bls iter predictor': bls_iter,
                        },
                    )
                )

            validation_loss = np.float64(0.0)
            # skip validation for testing
            # since it loads validation data from file system.
            # This should be improved in the future.
            if "PYTEST_CURRENT_TEST" not in os.environ:
                validation_loss = self.validate()
                old_lr = {p_group['lr'] for p_group in self.optimizer_pred.param_groups}
                self.scheduler.step(validation_loss)
                # if learning rate changed by scheduler also reduce decay parameter
                if not old_lr == {
                    p_group['lr'] for p_group in self.optimizer_pred.param_groups
                }:
                    t = t * 1 / self.decay_rate
                    logger.info(f'Decay t by {self.decay_rate} \t' f't: {t:1f}')
                    tracker(
                        TrackMetrics(
                            'track decay and learning rate',
                            {
                                'learning rate': float(
                                    self.optimizer_pred.param_groups[0]['lr']
                                ),
                                'decay rate': float(t),
                            },
                            i,
                        )
                    )

                    if t <= 1e-7:
                        time_end_pred = time.time()
                        time_total_init = time_end_init - time_start_init
                        time_total_pred = time_end_pred - time_start_pred
                        logger.info(
                            f'Minimum decay rate {t:1f} is reached. Stop training.'
                        )
                        return dict(
                            index=np.asarray(i),
                            epoch_loss_initializer=np.asarray(initializer_loss),
                            epoch_loss_predictor=np.asarray(predictor_loss),
                            barrier_value=np.asarray(barrier_value),
                            backtracking_iter=np.asarray(backtracking_iter),
                            gradient_norm=np.asarray(gradient_norm),
                            training_time_initializer=np.asarray(time_total_init),
                            training_time_predictor=np.asarray(time_total_pred),
                        )

            tracker(
                TrackMetrics(
                    'track loss, validation loss and barrier',
                    {
                        'epoch barrier predictor': float(barrier),
                        'epoch loss predictor': float(total_loss / len(data_loader)),
                        'epoch validation loss': float(validation_loss),
                    },
                    i,
                )
            )
            if i % 50 == 0:
                result = utils.TrainingPrediction(
                    u=batch['x'][0, :, :],
                    zp=batch['y'][0, :, :],
                    zp_hat=y[0, :, :].cpu().detach().numpy(),
                )
                tracker(
                    TrackFigures(
                        f'Save training trajectory {i}',
                        result,
                        'training_trajectory.png',
                    )
                )

            logger.info(
                f'Epoch {i + 1}/{self.epochs_predictor}\t'
                f'Total Loss (Predictor): {total_loss/len(data_loader):1f} \t'
                f'Validation Loss: {validation_loss:1f} \t'
                f'Barrier: {barrier:1f}\t'
                f'Backtracking Line Search iteration: {bls_iter}\t'
                f'Max gradient norm: {(max_grad/len(data_loader)):1f}'
            )

            predictor_loss.append(np.float64(total_loss))
            barrier_value.append(barrier.cpu().detach().numpy())
            backtracking_iter.append(np.float64(bls_iter))
            gradient_norm.append(np.float64(max_grad))

        time_end_pred = time.time()
        time_total_init = time_end_init - time_start_init
        time_total_pred = time_end_pred - time_start_pred

        tracker(
            TrackMetrics(
                'Track training time as metric',
                {
                    'Training time':float(time_total_pred),
                    'Training time initializer':float(time_total_init)
                }
            )
        )
        tracker(
            TrackParameters(
                'Track training time as parameter',
                {
                    'Training time':time.strftime("%H:%M:%S", time.gmtime(float(time_total_pred))),
                    'Training time initializer':time.strftime("%H:%M:%S", time.gmtime(float(time_total_init)))
                 }
            )
        )

        logger.info(
            f'Training time for initializer {time_total_init}s '
            f'and for predictor {time_total_pred}s'
        )

        return dict(
            index=np.asarray(i),
            epoch_loss_initializer=np.asarray(initializer_loss),
            epoch_loss_predictor=np.asarray(predictor_loss),
            barrier_value=np.asarray(barrier_value),
            backtracking_iter=np.asarray(backtracking_iter),
            gradient_norm=np.asarray(gradient_norm),
            training_time_initializer=np.asarray(time_total_init),
            training_time_predictor=np.asarray(time_total_pred),
        )

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        x0: Optional[NDArray[np.float64]] = None,
        initial_x0: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        if (
            self._control_mean is None
            or self._control_std is None
            or self._state_mean is None
            or self._state_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        self._initializer.eval()
        self._predictor.eval()

        initial_u = initial_control
        initial_y = initial_state
        u = control

        N, nu = control.shape

        initial_u = utils.normalize(initial_u, self._control_mean, self._control_std)
        initial_y = utils.normalize(initial_y, self._state_mean, self._state_std)
        u = utils.normalize(u, self._control_mean, self._control_std)

        with torch.no_grad():
            init_x = (
                torch.from_numpy(np.hstack((initial_u[1:], initial_y[:-1])))
                .unsqueeze(0)
                .double()
                .to(self.device)
            )
            pred_x = torch.from_numpy(u).unsqueeze(0).double().to(self.device)

            _, hx = self._initializer.forward(init_x)
            y, _ = self._predictor.forward(pred_x, hx=hx)
            y_np: NDArray[np.float64] = (
                y.cpu().detach().squeeze().numpy().astype(np.float64)
            )

        y_np = utils.denormalize(y_np, self._state_mean, self._state_std).reshape(
            (N, self.state_dim)
        )
        return y_np

    def save(
        self,
        file_path: Tuple[str, ...],
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> None:
        if (
            self._state_mean is None
            or self._state_std is None
            or self._control_mean is None
            or self._control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')

        torch.save(self._initializer.state_dict(), file_path[0])
        torch.save(self._predictor.state_dict(), file_path[1])
        with open(file_path[2], mode='w') as f:
            json.dump(
                {
                    'state_mean': self._state_mean.tolist(),
                    'state_std': self._state_std.tolist(),
                    'control_mean': self._control_mean.tolist(),
                    'control_std': self._control_std.tolist(),
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
        # technically parameter counts of both networks are equal
        init_count = sum(
            p.numel() for p in self._initializer.parameters() if p.requires_grad
        )
        predictor_count = sum(
            p.numel() for p in self._predictor.parameters() if p.requires_grad
        )
        return init_count + predictor_count

    def validate(
        self,
        horizon_size: Optional[int] = None,
    ) -> np.float64:
        if isinstance(self.initial_state_names, List):
            initial_state_names = self.initial_state_names
        self._predictor.eval()
        self._initializer.eval()
        if horizon_size is None:
            horizon_size = self.sequence_length * 5

        (
            u_init_list,
            y_init_list,
            x0_init_list,
            x0_list,
            u_list,
            y_list,
        ) = get_split_validation_data(
            self.control_names,
            self.state_names,
            initial_state_names,
            self.initial_window_size,
            horizon_size,
        )

        us = utils.normalize(np.stack(u_list), self.control_mean, self.control_std)
        ys = utils.normalize(np.stack(y_list), self.state_mean, self.state_std)

        us_init = utils.normalize(
            np.stack(u_init_list), self.control_mean, self.control_std
        )
        ys_init = utils.normalize(
            np.stack(y_init_list), self.state_mean, self.state_std
        )
        uy_init = np.concatenate((us_init[:, 1:, :], ys_init[:, :-1, :]), axis=2)

        us_torch = torch.from_numpy(us).to(self.device).double()
        ys_torch = torch.from_numpy(ys).to(self.device).double()
        uy_init_torch = torch.from_numpy(uy_init).to(self.device).double()

        # evaluate model on data use sequence length x 5 as prediction horizon
        with torch.no_grad():
            _, hx = self._initializer.forward(uy_init_torch)
            ys_hat_torch, _ = self._predictor.forward(us_torch, hx=hx)
        return np.float64(
            self.loss.forward(ys_torch, ys_hat_torch.to(self.device))
            .cpu()
            .detach()
            .numpy()
        )

    @property
    def initializer(self) -> HiddenStateForwardModule:
        return copy.deepcopy(self._initializer)

    @property
    def predictor(self) -> HiddenStateForwardModule:
        return copy.deepcopy(self._predictor)


class HybridConstrainedRnnConfig(DynamicIdentificationModelConfig):
    nwu: int
    gamma: float
    A_lin: List
    B_lin: List
    C_lin: List
    D_lin: List
    alpha: float
    beta: float
    loss: Literal['mse']
    decay_rate: float
    sequence_length: List[int]
    learning_rate: float
    batch_size: int
    epochs_predictor: List[int]
    clip_gradient_norm: Optional[float]
    epochs_without_projection: int
    initial_decay_parameter: float
    extend_state: bool
    nonlinearity: str
    decay_rate_lr: Optional[int] = 4
    enforce_constraints_method: Optional[Literal['barrier', 'projection']] = None
    epochs_with_const_decay: Optional[int]
    initial_window_size: Optional[int] = None
    normalize_rnn: Optional[bool] = False
    optimizer: Literal['SCS', 'MOSEK'] = 'SCS'
    controller_feedback: Optional[Literal['cf', 'nf', 'signal']] = 'cf'
    regularization: Optional[bool] = False
    multiplier_type: Optional[Literal['diagonal', 'static_zf']] = 'diagonal'
    loop_trafo: Optional[bool] = False
    warmstart: Optional[bool] = False


class HybridConstrainedRnn(base.NormalizedHiddenStatePredictorModel):
    CONFIG = HybridConstrainedRnnConfig

    def __init__(self, config: HybridConstrainedRnnConfig):
        super().__init__(config)

        torch.set_default_dtype(torch.float64)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)
        self.normalize_rnn = config.normalize_rnn
        self.optimizer = config.optimizer
        self.loop_trafo = config.loop_trafo

        self.control_dim = len(config.control_names)  # external input
        self.state_dim = len(config.state_names)  # output

        self.control_names = config.control_names
        self.state_names = config.state_names
        self.initial_state_names = config.initial_state_names
        self.regularization = config.regularization
        self.multiplier_type = config.multiplier_type
        self.warmstart = config.warmstart

        self.controller_feedback = config.controller_feedback

        if config.initial_window_size is not None:
            self.initial_window_size = config.initial_window_size
        else:
            self.initial_window_size = config.sequence_length[0]

        self.nwu = config.nwu
        self.nzu = self.nwu

        nx = len(config.A_lin)
        self.extend_state = nx < self.nwu and config.extend_state
        if self.extend_state:
            self.e = self.nwu - nx
            A_lin_tilde = np.concatenate(
                [
                    np.concatenate(
                        [
                            np.array(config.A_lin, dtype=np.float64),
                            np.zeros(shape=(nx, self.e)),
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.e, nx)),
                            np.zeros(shape=(self.e, self.e)),
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
            B_lin_tilde = np.concatenate(
                [
                    np.array(config.B_lin, dtype=np.float64),
                    np.zeros(shape=(self.e, self.control_dim)),
                ],
                axis=0,
            )
            C_lin_tilde = np.concatenate(
                [
                    np.array(config.C_lin, dtype=np.float64),
                    np.zeros(shape=(self.state_dim, self.e)),
                ],
                axis=1,
            )
        else:
            self.e = 0
            A_lin_tilde = np.array(config.A_lin, dtype=np.float64)
            B_lin_tilde = np.array(config.B_lin, dtype=np.float64)
            C_lin_tilde = np.array(config.C_lin, dtype=np.float64)

        self.nx = A_lin_tilde.shape[0]
        self._nx = A_lin_tilde.shape[0]
        self.ny = C_lin_tilde.shape[0]

        self.initial_decay_parameter = config.initial_decay_parameter
        self.decay_rate = config.decay_rate
        if config.epochs_with_const_decay is not None:
            self.epochs_with_const_decay = config.epochs_with_const_decay

        self.clip_gradient_norm = config.clip_gradient_norm

        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs_predictor = config.epochs_predictor
        self.epochs_without_projection = config.epochs_without_projection
        self.enforce_constraints_method = config.enforce_constraints_method
        self.gamma = config.gamma
        self.decay_rate_lr = config.decay_rate_lr

        self.nl = retrieve_nonlinearity_class(config.nonlinearity)
        self.nonlinearity = config.nonlinearity

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse"')

        self._predictor = rnn.HybridLinearizationRnn(
            A_lin=A_lin_tilde,
            B_lin=B_lin_tilde,
            C_lin=C_lin_tilde,
            alpha=config.alpha,
            beta=config.beta,
            nwu=self.nwu,
            nzu=self.nzu,
            gamma=config.gamma,
            nonlinearity=self.nl,
            device=self.device,
            normalize_rnn=self.normalize_rnn,
            optimizer=self.optimizer,
            enforce_constraints_method=config.enforce_constraints_method,
            controller_feedback=self.controller_feedback,
            multiplier_type=self.multiplier_type,
            loop_trafo=self.loop_trafo,
        ).to(self.device)

        self.linear = rnn.Linear(
            torch.tensor(config.A_lin).double().to(self.device),
            torch.tensor(config.B_lin).double().to(self.device),
            torch.tensor(config.C_lin).double().to(self.device),
            torch.tensor([[0.0]]).double().to(self.device),
        ).to(self.device)

        self.init_optimizer_scheduler()

    def init_optimizer_scheduler(self) -> None:
        self.optimizer_pred = optim.Adam(
            self._predictor.parameters(), lr=self.learning_rate
        )

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]] = None,
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> Dict[str, NDArray[np.float64]]:
        us = control_seqs
        ys = state_seqs
        self._predictor.train()
        self._predictor.to(self.device)
        step = 0

        if isinstance(initial_seqs, List):
            x0s: List[NDArray[np.float64]] = initial_seqs

        n_batch = len(x0s)
        N, nx = x0s[0].shape
        _, nu = us[0].shape
        # _, xs = self.linear.forward(
        #     torch.from_numpy(np.stack(x0s)[:, 0, :])
        #     .reshape(n_batch, nx, 1)
        #     .to(self.device)
        #     .double(),
        #     torch.from_numpy(np.stack(us))
        #     .reshape(n_batch, N, nu, 1)
        #     .to(self.device)
        #     .double(),
        #     True,
        # )

        # self._control_mean, self._control_std = utils.mean_stddev(us)
        # self._state_mean, self._state_std = utils.mean_stddev(
        #     list(xs.reshape((-1, nx)).cpu().detach().numpy())
        # )

        ## normalization currently not supported -> set zero
        self._control_mean, self._control_std = (
            np.zeros(shape=(self._predictor.nwp,)),
            np.zeros(shape=(self._predictor.nwp,)),
        )
        self._state_mean, self._state_std = (
            np.zeros(shape=(self._predictor.nx,)),
            np.zeros(shape=(self._predictor.nx,)),
        )
        self._predictor._x_mean = (
            torch.zeros(size=(self._predictor.nx,)).double().to(self.device)
        )
        self._predictor._x_std = (
            torch.zeros(size=(self._predictor.nx,)).double().to(self.device)
        )
        x_mean = self._state_mean
        wp_mean = self._control_mean
        self._predictor._wp_mean = (
            torch.zeros(size=(self._predictor.nwp, 1)).double().to(self.device)
        )
        self._predictor._wp_std = (
            torch.zeros(size=(self._predictor.nwp, 1)).double().to(self.device)
        )

        track_model_parameters(self, tracker)

        if self.enforce_constraints_method is not None:
            self._predictor.project_parameters()

        self._predictor.set_lure_system()

        time_start_pred = time.time()
        predictor_loss: List[np.float64] = []
        barrier_value: List[np.float64] = []
        gradient_norm: List[np.float64] = []
        steps_without_projection = self.epochs_without_projection
        steps_without_loss_decrease = 0
        old_loss: torch.Tensor = torch.tensor(100.0)
        t = self.initial_decay_parameter

        for seq_len, epoch_predictor in zip(
            self.sequence_length, self.epochs_predictor
        ):
            predictor_dataset = RecurrentPredictorInitializerInitialDataset(
                us,
                ys,
                x0s,
                seq_len,
                x_mean,
                wp_mean,
                self.initial_window_size,
                self.normalize_rnn,
            )
            no_decrease_count: int = 0
            old_validation_loss = np.float64(0.0)

            # for debugging
            # torch.autograd.set_detect_anomaly(True)
            for i in range(epoch_predictor):
                data_loader = data.DataLoader(
                    predictor_dataset, self.batch_size, shuffle=True, drop_last=True
                )
                total_loss: torch.Tensor = torch.tensor(0.0).to(self.device)
                max_grad: List[np.float64] = list()
                backtracking_iter: List[int] = list()
                for batch_idx, batch in enumerate(data_loader):

                    def closure():
                        self._predictor.zero_grad(set_to_none=True)
                        if self.extend_state:
                            x0 = torch.concat(
                                [
                                    batch['x0'].double(),
                                    torch.zeros(size=(self.batch_size, self.e)),
                                ],
                                dim=1,
                            ).to(self.device)
                            x0_init = torch.concat(
                                [
                                    batch['x0_init'][:, 0, :].double(),
                                    torch.zeros(size=(self.batch_size, self.e)),
                                ],
                                dim=1,
                            ).to(self.device)
                        else:
                            x0_init = batch['x0_init'][:, 0, :].double().to(self.device)
                            x0 = batch['x0'].double().to(self.device)

                        # warmstart
                        if self.warmstart:
                            _, (_, x_rnns) = self._predictor.forward(
                                x_pred=batch['wp_init'].double().to(self.device),
                                hx=(x0_init, torch.zeros_like(x0_init).to(self.device)),
                            )
                            zp_hat, _ = self._predictor.forward(
                                x_pred=batch['wp'].double().to(self.device),
                                hx=(
                                    x0,
                                    x_rnns,
                                ),
                            )

                        else:
                        # Predict and optimize
                            zp_hat, _ = self._predictor.forward(
                                x_pred=batch['wp'].double().to(self.device),
                                hx=(
                                    x0,
                                    torch.zeros_like(x0).to(self.device),
                                ),
                            )

                        zp_hat = zp_hat.to(self.device)
                        batch_loss = self.loss.forward(
                            zp_hat, batch['zp'].double().to(self.device)
                        )

                        regularization = torch.tensor(0.0).to(self.device)
                        if self.regularization:
                            regularization = self._predictor.get_regularization(
                                torch.tensor(1e-1)
                            )
                            batch_loss += regularization

                        barrier = torch.tensor(0.0).to(self.device)
                        if self.enforce_constraints_method == 'barrier':
                            barrier = self._predictor.get_barriers(
                                torch.tensor(t, device=self.device)
                            )
                            (batch_loss + barrier).backward()
                        elif (
                            self.enforce_constraints_method == 'projection'
                            or self.enforce_constraints_method is None
                        ):
                            batch_loss.backward()

                        if self.clip_gradient_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                parameters=self._predictor.parameters(),
                                max_norm=self.clip_gradient_norm,
                            )

                        # gradient infos
                        grads_norm = [
                            torch.linalg.norm(p.grad)
                            for p in filter(
                                lambda p: p.grad is not None,
                                self._predictor.parameters(),
                            )
                        ]

                        max_grad.append(max(grads_norm).cpu().detach().numpy())

                        tracker(
                            TrackMetrics(
                                'track loss, validation loss and barrier',
                                {
                                    'batch loss predictor': float(
                                        batch_loss.cpu().detach().numpy()
                                    ),
                                    'batch barrier predictor': float(
                                        barrier.cpu().detach().numpy()
                                    ),
                                    'batch regularization': float(
                                        regularization.cpu().detach().numpy()
                                    ),
                                },
                            )
                        )

                        return batch_loss, barrier, zp_hat, regularization

                    # save old parameter set
                    old_pars = [
                        par.clone().detach() for par in self._predictor.parameters()
                    ]

                    (
                        batch_loss,
                        barrier,
                        zp_hat,
                        regularization,
                    ) = self.optimizer_pred.step(closure)

                    # early stopping if prediction is NaN
                    if torch.isnan(batch_loss):
                        logger.info('Stop training. Batch loss is None.')
                        time_end_pred = time.time()
                        time_total_pred = time_end_pred - time_start_pred
                        return dict(
                            index=np.asarray(i),
                            epoch_loss_predictor=np.asarray(predictor_loss),
                            barrier_value=np.asarray(barrier_value),
                            gradient_norm=np.asarray(gradient_norm),
                            training_time_predictor=np.asarray(time_total_pred),
                        )

                    total_loss += batch_loss

                    new_pars = [
                        par.clone().detach() for par in self._predictor.parameters()
                    ]

                    try:
                        self._predictor.set_lure_system()
                    except AssertionError as msg:
                        logger.warning(msg)

                    bls_iter = int(0)
                    if self.enforce_constraints_method == 'barrier':
                        max_iter = 100
                        alpha = 0.5
                        while not self._predictor.check_constraints():
                            new_pars = [
                                alpha * old_par.clone() + (1 - alpha) * new_par
                                for old_par, new_par in zip(old_pars, new_pars)
                            ]

                            self._predictor.write_parameters(new_pars)

                            self._predictor.set_lure_system()

                            # no feasible parameter set
                            if bls_iter > max_iter - 1:
                                logger.info(
                                    f'BLS did not find feasible parameter set'
                                    f'after {bls_iter} iterations.'
                                    f'Training is stopped.'
                                )
                                time_end_pred = time.time()
                                time_total_pred = time_end_pred - time_start_pred
                                return dict(
                                    index=np.asarray(i),
                                    epoch_loss_predictor=np.asarray(predictor_loss),
                                    barrier_value=np.asarray(barrier_value),
                                    gradient_norm=np.asarray(gradient_norm),
                                    training_time_predictor=np.asarray(time_total_pred),
                                )
                            bls_iter += 1
                    backtracking_iter.append(bls_iter)

                validation_loss = np.float64(0.0)
                # skip validation for testing
                # since it loads validation data from file system.
                # This should be improved in the future.
                if "PYTEST_CURRENT_TEST" not in os.environ:
                    validation_loss = self.validate(
                        seq_len,
                        horizon_size=4000,
                    )
                    e = old_validation_loss - validation_loss
                    if e < -1e-5 and i > 0:
                        no_decrease_count += 1

                    # update old validation loss
                    old_validation_loss = validation_loss.copy()
                    if no_decrease_count >= self.epochs_with_const_decay:
                        # decay learning rate
                        for p_group in self.optimizer_pred.param_groups:
                            p_group['lr'] = p_group['lr'] * 1 / self.decay_rate_lr
                        # decay regularization
                        t = t * 1 / self.decay_rate
                        logger.info(
                            f'Decay t by {self.decay_rate} \t t: {t:1f} \n'
                            f'Decay learning rate by {self.decay_rate_lr} \t lr: {self.optimizer_pred.param_groups[0]["lr"]:1f} '
                        )
                        tracker(
                            TrackMetrics(
                                'track decay and learning rate',
                                {
                                    'learning rate': float(
                                        self.optimizer_pred.param_groups[0]['lr']
                                    ),
                                    'decay rate': float(t),
                                },
                                i,
                            )
                        )
                        # stop training if decay rate is too small
                        if t <= 1e-7:
                            time_end_pred = time.time()
                            time_total_pred = time_end_pred - time_start_pred
                            logger.info(
                                f'Minimum decay rate {t:1f} is reached. Stop training.'
                            )
                            return dict(
                                index=np.asarray(i),
                                epoch_loss_predictor=np.asarray(predictor_loss),
                                barrier_value=np.asarray(barrier_value),
                                backtracking_iter=np.asarray(backtracking_iter),
                                gradient_norm=np.asarray(gradient_norm),
                                training_time_predictor=np.asarray(time_total_pred),
                            )
                        # reset counter
                        no_decrease_count = 0

                tracker(
                    TrackMetrics(
                        'track total loss, validation loss and barrier',
                        {
                            'epoch barrier predictor': float(barrier),
                            'epoch loss predictor': float(
                                total_loss / len(data_loader)
                            ),
                            'epoch validation loss': float(validation_loss),
                            'epoch regularization': float(regularization),
                        },
                        i,
                    )
                )

                # plot training trajectories
                if (
                    step == 0
                    or step == int(epoch_predictor / 2)
                    or step == epoch_predictor - 1
                ):
                    with torch.no_grad():
                        y_lin = (
                            self.linear.forward(
                                batch['x0'][0, :].reshape(
                                    1, self._predictor.nx - self.e, 1
                                ),
                                batch['wp'][0, :, : self._predictor.nwp]
                                .reshape(1, seq_len, self._predictor.nwp, 1)
                                .double()
                                .to(self.device),
                            )[0]
                            .cpu()
                            .detach()
                            .numpy()
                        )
                    result = utils.TrainingPrediction(
                        u=batch['wp'][0, :, : self._predictor.nwp],
                        zp=batch['zp'][0, :, :],
                        zp_hat=zp_hat[0, :, :].cpu().detach().numpy(),
                        y_lin=y_lin[:, :, 0],
                    )
                    tracker(
                        TrackFigures(
                            f'Save output plot at step: {step}',
                            result,
                            f'training_trajectory_{step}.png',
                        )
                    )

                logger.info(
                    f'Epoch {i + 1}/{epoch_predictor}\t'
                    f'Total Loss (Predictor): {total_loss:1f} \t'
                    f'Validation Loss: {validation_loss:1f} \t'
                    f'Regularization: {regularization:1f} \t'
                    f'Barrier: {barrier:1f}\t'
                    f'Backtracking Line Search iteration: {max(backtracking_iter)}\t'
                    f'Max accumulated gradient norm: {np.max(max_grad):1f}'
                )
                predictor_loss.append(np.float64(total_loss))
                barrier_value.append(barrier.cpu().detach().numpy())
                gradient_norm.append(np.float64(np.mean(max_grad)))

                if self.enforce_constraints_method == 'projection':
                    if (
                        i % steps_without_projection == 0
                        and i > 0
                        and not self._predictor.check_constraints()
                    ):
                        self._predictor.project_parameters()
                        self._predictor.set_lure_system()
                        if old_loss - total_loss.cpu().detach().numpy() < 0:
                            steps_without_loss_decrease += 1
                        if steps_without_loss_decrease > 10:
                            logger.info(
                                f'Batch {batch_idx}, Epoch {i+1}'
                                f'Loss is not decreasing after projection.\t'
                            )
                            time_end_pred = time.time()
                            time_total_pred = time_end_pred - time_start_pred
                            return dict(
                                index=np.asarray(i),
                                epoch_loss_predictor=np.asarray(predictor_loss),
                                barrier_value=np.asarray(barrier_value),
                                gradient_norm=np.asarray(gradient_norm),
                                training_time_predictor=np.asarray(time_total_pred),
                            )
                        old_loss = total_loss.cpu().detach().numpy()
                step += 1
            # reset optimizer and scheduler
            self.init_optimizer_scheduler()

        time_end_pred = time.time()
        time_total_pred = time_end_pred - time_start_pred

        tracker(
            TrackMetrics(
                'Track training time as metric',
                {'Training time':float(time_total_pred)}
            )
        )
        tracker(
            TrackParameters(
                'Track training time as parameter',
                {'Training time':time.strftime("%H:%M:%S", time.gmtime(float(time_total_pred)))}
            )
        )      

        logger.info(
            f'Training time for predictor (HH:MM:SS) '
            f'{time.strftime("%H:%M:%S", time.gmtime(float(time_total_pred)))}'
        )

        return dict(
            index=np.asarray(i),
            epoch_loss_predictor=np.asarray(predictor_loss),
            barrier_value=np.asarray(barrier_value),
            gradient_norm=np.asarray(gradient_norm),
            training_time_predictor=np.asarray(time_total_pred),
        )

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        x0: Optional[NDArray[np.float64]] = None,
        initial_x0: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:
        if (
            self._control_mean is None
            or self._control_std is None
            or self._state_mean is None
            or self._state_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        if self.extend_state:
            x_mean = np.hstack(
                (
                    self._state_mean,
                    np.ones(
                        shape=self.e,
                    ),
                )
            )

            x_std = np.hstack(
                (
                    self._state_std,
                    np.ones(
                        shape=self.e,
                    ),
                )
            )
        else:
            x_mean = self._state_mean
            x_std = self._state_std
        self._predictor._x_mean = torch.from_numpy(x_mean).double().to(self.device)
        self._predictor._x_std = torch.from_numpy(x_std).double().to(self.device)
        self._predictor._wp_mean = (
            torch.from_numpy(self._control_mean).double().to(self.device)
        )
        self._predictor._wp_std = (
            torch.from_numpy(self._control_std).double().to(self.device)
        )

        self._predictor.eval()
        self._predictor.set_lure_system()

        N, nu = control.shape
        N_init, _ = initial_control.shape
        if self.normalize_rnn:
            u = np.hstack(
                (
                    control,
                    np.broadcast_to(
                        np.hstack((self._control_mean, x_mean)),
                        (N, self._control_mean.shape[0] + x_mean.shape[0]),
                    ),
                )
            )
            u_init = np.hstack(
                (
                    initial_control[:, : self.control_dim].reshape(N_init, -1),
                    np.broadcast_to(
                        np.hstack((self._control_mean, x_mean)),
                        (N_init, self._control_mean.shape[0] + x_mean.shape[0]),
                    ),
                )
            )
        else:
            u = control
            u_init = initial_control[:, : self.control_dim].reshape(N_init, -1)
        with torch.no_grad():
            pred_x = torch.from_numpy(u).unsqueeze(0).double().to(self.device)
            x0_1_torch = torch.from_numpy(x0)
            initial_x0_1_torch = torch.from_numpy(initial_x0)

            if self.extend_state:
                x0_torch = (
                    torch.concat([x0_1_torch, torch.zeros(size=(self.e,))], dim=0)
                    .reshape(shape=(1, self.nx))
                    .double()
                    .to(self.device)
                )
                initial_x0_torch = (
                    torch.concat(
                        [initial_x0_1_torch, torch.zeros(size=(self.e,))], dim=0
                    )
                    .reshape(shape=(1, self.nx))
                    .double()
                    .to(self.device)
                )
            else:
                x0_torch = x0_1_torch.reshape(shape=(1, -1)).double().to(self.device)
                initial_x0_torch = (
                    initial_x0_1_torch.reshape(shape=(1, -1)).double().to(self.device)
                )

            if self.warmstart:
                # warmstart fo hidden rnn state
                _, (_, x_rnns) = self._predictor.forward(
                    torch.from_numpy(u_init).unsqueeze(0).double().to(self.device),
                    (initial_x0_torch, torch.zeros_like(initial_x0_torch)),
                )
                y, _ = self._predictor.forward(pred_x, hx=(x0_torch, x_rnns[:, :]))
            else:
                y, _ = self._predictor.forward(pred_x, hx=(x0_torch, torch.zeros_like(x0_torch).to(self.device)))
            y_np: NDArray[np.float64] = (
                y.cpu().detach().numpy().reshape((N, self.ny)).astype(np.float64)
            )

        return y_np

    def save(
        self,
        file_path: Tuple[str, ...],
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> None:
        if (
            self._state_mean is None
            or self._state_std is None
            or self._control_mean is None
            or self._control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')

        torch.save(self._predictor.state_dict(), file_path[1])
        omega, sys_block_matrix = self._predictor.set_lure_system()
        np_pars = {
            key: np.float64(value.cpu().detach().numpy())
            for key, value in self._predictor.state_dict().items()
        }
        savemat(
            file_path[3],
            {
                'omega': np.float64(omega),
                'P_cal': np.float64(sys_block_matrix),
                'predictor_parameter': np_pars,
            },
        )

        with open(file_path[2], mode='w') as f:
            json.dump(
                {
                    'state_mean': self._state_mean.tolist(),
                    'state_std': self._state_std.tolist(),
                    'control_mean': self._control_mean.tolist(),
                    'control_std': self._control_std.tolist(),
                },
                f,
            )
        tracker(
            TrackArtifacts(
                'Save omega and system matrices',
                {
                    'system parameters': file_path[3],
                    'torch parameter': file_path[1],
                    'normalization': file_path[2],
                },
            )
        )

    def load(self, file_path: Tuple[str, ...]) -> None:
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
        return (
            'initializer.pth',
            'predictor.pth',
            'json',
            'mat',
        )

    def get_parameter_count(self) -> int:
        # technically parameter counts of both networks are equal
        # init_count = sum(
        #     p.numel() for p in self._initializer.parameters() if p.requires_grad
        # )
        predictor_count = sum(
            p.numel() for p in self._predictor.parameters() if p.requires_grad
        )
        # return init_count + predictor_count
        return predictor_count

    def validate(
        self,
        sequence_length: int,
        horizon_size: Optional[int] = None,
    ) -> np.float64:
        if isinstance(self.initial_state_names, List):
            initial_state_names = self.initial_state_names

        self._predictor.eval()
        if horizon_size is None:
            horizon_size = sequence_length * 5

        (
            u_init_list,
            y_init_list,
            x0_init_list,
            x0_list,
            u_list,
            y_list,
        ) = get_split_validation_data(
            self.control_names,
            self.state_names,
            initial_state_names,
            self.initial_window_size,
            horizon_size,
        )

        if self.normalize_rnn:
            us_wp_mean_x_mean = [
                np.hstack(
                    (
                        u,
                        np.broadcast_to(
                            np.hstack((self.control_mean, self.state_mean)),
                            (
                                horizon_size,
                                self.control_mean.shape[0] + self.state_mean.shape[0],
                            ),
                        ),
                    )
                )
                for u in u_list
            ]
            us_wp_mean_x_mean_init = [
                np.hstack(
                    (
                        u_init,
                        np.broadcast_to(
                            np.hstack((self.control_mean, self.state_mean)),
                            (
                                self.initial_window_size,
                                self.control_mean.shape[0] + self.state_mean.shape[0],
                            ),
                        ),
                    )
                )
                for u_init in u_init_list
            ]
        else:
            us_wp_mean_x_mean = u_list
            us_wp_mean_x_mean_init = u_init_list
        us = np.stack(us_wp_mean_x_mean)
        us_init = np.stack(us_wp_mean_x_mean_init)
        ys = np.stack(y_list)

        ys_torch = torch.from_numpy(ys).to(self.device).double()
        us_torch = torch.from_numpy(us).to(self.device).double()
        us_init_torch = torch.from_numpy(us_init).to(self.device).double()

        if self._nx is None:
            raise ValueError('size of internal state is not set.')

        e = self._nx - x0_list[0].shape[0]
        if e > 0:
            x0_list = [np.concatenate((x0, np.zeros(shape=(e,)))) for x0 in x0_list]
            x0_init_list = [
                np.concatenate((x0_init, np.zeros(shape=(e,))))
                for x0_init in x0_init_list
            ]
        x0_torch = torch.from_numpy(np.stack(x0_list)).to(self.device).double()
        x0_init_torch = torch.from_numpy(np.stack(x0_list)).to(self.device).double()

        with torch.no_grad():
            if self.warmstart:
            # warmstart
                _, (_, x_rnns) = self.predictor(
                    us_init_torch, hx=(x0_init_torch, torch.zeros_like(x0_init_torch))
                )

                ys_hat_torch, _ = self.predictor(
                    us_torch,
                    hx=(
                        x0_torch,
                        x_rnns[:, :],
                    ),
                )
            else:
                ys_hat_torch, _ = self.predictor(
                    us_torch,
                    hx=(
                        x0_torch,
                        torch.zeros_like(x0_torch).to(self.device),
                    ),
                )
        # return validation error normalized over all states
        return np.float64(
            self.loss.forward(ys_torch, ys_hat_torch).cpu().detach().numpy()
        )

    @property
    def predictor(self) -> HiddenStateForwardModule:
        # this should only return a copy of the model,
        # however deepcopy does not support non leaf nodes,
        # which the parameters of the lure system are.
        return self._predictor

class InputConstrainedRnnConfig(DynamicIdentificationModelConfig):
    nwu: int
    gamma: float
    A_lin: List
    B_lin: List
    C_lin: List
    D_lin: List
    alpha: float
    beta: float
    loss: Literal['mse']
    decay_rate: float
    sequence_length: int
    learning_rate: float
    batch_size: int
    epochs_predictor: int
    clip_gradient_norm: Optional[float]
    initial_decay_parameter: float
    extend_state: bool
    nonlinearity: str
    decay_rate_lr: Optional[int] = 4
    epochs_with_const_decay: Optional[int] = 10
    initial_window_size: Optional[int] = 100
    optimizer: Literal['SCS', 'MOSEK'] = 'SCS'
    init_omega: Literal['zero','rand'] = 'zero'
    constraint_type: Literal['convex', 'non-convex'] = 'convex'
    multiplier_type: Optional[Literal['diagonal', 'static_zf']] = 'diagonal'

class InputConstrainedRnn(base.DynamicIdentificationModel):
    CONFIG = InputConstrainedRnnConfig

    def __init__(self, config: InputConstrainedRnnConfig):
        super().__init__(config)

        torch.set_default_dtype(torch.float64)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)
        self.optimizer = config.optimizer

        self.nd = len(config.control_names) # size of performance input
        self.nh = len(config.B_lin[0]) # input size of linear system
        self.ne = len(config.state_names) # size of performance output

        self.control_names = config.control_names
        self.state_names = config.state_names
        self.initial_state_names = config.initial_state_names
        self.initial_window_size = config.initial_window_size
        self.constraint_type = config.constraint_type
        self.init_omega = config.init_omega
        self.multiplier_type = config.multiplier_type
        
        self.nwu = config.nwu
        self.nzu = self.nwu

        nx = len(config.A_lin)
        self.extend_state = nx < self.nwu and config.extend_state
        if self.extend_state:
            self.e = self.nwu - nx
            A_lin_tilde = np.concatenate(
                [
                    np.concatenate(
                        [
                            np.array(config.A_lin, dtype=np.float64),
                            np.zeros(shape=(nx, self.e)),
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.e, nx)),
                            np.zeros(shape=(self.e, self.e)),
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
            )
            B_lin_tilde = np.concatenate(
                [
                    np.array(config.B_lin, dtype=np.float64),
                    np.zeros(shape=(self.e, self.nh)),
                ],
                axis=0,
            )
            C_lin_tilde = np.concatenate(
                [
                    np.array(config.C_lin, dtype=np.float64),
                    np.zeros(shape=(self.ne, self.e)),
                ],
                axis=1,
            )
        else:
            self.e = 0
            A_lin_tilde = np.array(config.A_lin, dtype=np.float64)
            B_lin_tilde = np.array(config.B_lin, dtype=np.float64)
            C_lin_tilde = np.array(config.C_lin, dtype=np.float64)

        self.nx = A_lin_tilde.shape[0]
        self.ny = C_lin_tilde.shape[0]

        self.initial_decay_parameter = config.initial_decay_parameter
        self.decay_rate = config.decay_rate
        self.epochs_with_const_decay = config.epochs_with_const_decay
        self.clip_gradient_norm = config.clip_gradient_norm

        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs_predictor = config.epochs_predictor
        self.gamma = config.gamma
        self.decay_rate_lr = config.decay_rate_lr

        self.nl = retrieve_nonlinearity_class(config.nonlinearity)
        self.nonlinearity = config.nonlinearity

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse"')
        
        self._predictor = rnn

        self._predictor = rnn.InputLinearizationRnn(
            A_lin=A_lin_tilde,
            B_lin=B_lin_tilde,
            C_lin=C_lin_tilde,
            nd = self.nd,
            alpha=config.alpha,
            beta=config.beta,
            nwu=self.nwu,
            nzu=self.nzu,
            gamma=config.gamma,
            nonlinearity=self.nl,
            device=self.device,
            optimizer=self.optimizer,
            multiplier_type=self.multiplier_type,
        ).to(self.device)

        if self.constraint_type == 'non-convex':
            self._predictor.project_parameters()
            init_sim_parameter, _ = self._predictor.set_lure_system()
            self._predictor = rnn.InputLinearizationNonConvexRnn(
                A_lin=A_lin_tilde,
                B_lin=B_lin_tilde,
                C_lin=C_lin_tilde,
                nd = self.nd,
                alpha=config.alpha,
                beta=config.beta,
                nwu=self.nwu,
                nzu=self.nzu,
                gamma=config.gamma,
                nonlinearity=self.nl,
                device=self.device,
                multiplier_type=self.multiplier_type,
                init_pars=init_sim_parameter
            )
            assert self._predictor.check_constraints()

        self.optimizer_pred = optim.Adam(
            self._predictor.parameters(), lr=self.learning_rate
        )

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]] = None,
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> Dict[str, NDArray[np.float64]]:
        us = control_seqs
        ys = state_seqs
        self._predictor.train()
        self._predictor.to(self.device)
        step = 0

        if isinstance(initial_seqs, List):
            x0s: List[NDArray[np.float64]] = initial_seqs

        track_model_parameters(self, tracker)

        # if self.enforce_constraints_method is not None:
        if self.constraint_type == 'convex':
            self._predictor.project_parameters()
        self._predictor.set_lure_system()
        
        time_start_pred = time.time()
        predictor_loss: List[np.float64] = []
        barrier_value: List[np.float64] = []
        gradient_norm: List[np.float64] = []
        t = self.initial_decay_parameter

        no_decrease_count: int = 0
        old_validation_loss = np.float64(0.0)
        for i in range(self.epochs_predictor):
            predictor_dataset = RecurrentPredictorInitializerInitialDataset(
                us,
                ys,
                x0s,
                self.sequence_length,
                np.zeros(shape=(self.nx,1)),
                np.zeros(shape=(self.nd,1)),
                self.initial_window_size,
            )
            

            # for debugging
            # torch.autograd.set_detect_anomaly(True)
            data_loader = data.DataLoader(
                predictor_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss: torch.Tensor = torch.tensor(0.0).to(self.device)
            max_grad: List[np.float64] = list()
            backtracking_iter: List[int] = list()
            for batch_idx, batch in enumerate(data_loader):

                def closure():
                    self._predictor.zero_grad(set_to_none=True)
                    if self.extend_state:
                        x0 = torch.concat(
                            [
                                batch['x0'].double(),
                                torch.zeros(size=(self.batch_size, self.e)),
                            ],
                            dim=1,
                        ).to(self.device)
                        x0_init = torch.concat(
                            [
                                batch['x0_init'][:, 0, :].double(),
                                torch.zeros(size=(self.batch_size, self.e)),
                            ],
                            dim=1,
                        ).to(self.device)
                    else:
                        x0_init = batch['x0_init'][:, 0, :].double().to(self.device)
                        x0 = batch['x0'].double().to(self.device)

                    # warmstart
                    _, (_, x_rnns) = self._predictor.forward(
                        x_pred=batch['wp_init'].double().to(self.device),
                        hx=(x0_init, torch.zeros_like(x0_init).to(self.device)),
                    )
                    zp_hat, _ = self._predictor.forward(
                        x_pred=batch['wp'].double().to(self.device),
                        hx=(
                            x0,
                            x_rnns,
                        ),
                    )

                    zp_hat = zp_hat.to(self.device)
                    batch_loss = self.loss.forward(
                        zp_hat, batch['zp'].double().to(self.device)
                    )

                    barrier = self._predictor.get_barriers(
                            torch.tensor(t, device=self.device)
                        )
                    (batch_loss + barrier).backward()

                    if self.clip_gradient_norm is not None:
                        torch.nn.utils.clip_grad_norm_(
                            parameters=self._predictor.parameters(),
                            max_norm=self.clip_gradient_norm,
                        )

                    # gradient infos
                    grads_norm = [
                        torch.linalg.norm(p.grad)
                        for p in filter(
                            lambda p: p.grad is not None,
                            self._predictor.parameters(),
                        )
                    ]

                    max_grad.append(max(grads_norm).cpu().detach().numpy())

                    tracker(
                        TrackMetrics(
                            'track loss, validation loss and barrier',
                            {
                                'batch loss predictor': float(
                                    batch_loss.cpu().detach().numpy()
                                ),
                                'batch barrier predictor': float(
                                    barrier.cpu().detach().numpy()
                                )
                            },
                        )
                    )

                    return batch_loss, barrier, zp_hat

                # save old parameter set
                old_pars = [
                    par.clone().detach() for par in self._predictor.parameters()
                ]

                (
                    batch_loss,
                    barrier,
                    zp_hat,
                ) = self.optimizer_pred.step(closure)

                # early stopping if prediction is NaN
                if torch.isnan(batch_loss):
                    logger.info('Stop training. Batch loss is None.')
                    time_end_pred = time.time()
                    time_total_pred = time_end_pred - time_start_pred
                    return dict(
                        index=np.asarray(i),
                        epoch_loss_predictor=np.asarray(predictor_loss),
                        barrier_value=np.asarray(barrier_value),
                        gradient_norm=np.asarray(gradient_norm),
                        training_time_predictor=np.asarray(time_total_pred),
                    )

                total_loss += batch_loss

                new_pars = [
                    par.clone().detach() for par in self._predictor.parameters()
                ]

                try:
                    self._predictor.set_lure_system()
                except AssertionError as msg:
                    logger.warning(msg)

                bls_iter = int(0)
                max_iter = 100
                alpha = 0.5
                while not self._predictor.check_constraints():
                    new_pars = [
                        alpha * old_par.clone() + (1 - alpha) * new_par
                        for old_par, new_par in zip(old_pars, new_pars)
                    ]

                    self._predictor.write_parameters(new_pars)

                    self._predictor.set_lure_system()

                    # no feasible parameter set
                    if bls_iter > max_iter - 1:
                        logger.info(
                            f'BLS did not find feasible parameter set'
                            f'after {bls_iter} iterations.'
                            f'Training is stopped.'
                        )
                        time_end_pred = time.time()
                        time_total_pred = time_end_pred - time_start_pred
                        return dict(
                            index=np.asarray(i),
                            epoch_loss_predictor=np.asarray(predictor_loss),
                            barrier_value=np.asarray(barrier_value),
                            gradient_norm=np.asarray(gradient_norm),
                            training_time_predictor=np.asarray(time_total_pred),
                        )
                    bls_iter += 1
                backtracking_iter.append(bls_iter)

            validation_loss = np.float64(0.0)
            # skip validation for testing
            # since it loads validation data from file system.
            # This should be improved in the future.
            if "PYTEST_CURRENT_TEST" not in os.environ:
                validation_loss = self.validate(
                    self.sequence_length,
                )
                e = old_validation_loss - validation_loss
                if e < -1e-5 and i > 0:
                    no_decrease_count += 1

                # update old validation loss
                old_validation_loss = validation_loss.copy()
                if no_decrease_count >= self.epochs_with_const_decay:
                    # decay learning rate
                    for p_group in self.optimizer_pred.param_groups:
                        p_group['lr'] = p_group['lr'] * 1 / self.decay_rate_lr
                    # decay regularization
                    t = t * 1 / self.decay_rate
                    logger.info(
                        f'Decay t by {self.decay_rate} \t t: {t:1f} \n'
                        f'Decay learning rate by {self.decay_rate_lr} \t lr: {self.optimizer_pred.param_groups[0]["lr"]:1f} '
                    )
                    tracker(
                        TrackMetrics(
                            'track decay and learning rate',
                            {
                                'learning rate': float(
                                    self.optimizer_pred.param_groups[0]['lr']
                                ),
                                'decay rate': float(t),
                            },
                            i,
                        )
                    )
                    # stop training if decay rate is too small
                    if t <= 1e-7:
                        time_end_pred = time.time()
                        time_total_pred = time_end_pred - time_start_pred
                        logger.info(
                            f'Minimum decay rate {t:1f} is reached. Stop training.'
                        )
                        return dict(
                            index=np.asarray(i),
                            epoch_loss_predictor=np.asarray(predictor_loss),
                            barrier_value=np.asarray(barrier_value),
                            backtracking_iter=np.asarray(backtracking_iter),
                            gradient_norm=np.asarray(gradient_norm),
                            training_time_predictor=np.asarray(time_total_pred),
                        )
                    # reset counter
                    no_decrease_count = 0

            tracker(
                TrackMetrics(
                    'track total loss, validation loss and barrier',
                    {
                        'epoch barrier predictor': float(barrier),
                        'epoch loss predictor': float(
                            total_loss / len(data_loader)
                        ),
                        'epoch validation loss': float(validation_loss),
                    },
                    i,
                )
            )

            # plot training trajectories
            if (
                step == 0
                or step == int(self.epochs_predictor / 2)
                or step == self.epochs_predictor - 1
            ):
                with torch.no_grad():
                    result = utils.TrainingPrediction(
                        u=batch['wp'][0, :, : self._predictor.nd],
                        zp=batch['zp'][0, :, :],
                        zp_hat=zp_hat[0, :, :].cpu().detach().numpy(),
                        # y_lin=y_lin[:, :, 0],
                    )
                    tracker(
                        TrackFigures(
                            f'Save output plot at step: {step}',
                            result,
                            f'training_trajectory_{step}.png',
                        )
                    )

            logger.info(
                f'Epoch {i + 1}/{self.epochs_predictor}\t'
                f'Total Loss (Predictor): {total_loss:1f} \t'
                f'Validation Loss: {validation_loss:1f} \t'
                f'Barrier: {barrier:1f}\t'
                f'Backtracking Line Search iteration: {max(backtracking_iter)}\t'
                f'Max accumulated gradient norm: {np.max(max_grad):1f}'
            )
            predictor_loss.append(np.float64(total_loss))
            barrier_value.append(barrier.cpu().detach().numpy())
            gradient_norm.append(np.float64(np.mean(max_grad)))

            
            step += 1

        time_end_pred = time.time()
        time_total_pred = time_end_pred - time_start_pred

        tracker(
            TrackMetrics(
                'Track training time as metric',
                {'Training time':float(time_total_pred)}
            )
        )
        tracker(
            TrackParameters(
                'Track training time as parameter',
                {'Training time':time.strftime("%H:%M:%S", time.gmtime(float(time_total_pred)))}
            )
        )      

        logger.info(
            f'Training time for predictor (HH:MM:SS) '
            f'{time.strftime("%H:%M:%S", time.gmtime(float(time_total_pred)))}'
        )

        return dict(
            index=np.asarray(i),
            epoch_loss_predictor=np.asarray(predictor_loss),
            barrier_value=np.asarray(barrier_value),
            gradient_norm=np.asarray(gradient_norm),
            training_time_predictor=np.asarray(time_total_pred),
        )

    
    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        x0: Optional[NDArray[np.float64]] = None,
        initial_x0: Optional[NDArray[np.float64]] = None,
    ) -> NDArray[np.float64]:

        self._predictor.eval()
        self._predictor.set_lure_system()

        N, nu = control.shape
        N_init, _ = initial_control.shape

        u = control
        u_init = initial_control[:, : self.nd].reshape(N_init, -1)
        with torch.no_grad():
            u_init_torch = torch.from_numpy(u_init).unsqueeze(0).double().to(self.device)
            pred_x = torch.from_numpy(u).unsqueeze(0).double().to(self.device)
            
            x0_1_torch = torch.from_numpy(x0)
            initial_x0_1_torch = torch.from_numpy(initial_x0)

            if self.extend_state:
                x0_torch = (
                    torch.concat([x0_1_torch, torch.zeros(size=(self.e,))], dim=0)
                    .reshape(shape=(1, self.nx))
                    .double()
                    .to(self.device)
                )
                initial_x0_torch = (
                    torch.concat(
                        [initial_x0_1_torch, torch.zeros(size=(self.e,))], dim=0
                    )
                    .reshape(shape=(1, self.nx))
                    .double()
                    .to(self.device)
                )
            else:
                x0_torch = x0_1_torch.reshape(shape=(1, -1)).double().to(self.device)
                initial_x0_torch = (
                    initial_x0_1_torch.reshape(shape=(1, -1)).double().to(self.device)
                )

            # warmstart
            _, (_, x_rnns) = self._predictor.forward(
                u_init_torch,
                hx=(
                    initial_x0_torch, 
                    torch.zeros_like(initial_x0_torch).to(self.device)
                ),
            )

            y, _ = self._predictor.forward(
                pred_x, 
                hx=(
                    x0_torch, 
                    x_rnns
                )
            )
            y_np: NDArray[np.float64] = (
                y.cpu().detach().numpy().reshape((N, self.ny)).astype(np.float64)
            )

        return y_np

    def save(
        self,
        file_path: Tuple[str, ...],
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> None:

        torch.save(self._predictor.state_dict(), file_path[1])
        sim_parameter, sys_block_matrix = self._predictor.set_lure_system()
        np_pars = {
            key: np.float64(value.cpu().detach().numpy())
            for key, value in self._predictor.state_dict().items()
        }
        savemat(
            file_path[3],
            {
                'omega': np.float64(sim_parameter.theta),
                'P_cal': np.float64(sys_block_matrix),
                'predictor_parameter': np_pars,
            },
        )

        tracker(
            TrackArtifacts(
                'Save omega and system matrices',
                {
                    'system parameters': file_path[3],
                    'torch parameter': file_path[1],
                },
            )
        )

    def load(self, file_path: Tuple[str, ...]) -> None:
        self._predictor.load_state_dict(
            torch.load(file_path[1], map_location=self.device_name)
        )
        # with open(file_path[2], mode='r') as f:
        #     norm = json.load(f)
        # self._state_mean = np.array(norm['state_mean'], dtype=np.float64)
        # self._state_std = np.array(norm['state_std'], dtype=np.float64)
        # self._control_mean = np.array(norm['control_mean'], dtype=np.float64)
        # self._control_std = np.array(norm['control_std'], dtype=np.float64)

    def get_file_extension(self) -> Tuple[str, ...]:
        return (
            'initializer.pth',
            'predictor.pth',
            'json',
            'mat',
        )

    def get_parameter_count(self) -> int:
        # technically parameter counts of both networks are equal
        # init_count = sum(
        #     p.numel() for p in self._initializer.parameters() if p.requires_grad
        # )
        predictor_count = sum(
            p.numel() for p in self._predictor.parameters() if p.requires_grad
        )
        # return init_count + predictor_count
        return predictor_count

    def validate(
        self,
        sequence_length: int,
        horizon_size: Optional[int] = None,
    ) -> np.float64:
        if isinstance(self.initial_state_names, List):
            initial_state_names = self.initial_state_names

        self._predictor.eval()
        if horizon_size is None:
            horizon_size = sequence_length * 5

        (
            u_init_list,
            y_init_list,
            x0_init_list,
            x0_list,
            u_list,
            y_list,
        ) = get_split_validation_data(
            self.control_names,
            self.state_names,
            initial_state_names,
            self.initial_window_size,
            horizon_size,
        )

        us_init = np.stack(u_init_list)
        us = np.stack(u_list)
        ys = np.stack(y_list)

        us_init_torch = torch.from_numpy(us_init).to(self.device).double()
        ys_torch = torch.from_numpy(ys).to(self.device).double()
        us_torch = torch.from_numpy(us).to(self.device).double()


        e = self.nx - x0_list[0].shape[0]
        if e > 0:
            x0_list = [np.concatenate((x0, np.zeros(shape=(e,)))) for x0 in x0_list]
            x0_init_list = [
                np.concatenate((x0_init, np.zeros(shape=(e,))))
                for x0_init in x0_init_list
            ]
        x0_torch = torch.from_numpy(np.stack(x0_list)).to(self.device).double()
        x0_init_torch = torch.from_numpy(np.stack(x0_init_list)).to(self.device).double()

        with torch.no_grad():

            _, (_, x_rnns) = self._predictor.forward(
                us_init_torch,
                hx=(
                    x0_init_torch, 
                    torch.zeros_like(x0_init_torch).to(self.device)
                ),
            )
            ys_hat_torch, _ = self.predictor(
                us_torch,
                hx=(
                    x0_torch,
                    x_rnns,
                ),
            )
        # return validation error normalized over all states
        return np.float64(
            self.loss.forward(ys_torch, ys_hat_torch).cpu().detach().numpy()
        )

    @property
    def predictor(self) -> HiddenStateForwardModule:
        # this should only return a copy of the model,
        # however deepcopy does not support non leaf nodes,
        # which the parameters of the lure system are.
        return self._predictor



    

def get_split_validation_data(
    control_names: List[str],
    state_names: List[str],
    initial_state_names: List[str],
    sequence_length: int,
    horizon_size: int,
) -> Tuple[
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
    List[NDArray[np.float64]],
]:

    dataset_directory = os.path.join(
        os.path.expanduser(os.environ[DATASET_DIR_ENV_VAR]), 'processed', 'validation'
    )
    us, ys, x0s = load_simulation_data(
        directory=dataset_directory,
        control_names=control_names,
        state_names=state_names,
        initial_state_names=initial_state_names,
    )
    simulations = [TestSimulation(u, y, x0, '-') for u, y, x0 in zip(us, ys, x0s)]

    u_list: List[NDArray[np.float64]] = list()
    y_list: List[NDArray[np.float64]] = list()
    x0_list: List[NDArray[np.float64]] = list()
    u_init_list: List[NDArray[np.float64]] = list()
    y_init_list: List[NDArray[np.float64]] = list()
    x0_init_list: List[NDArray[np.float64]] = list()
    for sample in split_simulations(sequence_length, horizon_size, simulations):
        x0_init_list.append(sample.initial_x0)
        u_init_list.append(sample.initial_control)
        y_init_list.append(sample.initial_state)
        u_list.append(sample.true_control)
        y_list.append(sample.true_state)
        x0_list.append(sample.x0)

    return (u_init_list, y_init_list, x0_init_list, x0_list, u_list, y_list)


def retrieve_nonlinearity_class(nonlinearity: str) -> nn.Module:
    nl: nn.Module = nn.ReLU()
    if nonlinearity == 'Tanh':
        nl = nn.Tanh()
    return nl
