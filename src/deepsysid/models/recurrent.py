import copy
import json
import logging
import time
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from numpy.typing import NDArray

from ..networks import loss, rnn
from ..networks.rnn import HiddenStateForwardModule
from . import base, utils
from .base import DynamicIdentificationModelConfig
from .datasets import RecurrentInitializerDataset, RecurrentPredictorDataset

logger = logging.getLogger('deepsysid.pipeline.training')


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
                y, _ = self._initializer.forward(batch['x'].float().to(self.device))
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
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
        for i in range(self.epochs_predictor):
            data_loader = data.DataLoader(
                predictor_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0
            max_grad = 0
            for batch_idx, batch in enumerate(data_loader):
                self._predictor.zero_grad()
                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(batch['x0'].float().to(self.device))
                # Predict and optimize
                y, _ = self._predictor.forward(
                    batch['x'].float().to(self.device), hx=hx
                )
                y = y.to(self.device)
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
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
            index=np.asarray(i),
            initializer_loss=np.asarray(initializer_loss),
            predictor_loss=np.asarray(predictor_loss),
            gradient_norm=np.asarray(gradient_norm),
            training_time_init=np.asarray(time_total_init),
            training_time_pred=np.asarray(time_total_pred),
        )

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
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
                .float()
                .to(self.device)
            )
            pred_x = torch.from_numpy(u).unsqueeze(0).float().to(self.device)

            _, hx = self._initializer.forward(init_x)
            y, _ = self._predictor.forward(pred_x, hx=hx)
            y_np: NDArray[np.float64] = (
                y.cpu().detach().squeeze().numpy().astype(np.float64)
            )

        y_np = utils.denormalize(y_np, self.state_mean, self.state_std)
        return y_np

    def save(self, file_path: Tuple[str, ...]) -> None:
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
    loss: Literal['mse', 'msge']
    log_min_max_real_eigenvalues: Optional[bool] = False


class ConstrainedRnn(base.NormalizedHiddenStateInitializerPredictorModel):
    CONFIG = ConstrainedRnnConfig

    def __init__(self, config: ConstrainedRnnConfig):
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)

        self.nx = config.nx
        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)
        self.recurrent_dim = config.recurrent_dim

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

        self.log_min_max_real_eigenvalues = config.log_min_max_real_eigenvalues

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        elif config.loss == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse" or "msge"')

        self._predictor = rnn.LtiRnnConvConstr(
            nx=self.nx,
            nu=self.control_dim,
            ny=self.state_dim,
            nw=self.recurrent_dim,
            gamma=config.gamma,
            beta=config.beta,
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
    ) -> Dict[str, NDArray[np.float64]]:
        us = control_seqs
        ys = state_seqs
        self._predictor.initialize_lmi()
        self._predictor.to(self.device)
        self._predictor.train()
        self._initializer.train()

        self._control_mean, self._control_std = utils.mean_stddev(us)
        self._state_mean, self._state_std = utils.mean_stddev(ys)

        us = [
            utils.normalize(control, self._control_mean, self._control_std)
            for control in us
        ]
        ys = [utils.normalize(state, self._state_mean, self._state_std) for state in ys]

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
                y, _ = self._initializer.forward(batch['x'].float().to(self.device))
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
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
        t = self.initial_decay_parameter
        predictor_loss: List[np.float64] = []
        min_eigenvalue: List[np.float64] = []
        max_eigenvalue: List[np.float64] = []
        barrier_value: List[np.float64] = []
        gradient_norm: List[np.float64] = []
        backtracking_iter: List[np.float64] = []

        for i in range(self.epochs_predictor):
            data_loader = data.DataLoader(
                predictor_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0
            max_grad = 0
            for batch_idx, batch in enumerate(data_loader):
                self._predictor.zero_grad()
                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(batch['x0'].float().to(self.device))
                # Predict and optimize
                y, _ = self._predictor.forward(
                    batch['x'].float().to(self.device), hx=hx
                )
                y = y.to(self.device)
                barrier = self._predictor.get_barrier(t).to(self.device)
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_loss += batch_loss.item()
                (batch_loss + barrier).backward()

                # gradient infos
                grads_norm = [
                    torch.linalg.norm(p.grad) for p in self._predictor.parameters()
                ]
                max_grad += max(grads_norm)

                # save old parameter set
                old_pars = [
                    par.clone().detach() for par in self._predictor.parameters()
                ]

                self.optimizer_pred.step()

                # perform backtracking line search if constraints are not satisfied
                max_iter = 100
                alpha = 0.5
                bls_iter = 0
                while not self._predictor.check_constr():
                    for old_par, new_par in zip(old_pars, self._predictor.parameters()):
                        new_par.data = (
                            alpha * old_par.clone() + (1 - alpha) * new_par.data
                        )

                    if bls_iter > max_iter - 1:
                        M = self._predictor.get_constraints()
                        logger.warning(
                            f'Epoch {i+1}/{self.epochs_predictor}\t'
                            f'max real eigenvalue of M: '
                            f'{(torch.max(torch.real(torch.linalg.eig(M)[0]))):1f}\t'
                            f'Backtracking line search exceeded maximum iteration.'
                        )
                        return dict(
                            index=np.asarray(i),
                            initializer_loss=np.asarray(initializer_loss),
                            predictor_loss=np.asarray(predictor_loss),
                            barrier_value=np.asarray(barrier_value),
                            backtracking_iter=np.asarray(backtracking_iter),
                            gradient_norm=np.asarray(gradient_norm),
                            max_eigenvalue=np.asarray(max_eigenvalue),
                            min_eigenvalue=np.asarray(min_eigenvalue),
                        )
                    bls_iter += 1

            # decay t following the idea of interior point methods
            if i % self.epochs_with_const_decay == 0 and i != 0:
                t = t * 1 / self.decay_rate
                logger.info(f'Decay t by {self.decay_rate} \t' f't: {t:1f}')

            min_ev = np.float64('inf')
            max_ev = np.float64('inf')
            if self.log_min_max_real_eigenvalues:
                min_ev, max_ev = self._predictor.get_min_max_real_eigenvalues()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_predictor}\t'
                f'Total Loss (Predictor): {total_loss:1f} \t'
                f'Barrier: {barrier:1f}\t'
                f'Backtracking Line Search iteration: {bls_iter}\t'
                f'Max accumulated gradient norm: {max_grad:1f}'
            )
            predictor_loss.append(np.float64(total_loss))
            barrier_value.append(barrier.cpu().detach().numpy())
            backtracking_iter.append(np.float64(bls_iter))
            gradient_norm.append(np.float64(max_grad))
            max_eigenvalue.append(np.float64(max_ev))
            min_eigenvalue.append(np.float64(min_ev))

        time_end_pred = time.time()
        time_total_init = time_end_init - time_start_init
        time_total_pred = time_end_pred - time_start_pred
        logger.info(
            f'Training time for initializer {time_total_init}s '
            f'and for predictor {time_total_pred}s'
        )

        return dict(
            index=np.asarray(i),
            initializer_loss=np.asarray(initializer_loss),
            predictor_loss=np.asarray(predictor_loss),
            barrier_value=np.asarray(barrier_value),
            backtracking_iter=np.asarray(backtracking_iter),
            gradient_norm=np.asarray(gradient_norm),
            max_eigenvalue=np.asarray(max_eigenvalue),
            min_eigenvalue=np.asarray(min_eigenvalue),
            training_time_init=np.asarray(time_total_init),
            training_time_pred=np.asarray(time_total_pred),
        )

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
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

        initial_u = utils.normalize(initial_u, self._control_mean, self._control_std)
        initial_y = utils.normalize(initial_y, self._state_mean, self._state_std)
        u = utils.normalize(u, self._control_mean, self._control_std)

        with torch.no_grad():
            init_x = (
                torch.from_numpy(np.hstack((initial_u[1:], initial_y[:-1])))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            pred_x = torch.from_numpy(u).unsqueeze(0).float().to(self.device)

            _, hx = self._initializer.forward(init_x)
            y, _ = self._predictor.forward(pred_x, hx=hx)
            y_np: NDArray[np.float64] = (
                y.cpu().detach().squeeze().numpy().astype(np.float64)
            )

        y_np = utils.denormalize(y_np, self._state_mean, self._state_std)
        return y_np

    def save(self, file_path: Tuple[str, ...]) -> None:
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

    @property
    def initializer(self) -> HiddenStateForwardModule:
        return copy.deepcopy(self._initializer)

    @property
    def predictor(self) -> HiddenStateForwardModule:
        return copy.deepcopy(self._predictor)


class LSTMInitModelConfig(DynamicIdentificationModelConfig):
    recurrent_dim: int
    num_recurrent_layers: int
    dropout: float
    sequence_length: int
    learning_rate: float
    batch_size: int
    epochs_initializer: int
    epochs_predictor: int
    loss: Literal['mse', 'msge']


class LSTMInitModel(base.NormalizedHiddenStateInitializerPredictorModel):
    CONFIG = LSTMInitModelConfig

    def __init__(self, config: LSTMInitModelConfig):
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)

        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)

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

        self._predictor = rnn.BasicLSTM(
            input_dim=self.control_dim,
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=[self.state_dim],
            dropout=self.dropout,
        ).to(self.device)

        self._initializer = rnn.BasicLSTM(
            input_dim=self.control_dim + self.state_dim,
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
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
    ) -> Dict[str, NDArray[np.float64]]:
        epoch_losses_initializer = []
        epoch_losses_predictor = []

        self._predictor.train()
        self._initializer.train()

        self._control_mean, self._control_std = utils.mean_stddev(control_seqs)
        self._state_mean, self._state_std = utils.mean_stddev(state_seqs)

        control_seqs = [
            utils.normalize(control, self._control_mean, self._control_std)
            for control in control_seqs
        ]
        state_seqs = [
            utils.normalize(state, self._state_mean, self._state_std)
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
                y, _ = self._initializer.forward(batch['x'].float().to(self.device))
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
                self._predictor.zero_grad()
                # Initialize predictor with state of initializer network
                _, hx = self._initializer.forward(batch['x0'].float().to(self.device))
                # Predict and optimize
                y, _ = self._predictor.forward(
                    batch['x'].float().to(self.device), hx=hx
                )
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
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
    ) -> NDArray[np.float64]:
        if (
            self._state_mean is None
            or self._state_std is None
            or self._control_mean is None
            or self._control_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        self._initializer.eval()
        self._predictor.eval()

        initial_control = utils.normalize(
            initial_control, self._control_mean, self._control_std
        )
        initial_state = utils.normalize(
            initial_state, self._state_mean, self._state_std
        )
        control = utils.normalize(control, self._control_mean, self._control_std)

        with torch.no_grad():
            init_x = (
                torch.from_numpy(np.hstack((initial_control[1:], initial_state[:-1])))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            pred_x = torch.from_numpy(control).unsqueeze(0).float().to(self.device)

            _, hx = self._initializer.forward(init_x)
            y, _ = self._predictor.forward(pred_x, hx=hx)
            y_np: NDArray[np.float64] = (
                y.cpu().detach().squeeze().numpy().astype(np.float64)
            )

        y_np = utils.denormalize(y_np, self._state_mean, self._state_std)
        return y_np

    def save(self, file_path: Tuple[str, ...]) -> None:
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

    @property
    def initializer(self) -> HiddenStateForwardModule:
        return copy.deepcopy(self._initializer)

    @property
    def predictor(self) -> HiddenStateForwardModule:
        return copy.deepcopy(self._predictor)


class LSTMCombinedInitModelConfig(DynamicIdentificationModelConfig):
    recurrent_dim: int
    num_recurrent_layers: int
    dropout: float
    sequence_length: int
    learning_rate: float
    batch_size: int
    epochs: int
    loss: Literal['mse', 'msge']


class LSTMCombinedInitModel(base.DynamicIdentificationModel):
    CONFIG = LSTMCombinedInitModelConfig

    def __init__(self, config: LSTMCombinedInitModelConfig):
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)

        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)

        self.recurrent_dim = config.recurrent_dim
        self.num_recurrent_layers = config.num_recurrent_layers
        self.dropout = config.dropout

        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs = config.epochs

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        elif config.loss == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse" or "msge"')

        self._predictor = rnn.InitLSTM(
            input_dim=self.control_dim,
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=self.state_dim,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = optim.Adam(self._predictor.parameters(), lr=self.learning_rate)

        self._state_mean: Optional[NDArray[np.float64]] = None
        self._state_std: Optional[NDArray[np.float64]] = None
        self._control_mean: Optional[NDArray[np.float64]] = None
        self._control_std: Optional[NDArray[np.float64]] = None

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:
        epoch_losses_initializer = []
        epoch_losses_predictor = []

        self._predictor.train()

        self._control_mean, self._control_std = utils.mean_stddev(control_seqs)
        self._state_mean, self._state_std = utils.mean_stddev(state_seqs)

        control_seqs = [
            utils.normalize(control, self._control_mean, self._control_std)
            for control in control_seqs
        ]
        state_seqs = [
            utils.normalize(state, self._state_mean, self._state_std)
            for state in state_seqs
        ]

        initializer_dataset = RecurrentInitializerDataset(
            control_seqs, state_seqs, self.sequence_length
        )

        predictor_dataset = RecurrentPredictorDataset(
            control_seqs, state_seqs, self.sequence_length
        )

        time_start = time.time()
        for i in range(self.epochs):
            data_loader_init = data.DataLoader(
                initializer_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            data_loader_pred = data.DataLoader(
                predictor_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0.0
            total_init_loss = 0.0
            total_pred_loss = 0.0
            for batch_idx, (batch_init, batch_pred) in enumerate(
                zip(data_loader_init, data_loader_pred)
            ):

                y, h0 = self._predictor.forward(
                    batch_pred['x'].float().to(self.device),
                    x0=batch_init['x'].float().to(self.device),
                )

                batch_loss_init = self.loss.forward(
                    h0, batch_init['y'].float().to(self.device)
                )
                batch_loss_pred = self.loss.forward(
                    y, batch_pred['y'].float().to(self.device)
                )
                batch_loss = batch_loss_init + batch_loss_pred
                total_init_loss += batch_loss_init.item()
                total_pred_loss += batch_loss_pred.item()
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs} \t'
                f'Epoch Loss : {total_loss:.4f}\t'
                f'Init Loss: {total_init_loss:.4f}\t'
                f'Prediction Loss: {total_pred_loss:.4f}'
            )
            epoch_losses_initializer.append([i, total_init_loss])
            epoch_losses_predictor.append([i, total_pred_loss])

        time_end = time.time()

        time_total = time_end - time_start
        logger.info(f'Training time for predictor {time_total}s ')

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
    ) -> NDArray[np.float64]:
        if (
            self._state_mean is None
            or self._state_std is None
            or self._control_mean is None
            or self._control_std is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        self._predictor.eval()

        initial_control = utils.normalize(
            initial_control, self._control_mean, self._control_std
        )
        initial_state = utils.normalize(
            initial_state, self._state_mean, self._state_std
        )
        control = utils.normalize(control, self._control_mean, self._control_std)

        with torch.no_grad():
            init_x = (
                torch.from_numpy(np.hstack((initial_control[1:], initial_state[:-1])))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            pred_x = torch.from_numpy(control).unsqueeze(0).float().to(self.device)

            y, _ = self._predictor.forward(pred_x, x0=init_x)

            y_np: NDArray[np.float64] = (
                y.cpu().detach().squeeze().numpy().astype(np.float64)
            )

        y_np = utils.denormalize(y_np, self._state_mean, self._state_std)
        return y_np

    def save(self, file_path: Tuple[str, ...]) -> None:
        if (
            self._state_mean is None
            or self._state_std is None
            or self._control_mean is None
            or self._control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')

        torch.save(self._predictor.state_dict(), file_path[0])
        with open(file_path[1], mode='w') as f:
            json.dump(
                {
                    'state_mean': self._state_mean.tolist(),
                    'state_stddev': self._state_std.tolist(),
                    'control_mean': self._control_mean.tolist(),
                    'control_stddev': self._control_std.tolist(),
                },
                f,
            )

    def load(self, file_path: Tuple[str, ...]) -> None:
        self._predictor.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        with open(file_path[1], mode='r') as f:
            norm = json.load(f)
        self._state_mean = np.array(norm['state_mean'], dtype=np.float64)
        self._state_std = np.array(norm['state_stddev'], dtype=np.float64)
        self._control_mean = np.array(norm['control_mean'], dtype=np.float64)
        self._control_std = np.array(norm['control_stddev'], dtype=np.float64)

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'predictor.pth', 'json'

    def get_parameter_count(self) -> int:
        predictor_count = sum(
            p.numel() for p in self._predictor.parameters() if p.requires_grad
        )
        return predictor_count
