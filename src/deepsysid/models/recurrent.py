import json
import logging
import time
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from .. import utils
from ..networks import loss, rnn
from . import base
from .base import DynamicIdentificationModelConfig

logger = logging.getLogger()


class ConstrainedRnnConfig(DynamicIdentificationModelConfig):
    nx: int
    nw: int
    gamma: float
    beta: float
    decay_parameter: float
    num_recurrent_layers: int
    dropout: float
    sequence_length: int
    learning_rate: float
    batch_size: int
    epochs_initializer: int
    epochs_predictor: int
    loss: Literal['mse', 'msge']


class ConstrainedRnn(base.DynamicIdentificationModel):
    CONFIG = ConstrainedRnnConfig

    def __init__(self, config: ConstrainedRnnConfig):
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)

        self.nx = config.nx
        self.nu = len(config.control_names)
        self.ny = len(config.state_names)
        self.nw = config.nw

        self.decay_parameter = config.decay_parameter

        self.nw = config.nw
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

        self.predictor = rnn.LTIRnn(
            nx=self.nx,
            nu=self.nu,
            ny=self.ny,
            nw=self.nw,
            gamma=config.gamma,
            beta=config.beta,
        ).to(self.device)

        self.initializer = rnn.BasicLSTM(
            input_dim=self.nu + self.ny,
            recurrent_dim=self.nw,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=[self.ny],
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer_pred = optim.Adam(
            self.predictor.parameters(), lr=self.learning_rate
        )
        self.optimizer_init = optim.Adam(
            self.initializer.parameters(), lr=self.learning_rate
        )

        self.state_mean: Optional[np.ndarray] = None
        self.state_std: Optional[np.ndarray] = None
        self.control_mean: Optional[np.ndarray] = None
        self.control_std: Optional[np.ndarray] = None

    def train(self, control_seqs, state_seqs):
        us = control_seqs
        ys = state_seqs
        self.predictor.initialize_lmi()
        self.predictor.to(self.device)
        self.predictor.train()
        self.initializer.train()

        self.u_mean, self.u_stddev = utils.mean_stddev(us)
        self.y_mean, self.y_stddev = utils.mean_stddev(ys)

        us = [utils.normalize(control, self.u_mean, self.u_stddev) for control in us]
        ys = [utils.normalize(state, self.y_mean, self.y_stddev) for state in ys]

        initializer_dataset = _InitializerDataset(us, ys, self.sequence_length)

        time_start_init = time.time()
        for i in range(self.epochs_initializer):
            data_loader = data.DataLoader(
                initializer_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                self.initializer.zero_grad()
                y = self.initializer.forward(batch['x'].float().to(self.device))
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_init.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_initializer}\t'
                f'Epoch Loss (Initializer): {total_loss}'
            )
        time_end_init = time.time()
        predictor_dataset = _PredictorDataset(us, ys, self.sequence_length)

        time_start_pred = time.time()
        for i in range(self.epochs_predictor):
            data_loader = data.DataLoader(
                predictor_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0
            max_grad = 0
            for batch_idx, batch in enumerate(data_loader):
                self.predictor.zero_grad()
                # Initialize predictor with state of initializer network
                _, hx = self.initializer.forward(
                    batch['x0'].float().to(self.device), return_state=True
                )
                # Predict and optimize
                y = self.predictor.forward(
                    batch['x'].float().to(self.device), hx=hx
                ).to(self.device)
                barrier = self.predictor.get_barrier(self.decay_parameter).to(
                    self.device
                )
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_loss += batch_loss.item()
                (batch_loss + barrier).backward()

                # gradient infos
                grads_norm = [
                    torch.linalg.norm(p.grad) for p in self.predictor.parameters()
                ]
                max_grad += max(grads_norm)

                # save old parameter set
                old_pars = [par.clone().detach() for par in self.predictor.parameters()]

                self.optimizer_pred.step()

                # perform backtracking line search if constraints are not satisfied
                max_iter = 100
                alpha = 0.5
                bls_iter = 0
                while not self.predictor.check_constr():
                    for old_par, new_par in zip(old_pars, self.predictor.parameters()):
                        new_par.data = (
                            alpha * old_par.clone() + (1 - alpha) * new_par.data
                        )

                    if bls_iter > max_iter - 1:
                        M = self.predictor.get_constraints()
                        logger.warning(
                            f'Epoch {i+1}/{self.epochs_predictor}\t'
                            f'max real eigenvalue of M: '
                            f'{(torch.max(torch.real(torch.linalg.eig(M)[0]))):1f}\t'
                            f'Backtracking line search exceeded maximum iteration.'
                        )
                        return
                    bls_iter += 1

            logger.info(
                f'Epoch {i + 1}/{self.epochs_predictor}\t'
                f'Total Loss (Predictor): {total_loss:1f} \t'
                f'Barrier: {barrier:1f}\t'
                f'Backtracking Line Search iteration: {bls_iter}\t'
                f'Max accumulated gradient norm: {max_grad:1f}'
            )

        time_end_pred = time.time()
        time_total_init = time_end_init - time_start_init
        time_total_pred = time_end_pred - time_start_pred
        logger.info(
            f'Training time for initializer {time_total_init}s '
            f'and for predictor {time_total_pred}s'
        )

    def simulate(
        self,
        initial_control: np.ndarray,
        initial_state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        self.initializer.eval()
        self.predictor.eval()

        initial_u = initial_control
        initial_y = initial_state
        u = control

        initial_u = utils.normalize(initial_u, self.u_mean, self.u_stddev)
        initial_y = utils.normalize(initial_y, self.y_mean, self.y_stddev)
        u = utils.normalize(u, self.u_mean, self.u_stddev)

        with torch.no_grad():
            init_x = (
                torch.from_numpy(np.hstack((initial_u[1:], initial_y[:-1])))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            pred_x = torch.from_numpy(u).unsqueeze(0).float().to(self.device)

            _, hx = self.initializer.forward(init_x, return_state=True)
            y = self.predictor.forward(pred_x, hx=hx)
            y_np = y.cpu().detach().squeeze().numpy()  # type: ignore

        y_np = utils.denormalize(y_np, self.y_mean, self.y_stddev)
        return y_np

    def save(self, file_path: Tuple[str, ...]):
        if (
            self.y_mean is None
            or self.y_mean is None
            or self.u_mean is None
            or self.u_mean is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')

        torch.save(self.initializer.state_dict(), file_path[0])
        torch.save(self.predictor.state_dict(), file_path[1])
        with open(file_path[2], mode='w') as f:
            json.dump(
                {
                    'y_mean': self.y_mean.tolist(),
                    'y_stddev': self.y_stddev.tolist(),
                    'u_mean': self.u_mean.tolist(),
                    'u_stddev': self.u_stddev.tolist(),
                },
                f,
            )

    def load(self, file_path: Tuple[str, ...]):
        self.initializer.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        self.predictor.load_state_dict(
            torch.load(file_path[1], map_location=self.device_name)
        )
        with open(file_path[2], mode='r') as f:
            norm = json.load(f)
        self.y_mean = np.array(norm['y_mean'])
        self.y_stddev = np.array(norm['y_stddev'])
        self.u_mean = np.array(norm['u_mean'])
        self.u_stddev = np.array(norm['u_stddev'])

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'initializer.pth', 'predictor.pth', 'json'

    def get_parameter_count(self) -> int:
        # technically parameter counts of both networks are equal
        init_count = sum(
            p.numel() for p in self.initializer.parameters() if p.requires_grad
        )
        predictor_count = sum(
            p.numel() for p in self.predictor.parameters() if p.requires_grad
        )
        return init_count + predictor_count


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


class LSTMInitModel(base.DynamicIdentificationModel):
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

        self.predictor = rnn.BasicLSTM(
            input_dim=self.control_dim,
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=[self.state_dim],
            dropout=self.dropout,
        ).to(self.device)

        self.initializer = rnn.BasicLSTM(
            input_dim=self.control_dim + self.state_dim,
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=[self.state_dim],
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer_pred = optim.Adam(
            self.predictor.parameters(), lr=self.learning_rate
        )
        self.optimizer_init = optim.Adam(
            self.initializer.parameters(), lr=self.learning_rate
        )

        self.state_mean: Optional[np.ndarray] = None
        self.state_std: Optional[np.ndarray] = None
        self.control_mean: Optional[np.ndarray] = None
        self.control_std: Optional[np.ndarray] = None

    def train(self, control_seqs, state_seqs) -> Dict[str, np.ndarray]:
        epoch_losses_initializer = []
        epoch_losses_predictor = []

        self.predictor.train()
        self.initializer.train()

        self.control_mean, self.control_std = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_std = utils.mean_stddev(state_seqs)

        control_seqs = [
            utils.normalize(control, self.control_mean, self.control_std)
            for control in control_seqs
        ]
        state_seqs = [
            utils.normalize(state, self.state_mean, self.state_std)
            for state in state_seqs
        ]

        initializer_dataset = _InitializerDataset(
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
                y = self.initializer.forward(batch['x'].float().to(self.device))
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
        predictor_dataset = _PredictorDataset(
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
                _, hx = self.initializer.forward(
                    batch['x0'].float().to(self.device), return_state=True
                )
                # Predict and optimize
                y = self.predictor.forward(batch['x'].float().to(self.device), hx=hx)
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
            epoch_loss_initializer=np.array(epoch_losses_initializer),
            epoch_loss_predictor=np.array(epoch_losses_predictor),
            training_time_initializer=np.array([time_total_init]),
            training_time_predictor=np.array([time_total_pred]),
        )

    def simulate(
        self,
        initial_control: np.ndarray,
        initial_state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        self.initializer.eval()
        self.predictor.eval()

        initial_control = utils.normalize(
            initial_control, self.control_mean, self.control_std
        )
        initial_state = utils.normalize(initial_state, self.state_mean, self.state_std)
        control = utils.normalize(control, self.control_mean, self.control_std)

        with torch.no_grad():
            init_x = (
                torch.from_numpy(np.hstack((initial_control[1:], initial_state[:-1])))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            pred_x = torch.from_numpy(control).unsqueeze(0).float().to(self.device)

            _, hx = self.initializer.forward(init_x, return_state=True)
            y = self.predictor.forward(pred_x, hx=hx, return_state=False)
            y_np = y.cpu().detach().squeeze().numpy()  # type: ignore

        y_np = utils.denormalize(y_np, self.state_mean, self.state_std)
        return y_np

    def save(self, file_path: Tuple[str, ...]):
        if (
            self.state_mean is None
            or self.state_std is None
            or self.control_mean is None
            or self.control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')

        torch.save(self.initializer.state_dict(), file_path[0])
        torch.save(self.predictor.state_dict(), file_path[1])
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

    def load(self, file_path: Tuple[str, ...]):
        self.initializer.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        self.predictor.load_state_dict(
            torch.load(file_path[1], map_location=self.device_name)
        )
        with open(file_path[2], mode='r') as f:
            norm = json.load(f)
        self.state_mean = np.array(norm['state_mean'])
        self.state_std = np.array(norm['state_std'])
        self.control_mean = np.array(norm['control_mean'])
        self.control_std = np.array(norm['control_std'])

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'initializer.pth', 'predictor.pth', 'json'

    def get_parameter_count(self) -> int:
        # technically parameter counts of both networks are equal
        init_count = sum(
            p.numel() for p in self.initializer.parameters() if p.requires_grad
        )
        predictor_count = sum(
            p.numel() for p in self.predictor.parameters() if p.requires_grad
        )
        return init_count + predictor_count

class LSTMInitModelCombinedConfig(DynamicIdentificationModelConfig):
    recurrent_dim: int
    num_recurrent_layers: int
    dropout: float
    sequence_length: int
    learning_rate: float
    batch_size: int
    epochs: int
    loss: Literal['mse', 'msge']
class LSTMInitModelCombined(base.DynamicIdentificationModel):
    CONFIG = LSTMInitModelCombinedConfig

    def __init__(self, config: LSTMInitModelCombinedConfig):
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

        self.predictor = rnn.InitLSTM(
            input_dim=self.control_dim,
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=self.state_dim,
            dropout=self.dropout,
        ).to(self.device)
        self.optimizer = optim.Adam(
            self.predictor.parameters(), lr=self.learning_rate
        )

        self.state_mean: Optional[np.ndarray] = None
        self.state_std: Optional[np.ndarray] = None
        self.control_mean: Optional[np.ndarray] = None
        self.control_std: Optional[np.ndarray] = None

    def train(self, control_seqs, state_seqs):
        self.predictor.train()

        self.control_mean, self.control_stddev = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_stddev = utils.mean_stddev(state_seqs)

        control_seqs = [
            utils.normalize(control, self.control_mean, self.control_stddev)
            for control in control_seqs
        ]
        state_seqs = [
            utils.normalize(state, self.state_mean, self.state_stddev)
            for state in state_seqs
        ]

        initializer_dataset = _InitializerDataset(
            control_seqs, state_seqs, self.sequence_length
        )

        predictor_dataset = _PredictorDataset(
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
            for batch_idx, (batch_init, batch_pred) in enumerate(zip(data_loader_init, data_loader_pred)):

                y, h0 = self.predictor.forward(batch_pred['x'].float().to(self.device), x0=batch_init['x'].float().to(self.device), return_init=True)

                batch_loss_init = self.loss.forward(h0, batch_init['y'].float().to(self.device))
                batch_loss_pred = self.loss.forward(y, batch_pred['y'].float().to(self.device))
                batch_loss = batch_loss_init + batch_loss_pred
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs} \t'
                f'Epoch Loss : {total_loss:1f}\t'
                f'Init Loss: {batch_loss_init:1f}\t'
                f'Prediction Loss: {batch_loss_pred:1f}'
            )
        time_end = time.time()
       
        time_total = time_end - time_start
        logger.info(
            f'Training time for predictor {time_total}s '
        )

    def simulate(
        self,
        initial_control: np.ndarray,
        initial_state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
        self.initializer.eval()
        self.predictor.eval()

        initial_control = utils.normalize(
            initial_control, self.control_mean, self.control_stddev
        )
        initial_state = utils.normalize(
            initial_state, self.state_mean, self.state_stddev
        )
        control = utils.normalize(control, self.control_mean, self.control_stddev)

        with torch.no_grad():
            init_x = (
                torch.from_numpy(np.hstack((initial_control[1:], initial_state[:-1])))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            pred_x = torch.from_numpy(control).unsqueeze(0).float().to(self.device)

            y = self.predictor.forward(pred_x, h0=init_x, return_state=False)
            y_np = y.cpu().detach().squeeze().numpy()  # type: ignore

        y_np = utils.denormalize(y_np, self.state_mean, self.state_stddev)
        return y_np

    def save(self, file_path: Tuple[str, ...]):
        if (
            self.state_mean is None
            or self.state_stddev is None
            or self.control_mean is None
            or self.control_stddev is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')

        torch.save(self.predictor.state_dict(), file_path[0])
        with open(file_path[1], mode='w') as f:
            json.dump(
                {
                    'state_mean': self.state_mean.tolist(),
                    'state_stddev': self.state_stddev.tolist(),
                    'control_mean': self.control_mean.tolist(),
                    'control_stddev': self.control_stddev.tolist(),
                },
                f,
            )

    def load(self, file_path: Tuple[str, ...]):
        self.predictor.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        with open(file_path[1], mode='r') as f:
            norm = json.load(f)
        self.state_mean = np.array(norm['state_mean'])
        self.state_stddev = np.array(norm['state_stddev'])
        self.control_mean = np.array(norm['control_mean'])
        self.control_stddev = np.array(norm['control_stddev'])

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'predictor.pth', 'json'

    def get_parameter_count(self) -> int:
        # technically parameter counts of both networks are equal
        init_count = sum(
            p.numel() for p in self.initializer.parameters() if p.requires_grad
        )
        predictor_count = sum(
            p.numel() for p in self.predictor.parameters() if p.requires_grad
        )
        return init_count + predictor_count


class _InitializerDataset(data.Dataset):
    # x=[control state], y=[state]
    def __init__(self, control_seqs, state_seqs, sequence_length):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.x = None
        self.y = None
        self.__load_data(control_seqs, state_seqs)

    def __load_data(self, control_seqs, state_seqs):
        x_seq = list()
        y_seq = list()
        for control, state in zip(control_seqs, state_seqs):
            n_samples = int(
                (control.shape[0] - self.sequence_length - 1) / self.sequence_length
            )

            x = np.zeros(
                (n_samples, self.sequence_length, self.control_dim + self.state_dim)
            )
            y = np.zeros((n_samples, self.sequence_length, self.state_dim))

            for idx in range(n_samples):
                time = idx * self.sequence_length

                x[idx, :, :] = np.hstack(
                    (
                        control[time + 1 : time + 1 + self.sequence_length, :],
                        state[time : time + self.sequence_length, :],
                    )
                )
                y[idx, :, :] = state[time + 1 : time + 1 + self.sequence_length, :]

            x_seq.append(x)
            y_seq.append(y)

        self.x = np.vstack(x_seq)
        self.y = np.vstack(y_seq)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx]}


class _PredictorDataset(data.Dataset):
    def __init__(self, control_seqs, state_seqs, sequence_length):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.x0 = None
        self.x = None
        self.y = None
        self.__load_data(control_seqs, state_seqs)

    def __load_data(self, control_seqs, state_seqs):
        x0_seq = list()
        x_seq = list()
        y_seq = list()
        for control, state in zip(control_seqs, state_seqs):
            n_samples = int(
                (control.shape[0] - 2 * self.sequence_length) / self.sequence_length
            )

            x0 = np.zeros(
                (n_samples, self.sequence_length, self.control_dim + self.state_dim)
            )
            x = np.zeros((n_samples, self.sequence_length, self.control_dim))
            y = np.zeros((n_samples, self.sequence_length, self.state_dim))

            for idx in range(n_samples):
                time = idx * self.sequence_length

                x0[idx, :, :] = np.hstack(
                    (
                        control[time : time + self.sequence_length],
                        state[time : time + self.sequence_length, :],
                    )
                )
                x[idx, :, :] = control[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                y[idx, :, :] = state[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]

            x0_seq.append(x0)
            x_seq.append(x)
            y_seq.append(y)

        self.x0 = np.vstack(x0_seq)
        self.x = np.vstack(x_seq)
        self.y = np.vstack(y_seq)

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, idx):
        return {'x0': self.x0[idx], 'x': self.x[idx], 'y': self.y[idx]}
