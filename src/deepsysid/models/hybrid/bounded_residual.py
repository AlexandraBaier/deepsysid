import abc
import json
import logging
from typing import Dict, List, Literal, Optional, Tuple

import numpy as np
import torch
from torch import nn, optim
from torch.nn.functional import mse_loss
from torch.utils import data

from ...networks import loss, rnn
from .. import base, utils
from ..base import DynamicIdentificationModelConfig
from .physical import (
    MinimalManeuveringComponent,
    MinimalManeuveringConfig,
    NoOpPhysicalComponent,
    PhysicalComponent,
    PropulsionManeuveringComponent,
    PropulsionManeuveringConfig,
)
from .semiphysical import (
    BlankeComponent,
    LinearComponent,
    NoOpSemiphysicalComponent,
    SemiphysicalComponent,
)

logger = logging.getLogger(__name__)


class HybridResidualLSTMModelConfig(DynamicIdentificationModelConfig):
    recurrent_dim: int
    num_recurrent_layers: int
    dropout: float
    sequence_length: int
    learning_rate: float
    batch_size: int
    epochs_initializer: int
    epochs_parallel: int
    epochs_feedback: int
    loss: Literal['mse', 'msge']


class HybridMinimalManeuveringModelConfig(
    HybridResidualLSTMModelConfig, MinimalManeuveringConfig
):
    pass


class HybridPropulsionManeuveringModelConfig(
    HybridResidualLSTMModelConfig, PropulsionManeuveringConfig
):
    pass


class HybridResidualLSTMModel(base.DynamicIdentificationModel, abc.ABC):
    def __init__(
        self,
        physical: PhysicalComponent,
        semiphysical: SemiphysicalComponent,
        config: HybridResidualLSTMModelConfig,
        device_name: str,
    ):
        super().__init__(config)

        if physical.STATES is not None and any(
            sn not in config.state_names for sn in physical.STATES
        ):
            raise ValueError(f'physical expects states: {str(physical.STATES)}.')

        if physical.CONTROLS is not None and any(
            cn not in config.control_names for cn in physical.CONTROLS
        ):
            raise ValueError(f'physical expects controls: {str(physical.CONTROLS)}.')

        if semiphysical.STATES is not None and any(
            sn not in config.state_names for sn in semiphysical.STATES
        ):
            raise ValueError(
                f'semiphysical expects states: {str(semiphysical.STATES)}.'
            )

        if semiphysical.CONTROLS is not None and any(
            cn not in config.control_names for cn in semiphysical.CONTROLS
        ):
            raise ValueError(
                f'semiphysical expects controls: {str(semiphysical.CONTROLS)}.'
            )

        self.device_name = device_name
        self.device = torch.device(device_name)

        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)
        self.time_delta = float(config.time_delta)

        self.recurrent_dim = config.recurrent_dim
        self.num_recurrent_layers = config.num_recurrent_layers
        self.dropout = config.dropout

        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs_initializer = config.epochs_initializer
        self.epochs_parallel = config.epochs_parallel
        self.epochs_feedback = config.epochs_feedback

        if config.loss == 'mse':
            self.loss: nn.Module = nn.MSELoss().to(self.device)
        elif config.loss == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError('loss can only be "mse" or "msge"')

        self.physical = physical
        self.semiphysical = semiphysical

        if physical.STATES is None:
            self.physical_state_mask: np.ndarray = np.array(
                list(range(len(config.state_names))), dtype=np.int32
            )
        else:
            self.physical_state_mask = np.array(
                list(config.state_names.index(sn) for sn in physical.STATES),
                dtype=np.int32,
            )
        if physical.CONTROLS is None:
            self.physical_control_mask: np.ndarray = np.array(
                list(range(len(config.control_names))), dtype=np.int32
            )
        else:
            self.physical_control_mask = np.array(
                list(config.control_names.index(cn) for cn in physical.CONTROLS),
                dtype=np.int32,
            )

        if semiphysical.STATES is None:
            self.semiphysical_state_mask: np.ndarray = np.array(
                list(range(len(config.state_names))), dtype=np.int32
            )
        else:
            self.semiphysical_state_mask = np.array(
                list(config.state_names.index(sn) for sn in semiphysical.STATES),
                dtype=np.int32,
            )
        if semiphysical.CONTROLS is None:
            self.semiphysical_control_mask: np.ndarray = np.array(
                list(range(len(config.control_names))), dtype=np.int32
            )
        else:
            self.semiphysical_control_mask = np.array(
                list(config.control_names.index(cn) for cn in semiphysical.CONTROLS),
                dtype=np.int32,
            )

        self.blackbox = rnn.LinearOutputLSTM(
            input_dim=self.control_dim
            + self.state_dim,  # control input and whitebox estimate
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=self.state_dim,
            dropout=self.dropout,
        ).to(self.device)

        self.initializer = rnn.BasicLSTM(
            input_dim=self.control_dim + self.state_dim,
            recurrent_dim=self.recurrent_dim,
            output_dim=[self.state_dim],
            num_recurrent_layers=self.num_recurrent_layers,
            dropout=self.dropout,
        ).to(self.device)

        self.optimizer_initializer = optim.Adam(
            params=self.initializer.parameters(), lr=self.learning_rate
        )
        self.optimizer_end2end = optim.Adam(
            params=self.blackbox.parameters(), lr=self.learning_rate
        )

        self.state_mean: Optional[np.ndarray] = None
        self.state_std: Optional[np.ndarray] = None
        self.control_mean: Optional[np.ndarray] = None
        self.control_std: Optional[np.ndarray] = None

    def train(
        self, control_seqs: List[np.ndarray], state_seqs: List[np.ndarray]
    ) -> Dict[str, np.ndarray]:
        epoch_losses_initializer = []
        epoch_losses_teacher = []
        epoch_losses_multistep = []

        self.blackbox.train()
        self.initializer.train()
        self.physical.train()
        self.semiphysical.train()

        self.control_mean, self.control_std = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_std = utils.mean_stddev(state_seqs)

        un_control_seqs = control_seqs
        un_state_seqs = state_seqs
        control_seqs = [
            utils.normalize(control, self.control_mean, self.control_std)
            for control in control_seqs
        ]
        state_seqs = [
            utils.normalize(state, self.state_mean, self.state_std)
            for state in state_seqs
        ]

        state_mean = torch.from_numpy(self.state_mean).float().to(self.device)
        state_std = torch.from_numpy(self.state_std).float().to(self.device)

        def denormalize_state(x: torch.Tensor) -> torch.Tensor:
            return (x * state_std) + state_mean

        def scale_acc_physical(x: torch.Tensor) -> torch.Tensor:
            return x / state_std[self.physical_state_mask]  # type: ignore

        def scale_acc_physical_np(x: np.ndarray) -> np.ndarray:
            return x / self.state_std[self.physical_state_mask]  # type: ignore

        # Train linear model
        targets_seqs = []
        # Prepare target values for training of semiphysical model.
        for un_control, un_state in zip(un_control_seqs, un_state_seqs):
            target = utils.normalize(un_state[1:], self.state_mean, self.state_std)

            target[:, self.physical_state_mask] = target[
                :, self.physical_state_mask
            ] - scale_acc_physical_np(
                self.physical.time_delta
                * self.physical.forward(
                    torch.from_numpy(un_control[1:, self.physical_control_mask])
                    .float()
                    .to(self.device),
                    torch.from_numpy(un_state[:-1]).float().to(self.device),
                )
                .cpu()
                .detach()
                .numpy()
            )
            targets_seqs.append(target[:, self.semiphysical_state_mask])

        self.semiphysical.train_semiphysical(
            control_seqs=[
                cseq[:, self.semiphysical_control_mask] for cseq in un_control_seqs
            ],
            state_seqs=[
                sseq[:, self.semiphysical_state_mask] for sseq in un_state_seqs
            ],
            target_seqs=targets_seqs,
        )
        # Do not let the semiphysical model train together with the LSTM.
        self.semiphysical.eval()

        initializer_dataset = InitializerDataset(
            control_seqs, state_seqs, self.sequence_length
        )
        for i in range(self.epochs_initializer):
            data_loader = data.DataLoader(
                initializer_dataset, self.batch_size, shuffle=True, drop_last=True
            )
            total_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                self.initializer.zero_grad()
                y = self.initializer.forward(batch['x'].float().to(self.device))
                # This type error is ignored, since we know that y will not be a tuple.
                batch_loss = mse_loss(
                    y, batch['y'].float().to(self.device)  # type: ignore
                )
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_initializer.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_initializer} '
                f'- Epoch Loss (Initializer): {total_loss}'
            )
            epoch_losses_initializer.append([i, total_loss])

        dataset = End2EndDataset(
            control_seqs=control_seqs,
            state_seqs=state_seqs,
            un_control_seqs=un_control_seqs,
            un_state_seqs=un_state_seqs,
            sequence_length=self.sequence_length,
        )
        for i in range(self.epochs_parallel):
            data_loader = data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )
            total_epoch_loss = 0.0
            for batch in data_loader:
                self.blackbox.zero_grad()

                x_control_unnormed = batch['x_control_unnormed'].float().to(self.device)
                x_state_unnormed = batch['x_state_unnormed'].float().to(self.device)
                y_whitebox = (
                    torch.zeros((self.batch_size, self.sequence_length, self.state_dim))
                    .float()
                    .to(self.device)
                )

                for time in range(self.sequence_length):
                    y_semiphysical = self.semiphysical.forward(
                        control=x_control_unnormed[
                            :, time, self.semiphysical_control_mask
                        ],
                        state=x_state_unnormed[:, time, self.semiphysical_state_mask],
                    )
                    ydot_physical = scale_acc_physical(
                        self.physical.forward(
                            control=x_control_unnormed[
                                :, time, self.physical_control_mask
                            ],
                            state=x_state_unnormed[:, time, self.physical_state_mask],
                        )
                    )

                    y_whitebox[:, time, self.physical_state_mask] = (
                        y_whitebox[:, time, self.physical_state_mask]
                        + self.physical.time_delta * ydot_physical
                    )
                    y_whitebox[:, time, self.semiphysical_state_mask] = (
                        y_whitebox[:, time, self.semiphysical_state_mask]
                        + y_semiphysical
                    )

                x_init = batch['x_init'].float().to(self.device)
                x_pred = torch.cat(
                    (batch['x_pred'].float().to(self.device), y_whitebox), dim=2
                )  # serial connection

                _, hx_init = self.initializer.forward(x_init, return_state=True)

                y_blackbox = self.blackbox_forward(x_pred, y_whitebox, hx=hx_init)
                y = y_blackbox + y_whitebox

                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_end2end.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_parallel} '
                f'- Epoch Loss (Parallel): {total_epoch_loss}'
            )
            epoch_losses_teacher.append([i, total_epoch_loss])

        self.optimizer_end2end = optim.Adam(
            params=self.blackbox.parameters(), lr=self.learning_rate
        )
        for i in range(self.epochs_feedback):
            data_loader = data.DataLoader(
                dataset=dataset,
                batch_size=self.batch_size,
                shuffle=True,
                drop_last=True,
            )
            total_epoch_loss = 0.0
            for batch in data_loader:
                self.blackbox.zero_grad()

                current_state = batch['initial_state'].float().to(self.device)
                x_control_unnormed = batch['x_control_unnormed'].float().to(self.device)
                x_pred = batch['x_pred'].float().to(self.device)
                y_est = (
                    torch.zeros((self.batch_size, self.sequence_length, self.state_dim))
                    .float()
                    .to(self.device)
                )

                x_init = batch['x_init'].float().to(self.device)
                _, hx_init = self.initializer.forward(x_init, return_state=True)

                for time in range(self.sequence_length):
                    y_whitebox = (
                        torch.zeros((self.batch_size, self.state_dim))
                        .float()
                        .to(self.device)
                    )
                    y_semiphysical = self.semiphysical.forward(
                        x_control_unnormed[:, time, self.semiphysical_control_mask],
                        current_state[:, self.semiphysical_state_mask],
                    )

                    ydot_physical = scale_acc_physical(
                        self.physical.forward(
                            x_control_unnormed[:, time, self.physical_control_mask],
                            current_state[:, self.physical_state_mask],
                        )
                    )
                    y_whitebox[:, self.physical_state_mask] = (
                        y_whitebox[:, self.physical_state_mask]
                        + self.time_delta * ydot_physical
                    )
                    y_whitebox[:, self.semiphysical_state_mask] = (
                        y_whitebox[:, self.semiphysical_state_mask] + y_semiphysical
                    )

                    y_blackbox, hx_init = self.blackbox_forward(
                        torch.cat(
                            (x_pred[:, time, :].unsqueeze(1), y_whitebox.unsqueeze(1)),
                            dim=2,
                        ),
                        y_whitebox.unsqueeze(1),
                        hx=hx_init,
                        return_state=True,
                    )
                    current_state = y_blackbox.squeeze(1) + y_whitebox
                    y_est[:, time, :] = current_state
                    current_state = denormalize_state(current_state)
                batch_loss = self.loss.forward(
                    y_est, batch['y'].float().to(self.device)
                )
                total_epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_end2end.step()

            logger.info(
                f'Epoch {i + 1}/{self.epochs_feedback} '
                f'- Epoch Loss (Feedback): {total_epoch_loss}'
            )
            epoch_losses_multistep.append([i, total_epoch_loss])

        return dict(
            epoch_loss_initializer=np.array(epoch_losses_initializer),
            epoch_loss_teacher=np.array(epoch_losses_teacher),
            epoch_loss_multistep=np.array(epoch_losses_multistep),
        )

    def simulate(
        self,
        initial_control: np.ndarray,
        initial_state: np.ndarray,
        control: np.ndarray,
        threshold: float = np.infty,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        y, whitebox, blackbox = self.simulate_hybrid(
            initial_control=initial_control,
            initial_state=initial_state,
            control=control,
        )
        return y, dict(whitebox=whitebox, blackbox=blackbox)

    def simulate_hybrid(
        self,
        initial_control: np.ndarray,
        initial_state: np.ndarray,
        control: np.ndarray,
        threshold: float = np.infty,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.blackbox.eval()
        self.initializer.eval()
        self.semiphysical.eval()
        self.physical.eval()

        state_mean = torch.from_numpy(self.state_mean).float().to(self.device)
        state_std = torch.from_numpy(self.state_std).float().to(self.device)

        def denormalize_state(x: torch.Tensor) -> torch.Tensor:
            return (x * state_std) + state_mean

        def scale_acc_physical(x: torch.Tensor) -> torch.Tensor:
            return x / state_std[self.physical_state_mask]  # type: ignore

        un_control = control
        current_state_np = initial_state[-1, :]
        initial_control = utils.normalize(
            initial_control, self.control_mean, self.control_std
        )
        initial_state = utils.normalize(initial_state, self.state_mean, self.state_std)
        control = utils.normalize(control, self.control_mean, self.control_std)

        y = np.zeros((control.shape[0], self.state_dim))
        whitebox = np.zeros((control.shape[0], self.state_dim))
        blackbox = np.zeros((control.shape[0], self.state_dim))

        with torch.no_grad():
            x_init = (
                torch.from_numpy(np.hstack((initial_control, initial_state)))
                .unsqueeze(0)
                .float()
                .to(self.device)
            )
            _, hx = self.initializer.forward(
                x_init, return_state=True
            )  # hx is hidden state of predictor LSTM

            x_control_un = (
                torch.from_numpy(un_control).unsqueeze(0).float().to(self.device)
            )
            current_state = (
                torch.from_numpy(current_state_np).unsqueeze(0).float().to(self.device)
            )
            x_pred = torch.from_numpy(control).unsqueeze(0).float().to(self.device)
            for time in range(control.shape[0]):
                y_whitebox = (
                    torch.zeros((current_state.shape[0], self.state_dim))
                    .float()
                    .to(self.device)
                )
                y_semiphysical = self.semiphysical.forward(
                    control=x_control_un[:, time, self.semiphysical_control_mask],
                    state=current_state[:, self.semiphysical_state_mask],
                )
                ydot_physical = scale_acc_physical(
                    self.physical.forward(
                        x_control_un[:, time, self.physical_control_mask],
                        current_state[:, self.physical_state_mask],
                    )
                )
                y_whitebox[:, self.physical_state_mask] = (
                    y_whitebox[:, self.physical_state_mask]
                    + self.time_delta * ydot_physical
                )
                y_whitebox[:, self.semiphysical_state_mask] = (
                    y_whitebox[:, self.semiphysical_state_mask] + y_semiphysical
                )

                x_blackbox = torch.cat(
                    (x_pred[:, time, :], y_whitebox), dim=1
                ).unsqueeze(1)
                y_blackbox, hx = self.blackbox_forward(
                    x_blackbox, None, hx=hx, return_state=True
                )
                y_blackbox = torch.clamp(y_blackbox, -threshold, threshold)
                y_est = y_blackbox.squeeze(1) + y_whitebox
                current_state = denormalize_state(y_est)
                y[time, :] = current_state.cpu().detach().numpy()
                whitebox[time, :] = y_whitebox.cpu().detach().numpy()
                blackbox[time, :] = y_blackbox.squeeze(1).cpu().detach().numpy()

        return y, whitebox, blackbox

    def save(self, file_path: Tuple[str, ...]):
        if (
            self.state_mean is None
            or self.state_std is None
            or self.control_mean is None
            or self.control_std is None
        ):
            raise ValueError('Model has not been trained and cannot be saved.')

        torch.save(self.semiphysical.state_dict(), file_path[0])
        torch.save(self.blackbox.state_dict(), file_path[1])
        torch.save(self.initializer.state_dict(), file_path[2])

        semiphysical_params = [
            param.tolist() for param in self.semiphysical.get_parameters_to_save()
        ]

        with open(file_path[3], mode='w') as f:
            json.dump(
                {
                    'state_mean': self.state_mean.tolist(),
                    'state_std': self.state_std.tolist(),
                    'control_mean': self.control_mean.tolist(),
                    'control_std': self.control_std.tolist(),
                    'semiphysical': semiphysical_params,
                },
                f,
            )

    def load(self, file_path: Tuple[str, ...]):
        self.semiphysical.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )
        self.blackbox.load_state_dict(
            torch.load(file_path[1], map_location=self.device_name)
        )
        self.initializer.load_state_dict(
            torch.load(file_path[2], map_location=self.device_name)
        )
        with open(file_path[3], mode='r') as f:
            norm = json.load(f)
        self.state_mean = np.array(norm['state_mean'])
        self.state_std = np.array(norm['state_std'])
        self.control_mean = np.array(norm['control_mean'])
        self.control_std = np.array(norm['control_std'])
        self.semiphysical.load_parameters(
            [np.array(param) for param in norm['semiphysical']]
        )

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'semi-physical.pth', 'blackbox.pth', 'initializer.pth', 'json'

    def get_parameter_count(self) -> int:
        semiphysical_count = sum(p.numel() for p in self.semiphysical.parameters())
        blackbox_count = sum(
            p.numel() for p in self.blackbox.parameters() if p.requires_grad
        )
        initializer_count = sum(
            p.numel() for p in self.initializer.parameters() if p.requires_grad
        )
        return semiphysical_count + blackbox_count + initializer_count

    def blackbox_forward(self, x_pred, y_wb, hx=None, return_state=False):
        # TODO: x_pred should instead be x_control.
        # TODO: I don't remember the purpose of this function.
        #  Probably to generalize the code in some way?
        return self.blackbox.forward(x_pred, hx=hx, return_state=return_state)


class HybridMinimalManeuveringModel(HybridResidualLSTMModel):
    CONFIG = HybridMinimalManeuveringModelConfig

    def __init__(self, config: HybridMinimalManeuveringModelConfig):
        device = torch.device(config.device_name)
        physical = MinimalManeuveringComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = NoOpSemiphysicalComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )

        super().__init__(
            physical=physical,
            semiphysical=semiphysical,
            config=config,
            device_name=config.device_name,
        )


class HybridPropulsionManeuveringModel(HybridResidualLSTMModel):
    CONFIG = HybridPropulsionManeuveringModelConfig

    def __init__(self, config: HybridPropulsionManeuveringModelConfig):
        device = torch.device(config.device_name)
        physical = PropulsionManeuveringComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = NoOpSemiphysicalComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )

        super().__init__(
            physical=physical,
            semiphysical=semiphysical,
            config=config,
            device_name=config.device_name,
        )


class HybridLinearModel(HybridResidualLSTMModel):
    CONFIG = HybridResidualLSTMModelConfig

    def __init__(self, config: HybridResidualLSTMModelConfig):
        device = torch.device(config.device_name)
        physical = NoOpPhysicalComponent(time_delta=config.time_delta, device=device)
        semiphysical = LinearComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )

        super().__init__(
            physical=physical,
            semiphysical=semiphysical,
            config=config,
            device_name=config.device_name,
        )


class HybridBlankeModel(HybridResidualLSTMModel):
    CONFIG = HybridResidualLSTMModelConfig

    def __init__(self, config: HybridResidualLSTMModelConfig):
        device = torch.device(config.device_name)
        physical = NoOpPhysicalComponent(time_delta=config.time_delta, device=device)
        semiphysical = BlankeComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )

        super().__init__(
            physical=physical,
            semiphysical=semiphysical,
            config=config,
            device_name=config.device_name,
        )


class HybridLinearMinimalManeuveringModel(HybridResidualLSTMModel):
    CONFIG = HybridMinimalManeuveringModelConfig

    def __init__(self, config: HybridMinimalManeuveringModelConfig):
        device = torch.device(config.device_name)
        physical = MinimalManeuveringComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = LinearComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )

        super().__init__(
            physical=physical,
            semiphysical=semiphysical,
            config=config,
            device_name=config.device_name,
        )


class HybridLinearPropulsionManeuveringModel(HybridResidualLSTMModel):
    CONFIG = HybridPropulsionManeuveringModelConfig

    def __init__(self, config: HybridPropulsionManeuveringModelConfig):
        device = torch.device(config.device_name)
        physical = PropulsionManeuveringComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = LinearComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )

        super().__init__(
            physical=physical,
            semiphysical=semiphysical,
            config=config,
            device_name=config.device_name,
        )


class HybridBlankeMinimalManeuveringModel(HybridResidualLSTMModel):
    CONFIG = HybridMinimalManeuveringModelConfig

    def __init__(self, config: HybridMinimalManeuveringModelConfig):
        device = torch.device(config.device_name)
        physical = MinimalManeuveringComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = BlankeComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )

        super().__init__(
            physical=physical,
            semiphysical=semiphysical,
            config=config,
            device_name=config.device_name,
        )


class HybridBlankePropulsionModel(HybridResidualLSTMModel):
    CONFIG = HybridPropulsionManeuveringModelConfig

    def __init__(self, config: HybridPropulsionManeuveringModelConfig):
        device = torch.device(config.device_name)
        physical = PropulsionManeuveringComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = BlankeComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )

        super().__init__(
            physical=physical,
            semiphysical=semiphysical,
            config=config,
            device_name=config.device_name,
        )


class InitializerDataset(data.Dataset):
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


class End2EndDataset(data.Dataset):
    def __init__(
        self, control_seqs, state_seqs, sequence_length, un_control_seqs, un_state_seqs
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]

        self.x_init = None
        self.x_pred = None
        self.ydot = None
        self.y = None
        self.initial_state = None
        self.x_control_unnormed = None
        self.x_state_unnormed = None

        self.__load_data(control_seqs, state_seqs, un_control_seqs, un_state_seqs)

    def __load_data(self, control_seqs, state_seqs, un_control_seqs, un_state_seqs):
        x_init_seq = list()
        x_pred_seq = list()
        ydot_seq = list()
        y_seq = list()
        initial_state_seq = list()
        x_control_unnormed_seq = list()
        x_state_unnormed_seq = list()

        for control, state, un_control, un_state in zip(
            control_seqs, state_seqs, un_control_seqs, un_state_seqs
        ):
            n_samples = int(
                (control.shape[0] - self.sequence_length - 1)
                / (2 * self.sequence_length)
            )

            x_init = np.zeros(
                (n_samples, self.sequence_length, self.control_dim + self.state_dim)
            )
            x_pred = np.zeros((n_samples, self.sequence_length, self.control_dim))
            ydot = np.zeros((n_samples, self.sequence_length, self.state_dim))
            y = np.zeros((n_samples, self.sequence_length, self.state_dim))
            initial_state = np.zeros((n_samples, self.state_dim))
            x_control_unnormed = np.zeros(
                (n_samples, self.sequence_length, self.control_dim)
            )
            x_state_unnormed = np.zeros(
                (n_samples, self.sequence_length, self.state_dim)
            )

            for idx in range(n_samples):
                time = idx * self.sequence_length

                x_init[idx, :, :] = np.hstack(
                    (
                        control[time : time + self.sequence_length, :],
                        state[time : time + self.sequence_length, :],
                    )
                )
                x_pred[idx, :, :] = control[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                ydot[idx, :, :] = (
                    state[
                        time + self.sequence_length : time + 2 * self.sequence_length, :
                    ]
                    - state[
                        time
                        + self.sequence_length
                        - 1 : time
                        + 2 * self.sequence_length
                        - 1,
                        :,
                    ]
                )
                y[idx, :, :] = state[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                initial_state[idx, :] = un_state[time + self.sequence_length - 1, :]
                x_control_unnormed[idx, :, :] = un_control[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                x_state_unnormed[idx, :, :] = un_state[
                    time
                    + self.sequence_length
                    - 1 : time
                    + 2 * self.sequence_length
                    - 1,
                    :,
                ]

            x_init_seq.append(x_init)
            x_pred_seq.append(x_pred)
            ydot_seq.append(ydot)
            y_seq.append(y)
            initial_state_seq.append(initial_state)
            x_control_unnormed_seq.append(x_control_unnormed)
            x_state_unnormed_seq.append(x_state_unnormed)

        self.x_init = np.vstack(x_init_seq)
        self.x_pred = np.vstack(x_pred_seq)
        self.ydot = np.vstack(ydot_seq)
        self.y = np.vstack(y_seq)
        self.initial_state = np.vstack(initial_state_seq)
        self.x_control_unnormed = np.vstack(x_control_unnormed_seq)
        self.x_state_unnormed = np.vstack(x_state_unnormed_seq)

    def __len__(self):
        return self.x_init.shape[0]

    def __getitem__(self, idx):
        return {
            'x_init': self.x_init[idx],
            'x_pred': self.x_pred[idx],
            'ydot': self.ydot[idx],
            'y': self.y[idx],
            'initial_state': self.initial_state[idx],
            'x_control_unnormed': self.x_control_unnormed[idx],
            'x_state_unnormed': self.x_state_unnormed[idx],
        }
