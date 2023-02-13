"""
Serial hybrid architecture proposed by
N. Mohajerin, M. Mozifian and S. Waslander,
"Deep Learning a Quadrotor Dynamic Model for Multi-Step Prediction,"
2018 IEEE International Conference on Robotics and Automation (ICRA),
Brisbane, QLD, Australia, 2018, pp. 2454-2459, doi: 10.1109/ICRA.2018.8460840.

|------------>               |---------------->
Input Model -> Physics Model -> Output Model -> + ->
            |----------------->
"""
import abc
import json
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray
from pydantic import BaseModel
from torch import nn
from torch.nn.functional import mse_loss
from torch.optim import Adam
from torch.utils.data import DataLoader

from deepsysid.models import utils
from deepsysid.models.base import (
    DynamicIdentificationModel,
    DynamicIdentificationModelConfig,
)
from deepsysid.models.datasets import RecurrentInitializerPredictorDataset
from deepsysid.networks.rnn import InitializerPredictorLSTM

logger = logging.getLogger(__name__)


class PhysicsExternalInputModel(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, control: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def expected_states(self) -> List[str]:
        pass

    @abc.abstractmethod
    def external_input_size(self) -> int:
        pass


class Maneuvering4DOFForceInputComponentConfig(BaseModel):
    m: float
    g: float
    rho_water: float
    disp: float
    gm: float
    ixx: float
    izz: float
    xg: float
    zg: float
    xud: float
    yvd: float
    ypd: float
    yrd: float
    kvd: float
    kpd: float
    krd: float
    nvd: float
    npd: float
    nrd: float


class Maneuvering4DOFForceInputComponent(PhysicsExternalInputModel):
    STATES = ['u', 'v', 'p', 'r', 'phi']

    def __init__(
        self, time_delta: float, config: Maneuvering4DOFForceInputComponentConfig
    ):
        super().__init__()

        self.time_delta = time_delta

        m = config.m
        g = config.g
        rho_water = config.rho_water
        disp = config.disp
        gm = config.gm

        ixx = config.ixx
        izz = config.izz
        xg = config.xg
        zg = config.zg

        xud = config.xud
        yvd = config.yvd
        ypd = config.ypd
        yrd = config.yrd
        kvd = config.kvd
        kpd = config.kpd
        krd = config.krd
        nvd = config.nvd
        npd = config.npd
        nrd = config.nrd

        m_rb = torch.tensor(
            [
                [m, 0.0, 0.0, 0.0],
                [0.0, m, -m * zg, m * xg],
                [0.0, -m * zg, ixx, 0.0],
                [0.0, m * xg, 0.0, izz],
            ]
        ).float()

        m_a = torch.tensor(
            [
                [xud, 0.0, 0.0, 0.0],
                [0.0, yvd, ypd, yrd],
                [0.0, kvd, kpd, krd],
                [0.0, nvd, npd, nrd],
            ]
        ).float()

        self.inv_mass = nn.Parameter(torch.inverse(m_rb + m_a).t())
        self.inv_mass.requires_grad = False

        self.crb = nn.Parameter(
            torch.tensor(
                [
                    [0.0, -m, m * zg, -m * xg],
                    [m, 0.0, 0.0, 0.0],
                    [-m * zg, 0.0, 0.0, 0.0],
                    [m * xg, 0.0, 0.0, 0.0],
                ]
            )
            .float()
            .t()
        )
        self.crb.requires_grad = False

        self.buoyancy = nn.Parameter(
            torch.tensor(
                [
                    [0.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 0.0],
                    [
                        0.0,
                        0.0,
                        rho_water * g * disp * gm,
                        0.0,
                    ],
                    [0.0, 0.0, 0.0, 0.0],
                ]
            ).float()
        )
        self.buoyancy.requires_grad = False

    def forward(self, control: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        velocity = state[:, :, :4]
        position = torch.zeros(
            (state.shape[0], state.shape[1], 4), device=state.device, dtype=state.dtype
        )
        position[:, :, 2] = state[:, :, 4]

        tau_crb = torch.matmul(velocity[:, :, 3].unsqueeze(2) * velocity, self.crb)
        tau_hs = torch.matmul(position, self.buoyancy)
        tau_total = control - tau_crb - tau_hs

        acceleration = torch.matmul(tau_total, self.inv_mass)

        velocity_new = velocity + self.time_delta * acceleration
        roll_new = position[:, :, 2] + self.time_delta * velocity[:, :, 2]
        state_new = torch.cat((velocity_new, roll_new.unsqueeze(2)), dim=2)
        return state_new

    def expected_states(self) -> List[str]:
        return self.STATES

    def external_input_size(self) -> int:
        return 4


class QuadcopterForceInputComponentConfig(BaseModel):
    m: float
    g: float
    l: float
    d: float
    ixx: float
    izz: float
    kr: float
    kt: float


class QuadcopterForceInputComponent(PhysicsExternalInputModel):
    """
    Equations (19-30) in
    "Freddi et al.
    A Feedback Linearization Approach to Fault Tolerance in Quadrotor Vehicles.
    IFAC World Congress. 2011."

    Control forces u_f, tau_q, tau_r are computed by external model
        such as a neural network.
    """

    STATES = ['phi', 'theta', 'psi', 'p', 'q', 'r', 'dx', 'dy', 'dz']

    def __init__(
        self, time_delta: float, config: QuadcopterForceInputComponentConfig
    ) -> None:
        super().__init__()

        self.time_delta = time_delta
        self.m = config.m
        self.g = config.g
        self.length = config.l
        self.d = config.m
        self.ixx = config.ixx
        self.izz = config.izz
        self.kr = config.kr
        self.kt = config.kt

    def forward(self, control: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        :control: thruster forces computed by neural network
        :state: (batch, time, len(self.STATES))
        :return: (batch, time, len(self.STATES))
        """
        uf = control[:, :, 0]
        tauq = control[:, :, 1]
        taur = control[:, :, 2]

        x1 = state[:, :, 0]
        x2 = state[:, :, 1]
        x3 = state[:, :, 2]
        x4 = state[:, :, 3]
        x5 = state[:, :, 4]
        x6 = state[:, :, 5]
        x10 = state[:, :, 6]
        x11 = state[:, :, 7]
        x12 = state[:, :, 8]

        cx1 = torch.cos(x1)
        sx1 = torch.sin(x1)
        cx2 = torch.cos(x2)
        sx2 = torch.sin(x2)
        tx2 = torch.tan(x2)
        cx3 = torch.cos(x3)
        sx3 = torch.sin(x3)

        x1d = x4 + x5 * sx1 * tx2 + x6 * cx1 * tx2
        x2d = x5 * cx1 - x6 * sx1
        x3d = 1.0 / cx2 * (x5 * sx1 + x6 * cx1)
        x4d = (
            1.0
            / self.ixx
            * (
                -self.kr * x4
                - x5 * x6 * (self.izz - self.ixx)
                + 0.5 * self.length * (uf - taur / self.d)
            )
        )
        x5d = 1.0 / self.ixx * (-self.kr * x4 - x5 * x6 * (self.izz - self.izz) + tauq)
        x6d = 1.0 / self.ixx * (-self.kr * x6 + taur)
        x10d = (
            1.0 / self.m * (cx1 * sx2 * cx3 + sx1 * sx3) * uf - self.kt / self.m * x10
        )
        x11d = (
            1.0 / self.m * (cx1 * sx2 * sx3 - sx1 * cx3) * uf - self.kt / self.m * x11
        )
        x12d = 1.0 / self.m * (uf * cx1 * cx2 - self.kt * x12 - self.m * self.g)

        x1next = x1 + self.time_delta * x1d
        x2next = x2 + self.time_delta * x2d
        x3next = x3 + self.time_delta * x3d
        x4next = x4 + self.time_delta * x4d
        x5next = x5 + self.time_delta * x5d
        x6next = x6 + self.time_delta * x6d
        x10next = x10 + self.time_delta * x10d
        x11next = x11 + self.time_delta * x11d
        x12next = x12 + self.time_delta * x12d

        return torch.cat(
            (
                x1next.unsqueeze(2),
                x2next.unsqueeze(2),
                x3next.unsqueeze(2),
                x4next.unsqueeze(2),
                x5next.unsqueeze(2),
                x6next.unsqueeze(2),
                x10next.unsqueeze(2),
                x11next.unsqueeze(2),
                x12next.unsqueeze(2),
            ),
            dim=2,
        )

    def expected_states(self) -> List[str]:
        return self.STATES

    def external_input_size(self) -> int:
        return 3


class SerialParallelNetwork(nn.Module):
    def __init__(
        self,
        physics_model: PhysicsExternalInputModel,
        control_dim: int,
        state_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        dropout: float,
        state_normalization_gain: torch.Tensor,
    ) -> None:
        super().__init__()

        self.input_model = InitializerPredictorLSTM(
            predictor_input_dim=control_dim + state_dim,
            initializer_input_dim=control_dim + state_dim,
            output_dim=physics_model.external_input_size(),
            recurrent_dim=recurrent_dim,
            num_recurrent_layers=num_recurrent_layers,
            dropout=dropout,
        )

        self.physics_model = physics_model

        self.output_model = InitializerPredictorLSTM(
            initializer_input_dim=control_dim + state_dim,
            predictor_input_dim=control_dim + 2 * state_dim,
            output_dim=state_dim,
            recurrent_dim=recurrent_dim,
            num_recurrent_layers=num_recurrent_layers,
            dropout=dropout,
        )

        self.state_normalization_gain = state_normalization_gain

    def forward_teacher(
        self,
        initial_control: torch.Tensor,
        initial_state: torch.Tensor,
        controls: torch.Tensor,
        states: torch.Tensor,
    ) -> torch.Tensor:
        """
        :initial_control: (batch, window, control)
        :initial_state: (batch, window, state)
        :controls: (batch, horizon, control)
        :states: (batch, horizon, state) divided by state_normalization_gain
        :return: (batch, horizon, state)
        """
        window = torch.cat((initial_control, initial_state), dim=2)
        horizon = torch.cat((controls, states), dim=2)

        physics_input = self.input_model.forward(horizon, window)

        physics_output = self.physics_model.forward(
            physics_input, self.state_normalization_gain * states
        )
        physics_output = 1.0 / self.state_normalization_gain * physics_output

        state_residuals = self.output_model.forward(
            predictor_input=torch.cat((horizon, physics_output), dim=2),
            initializer_input=window,
        )
        prediction: torch.Tensor = physics_output + state_residuals

        return prediction

    def forward_feedback(
        self,
        initial_control: torch.Tensor,
        initial_state: torch.Tensor,
        controls: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """
        :initial_control: (batch, window, control)
        :initial_state: (batch, window, state)
        :controls: (batch, horizon, control)
        :state: (batch, state)
        :return: (batch, horizon, state)
        """
        predicted_states = torch.zeros(
            (initial_state.shape[0], controls.shape[1], initial_state.shape[2]),
            device=initial_state.device,
            dtype=initial_state.dtype,
        )
        for time in range(controls.shape[1]):
            new_state = self.forward_teacher(
                initial_control=initial_control,
                initial_state=initial_state,
                controls=controls[:, time, :].unsqueeze(1),
                states=state.unsqueeze(1),
            )
            state = new_state.squeeze(1)
            predicted_states[:, time, :] = state
            initial_control = torch.cat(
                (initial_control[:, 1:, :], controls[:, time, :].unsqueeze(1)), dim=1
            )
            initial_state = torch.cat(
                (initial_state[:, 1:, :], state.unsqueeze(1)), dim=1
            )

        return predicted_states

    def forward(self) -> None:
        raise NotImplementedError('Use either forward_teacher or forward_feedback.')


class SerialParallelHybridModelConfig(DynamicIdentificationModelConfig):
    epochs: int
    batch_size: int
    sequence_length: int
    learning_rate: float
    recurrent_dim: int
    num_recurrent_layers: int
    dropout: float


class SerialParallelHybridModel(DynamicIdentificationModel, metaclass=abc.ABCMeta):
    CONFIG = SerialParallelHybridModelConfig

    def __init__(
        self,
        config: SerialParallelHybridModelConfig,
        physics_model: PhysicsExternalInputModel,
    ) -> None:
        if set(physics_model.expected_states()) != set(config.state_names):
            raise ValueError(
                'physics_model expects and requires exactly these states '
                f'{physics_model.expected_states()}, however these states '
                f'were provided {config.state_names} by the configuration.'
            )

        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(config.device_name)

        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)
        self.recurrent_dim = config.recurrent_dim
        self.num_recurrent_layers = config.num_recurrent_layers
        self.dropout = config.dropout

        self.epochs = config.epochs
        self.batch_size = config.batch_size
        self.sequence_length = config.sequence_length
        self.learning_rate = config.learning_rate

        self.physics_model = physics_model

        self.physics_state_mask = np.array(
            [config.state_names.index(sn) for sn in physics_model.expected_states()],
            dtype=np.int32,
        )
        self.output_state_mask = np.array(
            [physics_model.expected_states().index(sn) for sn in config.state_names],
            dtype=np.int32,
        )

        self.network: Optional[SerialParallelNetwork] = None
        self.normalization_gains: Optional[NDArray[np.float64]] = None
        self.control_mean: Optional[NDArray[np.float64]] = None
        self.control_std: Optional[NDArray[np.float64]] = None

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Optional[Dict[str, NDArray[np.float64]]]:
        epoch_losses = []

        # Reorder the states in state_seqs to match physics_model.expected_states().
        state_seqs = [state[:, self.physics_state_mask] for state in state_seqs]

        # Compute normalization statistics.
        self.control_mean, self.control_std = utils.mean_stddev(control_seqs)
        self.normalization_gains = np.max(np.vstack(state_seqs), axis=0)

        # Normalize and prepare dataset.
        control_seqs = [
            utils.normalize(control, self.control_mean, self.control_std)
            for control in control_seqs
        ]
        state_seqs = [
            state / self.normalization_gains for state in state_seqs  # type: ignore
        ]
        dataset = RecurrentInitializerPredictorDataset(
            control_seqs=control_seqs,
            state_seqs=state_seqs,
            sequence_length=self.sequence_length,
        )

        # Initialize network and corresponding optimizer
        self.network = self._construct_network()
        self.network.train()
        optimizer = Adam(self.network.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            data_loader = DataLoader(
                dataset, self.batch_size, shuffle=True, drop_last=True
            )
            epoch_loss = 0.0
            for batch in data_loader:
                self.network.zero_grad()

                initial_control = batch['control_window'].float().to(self.device)
                initial_state = batch['state_window'].float().to(self.device)
                controls = batch['control_horizon'].float().to(self.device)
                true_states = batch['state_horizon'].float().to(self.device)
                states = torch.cat(
                    (initial_state[:, -1, :].unsqueeze(1), true_states[:, :-1, :]),
                    dim=1,
                )
                pred_states = self.network.forward_teacher(
                    initial_control=initial_control,
                    initial_state=initial_state,
                    controls=controls,
                    states=states,
                )

                batch_loss = mse_loss(pred_states, true_states)
                epoch_loss += batch_loss.item()
                batch_loss.backward()
                optimizer.step()

            logger.info(f'Epoch {epoch + 1}/{self.epochs} - ' f'Loss: {epoch_loss}')
            epoch_losses.append([epoch, epoch_loss])

        return dict(epoch_loss=np.array(epoch_losses, dtype=np.float64))

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> Union[
        NDArray[np.float64], Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]
    ]:
        if (
            self.normalization_gains is None
            or self.control_mean is None
            or self.control_std is None
            or self.physics_model is None
            or self.network is None
        ):
            raise ValueError('Model needs to be trained or loaded prior to simulate.')

        self.network.eval()

        initial_control = utils.normalize(
            initial_control, self.control_mean, self.control_std
        )
        initial_state = (
            initial_state[:, self.physics_state_mask] / self.normalization_gains
        )
        control = utils.normalize(control, self.control_mean, self.control_std)

        with torch.no_grad():
            initial_control_tch = (
                torch.from_numpy(initial_control).float().to(self.device)
            )
            initial_state_tch = torch.from_numpy(initial_state).float().to(self.device)
            state_tch = torch.from_numpy(initial_state[-1]).float().to(self.device)
            controls_tch = torch.from_numpy(control).float().to(self.device)
            pred_states_tch = self.network.forward_feedback(
                initial_control=initial_control_tch.unsqueeze(0),
                initial_state=initial_state_tch.unsqueeze(0),
                controls=controls_tch.unsqueeze(0),
                state=state_tch.unsqueeze(0),
            )

            pred_states: NDArray[np.float64] = (
                pred_states_tch.cpu().detach().squeeze().numpy().astype(np.float64)
            )

        pred_states = self.normalization_gains * pred_states
        pred_states = pred_states[:, self.output_state_mask]

        return pred_states

    def save(self, file_path: Tuple[str, ...]) -> None:
        if (
            self.network is None
            or self.normalization_gains is None
            or self.control_mean is None
            or self.control_std is None
        ):
            raise ValueError('Model needs to be trained or loaded prior to saving.')

        torch.save(self.network.state_dict(), file_path[0])
        with open(file_path[1], mode='w') as f:
            json.dump(
                {
                    'control_mean': self.control_mean.tolist(),
                    'control_std': self.control_std.tolist(),
                    'normalization_gains': self.normalization_gains.tolist(),
                },
                f,
            )

    def load(self, file_path: Tuple[str, ...]) -> None:
        with open(file_path[1], mode='r') as f:
            norm = json.load(f)
        self.control_mean = np.array(norm['control_mean'], dtype=np.float64)
        self.control_std = np.array(norm['control_std'], dtype=np.float64)
        self.normalization_gains = np.array(
            norm['normalization_gains'], dtype=np.float64
        )

        self.network = self._construct_network()
        self.network.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'pth', 'json'

    def get_parameter_count(self) -> int:
        return -1

    def _construct_network(self) -> SerialParallelNetwork:
        return SerialParallelNetwork(
            physics_model=self.physics_model.to(self.device),
            control_dim=self.control_dim,
            state_dim=self.state_dim,
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            dropout=self.dropout,
            state_normalization_gain=torch.tensor(self.normalization_gains)
            .float()
            .to(self.device),
        ).to(self.device)


class SerialParallelQuadcopterModelConfig(
    SerialParallelHybridModelConfig, QuadcopterForceInputComponentConfig
):
    pass


class SerialParallelQuadcopterModel(SerialParallelHybridModel):
    CONFIG = SerialParallelQuadcopterModelConfig

    def __init__(self, config: SerialParallelQuadcopterModelConfig) -> None:
        super().__init__(
            config,
            physics_model=QuadcopterForceInputComponent(
                time_delta=config.time_delta, config=config
            ),
        )


class SerialParallel4DOFShipModelConfig(
    SerialParallelHybridModelConfig, Maneuvering4DOFForceInputComponentConfig
):
    pass


class SerialParallel4DOFShipModel(SerialParallelHybridModel):
    CONFIG = SerialParallel4DOFShipModelConfig

    def __init__(self, config: SerialParallel4DOFShipModelConfig) -> None:
        super().__init__(
            config,
            physics_model=Maneuvering4DOFForceInputComponent(
                time_delta=config.time_delta, config=config
            ),
        )
