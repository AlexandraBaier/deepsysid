import abc
import json
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import NDArray

from ...tracker.base import BaseEventTracker
from .. import utils
from ..base import DynamicIdentificationModel, DynamicIdentificationModelConfig
from .component_interfacing import ComponentMapper
from .physical import (
    BasicPelicanMotionComponent,
    BasicPelicanMotionConfig,
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
    QuadraticComponent,
    SemiphysicalComponent,
)


class PhysicsOnlyModel(DynamicIdentificationModel, abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        physical: PhysicalComponent,
        semiphysical: SemiphysicalComponent,
        config: DynamicIdentificationModelConfig,
    ) -> None:
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(config.device_name)
        self.state_dim = len(config.state_names)
        self.time_delta = config.time_delta

        self.physical = physical
        self.semiphysical = semiphysical

        self.physical_mapper = ComponentMapper(
            control_variables=config.control_names,
            state_variables=config.state_names,
            component=physical,
        )
        self.semiphysical_mapper = ComponentMapper(
            control_variables=config.control_names,
            state_variables=config.state_names,
            component=semiphysical,
        )

        self.state_mean: Optional[NDArray[np.float64]] = None
        self.state_std: Optional[NDArray[np.float64]] = None
        self.control_mean: Optional[NDArray[np.float64]] = None
        self.control_std: Optional[NDArray[np.float64]] = None

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]] = None,
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> Optional[Dict[str, NDArray[np.float64]]]:
        self.physical.train()
        self.semiphysical.train()

        self.control_mean, self.control_std = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_std = utils.mean_stddev(state_seqs)

        un_control_seqs = control_seqs
        un_state_seqs = state_seqs

        state_std = self.state_std

        def scale_acc_physical_np(x: NDArray[np.float64]) -> NDArray[np.float64]:
            out: NDArray[np.float64] = x / self.physical_mapper.get_expected_state(
                state_std
            )
            return out

        # Train linear model
        targets_seqs = []
        # Prepare target values for training of semiphysical model.
        for un_control, un_state in zip(un_control_seqs, un_state_seqs):
            target = utils.normalize(un_state[1:], self.state_mean, self.state_std)

            self.physical_mapper.set_provided_state(
                target,
                self.physical_mapper.get_expected_state(target)
                - scale_acc_physical_np(
                    self.physical.time_delta
                    * self.physical.forward(
                        torch.from_numpy(
                            self.physical_mapper.get_expected_control(un_control[1:])
                        )
                        .float()
                        .to(self.device),
                        torch.from_numpy(
                            self.physical_mapper.get_expected_state(un_state[:-1])
                        )
                        .float()
                        .to(self.device),
                    )
                    .cpu()
                    .detach()
                    .numpy()
                ),
            )
            targets_seqs.append(self.semiphysical_mapper.get_expected_state(target))

        self.semiphysical.train_semiphysical(
            control_seqs=[
                self.semiphysical_mapper.get_expected_control(cseq)
                for cseq in un_control_seqs
            ],
            state_seqs=[
                self.semiphysical_mapper.get_expected_state(sseq)
                for sseq in un_state_seqs
            ],
            target_seqs=targets_seqs,
        )

        return None

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

        state_mean = torch.from_numpy(self.state_mean).float().to(self.device)
        state_std = torch.from_numpy(self.state_std).float().to(self.device)

        def denormalize_state(x: torch.Tensor) -> torch.Tensor:
            return (x * state_std) + state_mean

        def scale_acc_physical(x: torch.Tensor) -> torch.Tensor:
            return x / self.physical_mapper.get_expected_state(state_std)

        un_control = control
        current_state_np = initial_state[-1, :]
        control = utils.normalize(control, self.control_mean, self.control_std)

        y = np.zeros((control.shape[0], self.state_dim), dtype=np.float64)
        whitebox = np.zeros((control.shape[0], self.state_dim), dtype=np.float64)

        with torch.no_grad():
            x_control_un = (
                torch.from_numpy(un_control).unsqueeze(0).float().to(self.device)
            )
            current_state = (
                torch.from_numpy(current_state_np).unsqueeze(0).float().to(self.device)
            )
            for time in range(control.shape[0]):
                y_whitebox = (
                    torch.zeros((current_state.shape[0], self.state_dim))
                    .float()
                    .to(self.device)
                )
                y_semiphysical = self.semiphysical.forward(
                    control=self.semiphysical_mapper.get_expected_control(
                        x_control_un[:, time]
                    ),
                    state=self.semiphysical_mapper.get_expected_state(current_state),
                )
                ydot_physical = scale_acc_physical(
                    self.physical.forward(
                        self.physical_mapper.get_expected_control(
                            x_control_un[:, time]
                        ),
                        self.physical_mapper.get_expected_state(current_state),
                    )
                )
                self.physical_mapper.set_provided_state(
                    y_whitebox,
                    self.physical_mapper.get_expected_state(y_whitebox)
                    + self.time_delta * ydot_physical,
                )
                self.semiphysical_mapper.set_provided_state(
                    y_whitebox,
                    self.semiphysical_mapper.get_expected_state(y_whitebox)
                    + y_semiphysical,
                )

                current_state = denormalize_state(y_whitebox)
                y[time, :] = current_state.cpu().detach().numpy()
                whitebox[time, :] = y_whitebox.cpu().detach().numpy()

        return y

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

        torch.save(self.semiphysical.state_dict(), file_path[0])

        semiphysical_params = [
            param.tolist() for param in self.semiphysical.get_parameters_to_save()
        ]

        with open(file_path[1], mode='w') as f:
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

    def load(self, file_path: Tuple[str, ...]) -> None:
        self.semiphysical.load_state_dict(
            torch.load(file_path[0], map_location=self.device_name)
        )

        with open(file_path[1], mode='r') as f:
            norm = json.load(f)
        self.state_mean = np.array(norm['state_mean'], dtype=np.float64)
        self.state_std = np.array(norm['state_std'], dtype=np.float64)
        self.control_mean = np.array(norm['control_mean'], dtype=np.float64)
        self.control_std = np.array(norm['control_std'], dtype=np.float64)
        self.semiphysical.load_parameters(
            [np.array(param, dtype=np.float64) for param in norm['semiphysical']]
        )

    def get_file_extension(self) -> Tuple[str, ...]:
        return 'pth', 'json'

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.semiphysical.parameters())


class PhysicsOnlyLinearModel(PhysicsOnlyModel):
    def __init__(self, config: DynamicIdentificationModelConfig):
        device = torch.device(config.device_name)
        physical = NoOpPhysicalComponent(time_delta=config.time_delta, device=device)
        semiphysical = LinearComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )
        super().__init__(physical=physical, semiphysical=semiphysical, config=config)


class PhysicsOnlyQuadraticModel(PhysicsOnlyModel):
    def __init__(self, config: DynamicIdentificationModelConfig):
        device = torch.device(config.device_name)
        physical = NoOpPhysicalComponent(time_delta=config.time_delta, device=device)
        semiphysical = QuadraticComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )
        super().__init__(physical=physical, semiphysical=semiphysical, config=config)


class PhysicsOnlyBlankeModel(PhysicsOnlyModel):
    def __init__(self, config: DynamicIdentificationModelConfig):
        device = torch.device(config.device_name)
        physical = NoOpPhysicalComponent(time_delta=config.time_delta, device=device)
        semiphysical = BlankeComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )
        super().__init__(physical=physical, semiphysical=semiphysical, config=config)


class PhysicsOnlyMinimalManeuveringConfig(
    DynamicIdentificationModelConfig, MinimalManeuveringConfig
):
    pass


class PhysicsOnlyMinimalManeuveringModel(PhysicsOnlyModel):
    CONFIG = PhysicsOnlyMinimalManeuveringConfig

    def __init__(self, config: PhysicsOnlyMinimalManeuveringConfig):
        device = torch.device(config.device_name)
        physical = MinimalManeuveringComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = NoOpSemiphysicalComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )
        super().__init__(physical=physical, semiphysical=semiphysical, config=config)


class PhysicsOnlyLinearMinimalManeuveringModel(PhysicsOnlyModel):
    CONFIG = PhysicsOnlyMinimalManeuveringConfig

    def __init__(self, config: PhysicsOnlyMinimalManeuveringConfig):
        device = torch.device(config.device_name)
        physical = MinimalManeuveringComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = LinearComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )
        super().__init__(physical=physical, semiphysical=semiphysical, config=config)


class PhysicsOnlyBlankeMinimalManeuveringModel(PhysicsOnlyModel):
    CONFIG = PhysicsOnlyMinimalManeuveringConfig

    def __init__(self, config: PhysicsOnlyMinimalManeuveringConfig):
        device = torch.device(config.device_name)
        physical = MinimalManeuveringComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = BlankeComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )
        super().__init__(physical=physical, semiphysical=semiphysical, config=config)


class PhysicsOnlyPropulsionManeuveringConfig(
    DynamicIdentificationModelConfig, PropulsionManeuveringConfig
):
    pass


class PhysicsOnlyPropulsionManeuveringModel(PhysicsOnlyModel):
    CONFIG = PhysicsOnlyPropulsionManeuveringConfig

    def __init__(self, config: PhysicsOnlyPropulsionManeuveringConfig):
        device = torch.device(config.device_name)
        physical = PropulsionManeuveringComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = NoOpSemiphysicalComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )
        super().__init__(physical=physical, semiphysical=semiphysical, config=config)


class PhysicsOnlyLinearPropulsionManeuveringModel(PhysicsOnlyModel):
    CONFIG = PhysicsOnlyPropulsionManeuveringConfig

    def __init__(self, config: PhysicsOnlyPropulsionManeuveringConfig):
        device = torch.device(config.device_name)
        physical = PropulsionManeuveringComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = LinearComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )
        super().__init__(physical=physical, semiphysical=semiphysical, config=config)


class PhysicsOnlyBlankePropulsionManeuveringModel(PhysicsOnlyModel):
    CONFIG = PhysicsOnlyPropulsionManeuveringConfig

    def __init__(self, config: PhysicsOnlyPropulsionManeuveringConfig):
        device = torch.device(config.device_name)
        physical = PropulsionManeuveringComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = BlankeComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )
        super().__init__(physical=physical, semiphysical=semiphysical, config=config)


class PhysicsOnlyBasicPelicanMotionModelConfig(
    DynamicIdentificationModelConfig, BasicPelicanMotionConfig
):
    pass


class PhysicsOnlyBasicPelicanMotionModel(PhysicsOnlyModel):
    CONFIG = PhysicsOnlyBasicPelicanMotionModelConfig

    def __init__(self, config: PhysicsOnlyBasicPelicanMotionModelConfig):
        device = torch.device(config.device_name)
        physical = BasicPelicanMotionComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = NoOpSemiphysicalComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )
        super().__init__(physical=physical, semiphysical=semiphysical, config=config)


class PhysicsOnlyLinearBasicPelicanMotionModel(PhysicsOnlyModel):
    CONFIG = PhysicsOnlyBasicPelicanMotionModelConfig

    def __init__(self, config: PhysicsOnlyBasicPelicanMotionModelConfig):
        device = torch.device(config.device_name)
        physical = BasicPelicanMotionComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = LinearComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )
        super().__init__(physical=physical, semiphysical=semiphysical, config=config)


class PhysicsOnlyQuadraticBasicPelicanMotionModel(PhysicsOnlyModel):
    CONFIG = PhysicsOnlyBasicPelicanMotionModelConfig

    def __init__(self, config: PhysicsOnlyBasicPelicanMotionModelConfig):
        device = torch.device(config.device_name)
        physical = BasicPelicanMotionComponent(
            time_delta=config.time_delta, device=device, config=config
        )
        semiphysical = QuadraticComponent(
            control_dim=len(config.control_names),
            state_dim=len(config.state_names),
            device=device,
        )
        super().__init__(physical=physical, semiphysical=semiphysical, config=config)
