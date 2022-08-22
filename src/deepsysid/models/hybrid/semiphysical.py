import abc
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import r2_score
from torch import nn

from ... import utils

logger = logging.getLogger()


class SemiphysicalComponent(nn.Module, abc.ABC):
    STATES: Optional[List[str]] = None
    CONTROLS: Optional[List[str]] = None

    def __init__(self, control_dim: int, state_dim: int, device: torch.device):
        super().__init__()

        self.control_dim = control_dim
        self.state_dim = state_dim
        self.device = device

    @abc.abstractmethod
    def forward(self, control: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        :param control: shape (N, _)
        :param state: shape (N, S)
        :return: (N, S)
        """
        pass

    @abc.abstractmethod
    def train_semiphysical(
        self,
        control_seqs: List[np.ndarray],
        state_seqs: List[np.ndarray],
        target_seqs: List[np.ndarray],
    ):
        pass

    def get_parameters_to_save(self) -> List[np.ndarray]:
        return []

    def load_parameters(self, parameters: List[np.ndarray]):
        pass


class NoOpSemiphysicalComponent(SemiphysicalComponent):
    def forward(self, control: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(state)

    def train_semiphysical(
        self,
        control_seqs: List[np.ndarray],
        state_seqs: List[np.ndarray],
        target_seqs: List[np.ndarray],
    ):
        pass


class LinearComponent(SemiphysicalComponent):
    def __init__(self, control_dim: int, state_dim: int, device: torch.device):
        super().__init__(control_dim, state_dim, device)

        self.model = (
            nn.Linear(
                in_features=self.get_semiphysical_features(),
                out_features=state_dim,
                bias=False,
            )
            .float()
            .to(self.device)
        )
        self.model.weight.requires_grad = False

        self.semiphysical_mean: Optional[np.ndarray] = None
        self.semiphysical_std: Optional[np.ndarray] = None

    def forward(self, control: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        if self.semiphysical_mean is None or self.semiphysical_std is None:
            raise ValueError(
                'Semiphysical component has not been trained prior to forward.'
            )

        semiphysical_mean = (
            torch.from_numpy(self.semiphysical_mean).float().to(self.device)
        )
        semiphysical_std = (
            torch.from_numpy(self.semiphysical_std).float().to(self.device)
        )

        semiphysical_in = utils.normalize(
            self.expand_semiphysical_input(control, state),
            semiphysical_mean,
            semiphysical_std,
        )

        return self.model.forward(semiphysical_in)

    def get_semiphysical_features(self) -> int:
        return self.control_dim + self.state_dim

    def expand_semiphysical_input(
        self, control: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((control, state), dim=1)

    def train_semiphysical(
        self,
        control_seqs: List[np.ndarray],
        state_seqs: List[np.ndarray],
        target_seqs: List[np.ndarray],
    ):
        semiphysical_in_seqs = [
            self.expand_semiphysical_input(
                torch.from_numpy(control[1:]), torch.from_numpy(state[:-1])
            )
            .cpu()
            .detach()
            .numpy()
            for control, state in zip(control_seqs, state_seqs)
        ]

        self.semiphysical_mean, self.semiphysical_std = utils.mean_stddev(
            semiphysical_in_seqs
        )

        semiphysical_in_seqs = [
            utils.normalize(
                semiphysical_in, self.semiphysical_mean, self.semiphysical_std
            )
            for semiphysical_in in semiphysical_in_seqs
        ]

        train_x = np.vstack(semiphysical_in_seqs)
        train_y = np.vstack(target_seqs)

        # No intercept in linear time invariant systems
        regressor = LinearRegression(fit_intercept=False)
        regressor.fit(train_x, train_y)
        linear_fit = r2_score(
            regressor.predict(train_x), train_y, multioutput='uniform_average'
        )
        logger.info(f'Whitebox R2 Score: {linear_fit}')

        self.model.weight = nn.Parameter(
            torch.from_numpy(regressor.coef_).float().to(self.device),
            requires_grad=False,
        )

    def get_parameters_to_save(self) -> List[np.ndarray]:
        if self.semiphysical_mean is None or self.semiphysical_std is None:
            raise ValueError(
                'Semiphysical component has not been trained '
                'prior to get_parameters_to_save.'
            )
        return [self.semiphysical_mean, self.semiphysical_std]

    def load_parameters(self, parameters: List[np.ndarray]):
        self.semiphysical_mean = parameters[0]
        self.semiphysical_std = parameters[1]


class QuadraticComponent(LinearComponent):
    def get_semiphysical_features(self) -> int:
        return 2 * (self.control_dim + self.state_dim)

    def expand_semiphysical_input(
        self, control: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((control, state, control * control, state * state), dim=1)


class BlankeComponent(LinearComponent):
    STATES = ['u', 'v', 'p', 'r', 'phi']

    def __init__(self, control_dim: int, state_dim: int, device: torch.device):
        super().__init__(control_dim, state_dim, device)

        self.model = (
            nn.Linear(
                in_features=self.get_semiphysical_features(),
                out_features=len(BlankeComponent.STATES),
                bias=False,
            )
            .float()
            .to(self.device)
        )
        self.model.weight.requires_grad = False

    def train_semiphysical(
        self,
        control_seqs: List[np.ndarray],
        state_seqs: List[np.ndarray],
        target_seqs: List[np.ndarray],
    ):
        semiphysical_in_seqs = [
            self.expand_semiphysical_input(
                torch.from_numpy(control[1:]), torch.from_numpy(state[:-1])
            )
            .cpu()
            .detach()
            .numpy()
            for control, state in zip(control_seqs, state_seqs)
        ]

        self.semiphysical_mean, self.semiphysical_std = utils.mean_stddev(
            semiphysical_in_seqs
        )

        semiphysical_in_seqs = [
            utils.normalize(
                semiphysical_in, self.semiphysical_mean, self.semiphysical_std
            )
            for semiphysical_in in semiphysical_in_seqs
        ]

        train_x = np.vstack(semiphysical_in_seqs)
        train_y = np.vstack(target_seqs)

        # Train each dimension as separate equation
        def train_dimension(dim_mask: Tuple[int, ...], dim_name: str, dim_idx: int):
            regressor = RidgeCV(
                alphas=np.logspace(-6, 6, 25),
                fit_intercept=False,
            )
            regressor.fit(train_x[:, dim_mask], train_y[:, dim_idx])
            linear_fit = r2_score(
                regressor.predict(train_x[:, dim_mask]),
                train_y[:, dim_idx],
                multioutput='uniform_average',
            )
            logger.info(f'Whitebox R2 Score ({dim_name}): {linear_fit}')
            logger.info(f'Chosen L2 coefficient ({dim_name}): {regressor.alpha_}')
            return regressor

        mask_u = (0, 5, 6)
        mask_v = (1, 7, 8, 9, 10, 11, 12, 13, 14)
        mask_r = (3, 7, 11, 15, 16, 17, 18, 19)
        mask_phi = (2, 4)
        reg_u = train_dimension(mask_u, 'u', self.STATES.index('u'))
        reg_v = train_dimension(mask_v, 'v', self.STATES.index('v'))
        reg_r = train_dimension(mask_r, 'r', self.STATES.index('r'))
        reg_phi = train_dimension(mask_phi, 'phi', self.STATES.index('phi'))

        weight = np.zeros(
            (len(BlankeComponent.STATES), self.get_semiphysical_features())
        )
        weight[self.STATES.index('u'), mask_u] = reg_u.coef_
        weight[self.STATES.index('v'), mask_v] = reg_v.coef_
        weight[self.STATES.index('r'), mask_r] = reg_r.coef_
        weight[self.STATES.index('phi'), mask_phi] = reg_phi.coef_

        self.model.weight = nn.Parameter(
            torch.from_numpy(weight).float().to(self.device), requires_grad=False
        )

    def get_semiphysical_features(self) -> int:
        return 20 + self.control_dim

    def expand_semiphysical_input(
        self, control: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        u = state[:, self.STATES.index('u')].unsqueeze(1)
        v = state[:, self.STATES.index('v')].unsqueeze(1)
        p = state[:, self.STATES.index('p')].unsqueeze(1)
        r = state[:, self.STATES.index('r')].unsqueeze(1)
        phi = state[:, self.STATES.index('phi')].unsqueeze(1)
        au = torch.abs(u)
        av = torch.abs(v)
        ar = torch.abs(r)
        auv = torch.abs(u * v)
        aur = torch.abs(u * r)
        auphi = torch.abs(u * phi)

        state = torch.cat(
            (
                u,  # 0: X
                v,  # 1: Y
                p,  # 2: phi
                r,  # 3: N
                phi,  # 4: phi
                au * u,  # 5: X
                v * r,  # 6: X
                au * v,  # 7: Y, N
                u * r,  # 8: Y
                av * v,  # 9: Y
                ar * v,  # 10: Y
                av * r,  # 11: Y, N
                auv * phi,  # 12: Y
                aur * phi,  # 13: Y
                u * u * phi,  # 14: Y
                au * r,  # 15: N
                ar * r,  # 16: N
                auphi * phi,  # 17: N
                ar * u * phi,  # 18: N
                au * u * phi,  # 19: N
            ),
            dim=1,
        )
        return torch.cat((control, state), dim=1)
