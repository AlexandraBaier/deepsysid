import abc
import logging
from typing import List, Optional, Tuple

import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from torch import nn

from ... import utils
from .physical import PhysicalComponent

logger = logging.getLogger()


class SemiphysicalComponent(nn.Module, abc.ABC):
    def __init__(self, control_dim: int, state_dim: int, device: torch.device):
        super().__init__()

        self.control_dim = control_dim
        self.state_dim = state_dim
        self.device = device

        self.control_mean: Optional[np.ndarray] = None
        self.control_std: Optional[np.ndarray] = None
        self.state_mean: Optional[np.ndarray] = None
        self.state_std: Optional[np.ndarray] = None

    def set_normalization_values(
        self,
        control_mean: np.ndarray,
        state_mean: np.ndarray,
        control_std: np.ndarray,
        state_std: np.ndarray
    ):
        self.control_mean = control_mean
        self.state_mean = state_mean
        self.control_std = control_std
        self.state_std = state_std

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
        physical: PhysicalComponent,
    ):
        pass


class NoOpSemiphysicalComponent(SemiphysicalComponent):
    def forward(self, control: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(state)

    def train_semiphysical(
        self,
        control_seqs: List[np.ndarray],
        state_seqs: List[np.ndarray],
        physical: PhysicalComponent,
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

    def forward(self, control: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        control_mean = torch.from_numpy(self.control_mean).float().to(self.device)
        control_std = torch.from_numpy(self.control_std).float().to(self.device)
        state_mean = torch.from_numpy(self.state_mean).float().to(self.device)
        state_std = torch.from_numpy(self.state_std).float().to(self.device)

        control = utils.normalize(control, control_mean, control_std)
        state = utils.normalize(state, state_mean, state_std)
        semiphysical_in = self.expand_semiphysical_input(control, state)
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
        physical: PhysicalComponent,
    ):
        semiphysical_in_seqs = [
            self.expand_semiphysical_input(
                torch.from_numpy(
                    utils.normalize(control[1:], self.control_mean, self.control_std)
                ),
                torch.from_numpy(
                    utils.normalize(state[:-1], self.state_mean, self.state_std)
                ),
            )
            for control, state in zip(control_seqs, state_seqs)
        ]

        train_x_list, train_y_list = [], []
        for sp_in, un_control, un_state in zip(
            semiphysical_in_seqs, control_seqs, state_seqs
        ):
            ydot_physical = (
                physical.forward(
                    torch.from_numpy(un_control[1:, :]).float().to(self.device),
                    torch.from_numpy(un_state[:-1, :]).float().to(self.device),
                )
                .cpu()
                .detach()
                .numpy()
            )
            ydot_physical = ydot_physical / self.state_std

            train_x_list.append(sp_in)
            train_y_list.append(
                utils.normalize(un_state[1:], self.state_mean, self.state_std)
                - (physical.time_delta * ydot_physical)
            )

        train_x = np.vstack(train_x_list)
        train_y = np.vstack(train_y_list)

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


class QuadraticComponent(LinearComponent):
    def get_semiphysical_features(self) -> int:
        return 2 * (self.control_dim + self.state_dim)

    def expand_semiphysical_input(
        self, control: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat((control, state, control * control, state * state), dim=1)


class BlankeComponent(LinearComponent):
    def train_semiphysical(
        self,
        control_seqs: List[np.ndarray],
        state_seqs: List[np.ndarray],
        physical: PhysicalComponent,
    ):
        semiphysical_in_seqs = [
            self.expand_semiphysical_input(
                torch.from_numpy(
                    utils.normalize(control[1:], self.control_mean, self.control_std)
                ),
                torch.from_numpy(
                    utils.normalize(state[:-1], self.state_mean, self.state_std)
                ),
            )
            for control, state in zip(control_seqs, state_seqs)
        ]

        train_x_list, train_y_list = [], []
        for sp_in, un_control, un_state in zip(
            semiphysical_in_seqs, control_seqs, state_seqs
        ):
            ydot_physical = (
                physical.forward(
                    torch.from_numpy(un_control[1:, :]).float().to(self.device),
                    torch.from_numpy(un_state[:-1, :]).float().to(self.device),
                )
                .cpu()
                .detach()
                .numpy()
            )
            ydot_physical = ydot_physical / self.state_std

            train_x_list.append(sp_in)
            train_y_list.append(
                utils.normalize(un_state[1:], self.state_mean, self.state_std)
                - (physical.time_delta * ydot_physical)
            )

        train_x = np.vstack(train_x_list)
        train_y = np.vstack(train_y_list)

        # Train each dimension as separate equation
        def train_dimension(dim_mask: Tuple[int, ...], dim_name: str, dim_idx: int):
            regressor = LinearRegression(fit_intercept=False)
            regressor.fit(train_x[:, dim_mask], train_y[:, dim_idx])
            linear_fit = r2_score(
                regressor.predict(train_x[:, dim_mask]),
                train_y[:, dim_idx],
                multioutput='uniform_average',
            )
            logger.info(f'Whitebox R2 Score ({dim_name}): {linear_fit}')
            return regressor

        mask_u = (0, 5, 6)
        mask_v = (1, 7, 8, 9, 10, 11, 12, 13, 14)
        mask_r = (3, 7, 11, 15, 16, 17, 18, 19)
        mask_phi = (2, 4)
        reg_u = train_dimension(mask_u, 'u', 0)
        reg_v = train_dimension(mask_v, 'v', 1)
        reg_r = train_dimension(mask_r, 'r', 3)
        reg_phi = train_dimension(mask_phi, 'phi', 4)

        weight = np.zeros((self.state_dim, self.get_semiphysical_in_features()))
        weight[0, mask_u] = reg_u.coef_
        weight[1, mask_v] = reg_v.coef_
        weight[3, mask_r] = reg_r.coef_
        weight[4, mask_phi] = reg_phi.coef_

        self.model.weight = nn.Parameter(
            torch.from_numpy(weight).float().to(self.device), requires_grad=False
        )

    def get_semiphysical_in_features(self) -> int:
        return 20 + self.control_dim

    def expand_semiphysical_input(
        self, control: torch.Tensor, state: torch.Tensor
    ) -> torch.Tensor:
        u = state[:, 0].unsqueeze(1)
        v = state[:, 1].unsqueeze(1)
        p = state[:, 2].unsqueeze(1)
        r = state[:, 3].unsqueeze(1)
        phi = state[:, 4].unsqueeze(1)
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
