"""
Switching mechanisms
- mask(Linear(h, c)) (model trained via gradient descent)
  - mask=softmax
  - mask=norm
- k-means (inverse distance to clusters as weighting)
  - gradient descent based training
  - least-squares based training

"""

import abc
import dataclasses
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import LSTM


@dataclasses.dataclass
class SwitchingLSTMOutput:
    outputs: torch.Tensor
    states: torch.Tensor
    hx: Tuple[torch.Tensor, torch.Tensor]
    system_matrices: torch.Tensor
    control_matrices: torch.Tensor

    def __post_init__(self) -> None:
        if not (len(self.outputs.shape) == 3):
            raise ValueError(
                f'outputs should be a 3-dimensional tensor (batch, time, output), '
                f'but is not: {self.outputs.shape=}.'
            )
        if not (len(self.states.shape) == 3):
            raise ValueError(
                f'states should be a 3-dimensional tensor (batch, time, state), '
                f'but is not: {self.states.shape=}.'
            )
        if not (len(self.system_matrices.shape) == 4):
            raise ValueError(
                f'system_matrices should be a 4-dimensional tensor '
                f'(batch, time, state, state) but is not: '
                f'{self.system_matrices.shape=}.'
            )
        if not (len(self.control_matrices.shape) == 4):
            raise ValueError(
                f'control_matrices should be a 4-dimensional tensor '
                f'(batch, time, state, control) but is not: '
                f'{self.control_matrices.shape=}.'
            )
        if not (
            self.outputs.shape[0]
            == self.states.shape[0]
            == self.system_matrices.shape[0]
            == self.control_matrices.shape[0]
        ):
            raise ValueError(
                f'Batch dimension (dimension 0) of outputs, states, system_matrices '
                f'and control_matrices needs to match, but does not match: '
                f'{self.outputs.shape=}, '
                f'{self.states.shape=}, '
                f'{self.system_matrices.shape=}, '
                f'{self.control_matrices.shape=}.'
            )

        if not (
            self.outputs.shape[1]
            == self.states.shape[1]
            == self.system_matrices.shape[1]
            == self.control_matrices.shape[1]
        ):
            raise ValueError(
                f'Time dimension (dimension 1) of outputs, states, system_matrices '
                f'and control_matrices needs to match, but does not match: '
                f'{self.outputs.shape=}, '
                f'{self.states.shape=}, '
                f'{self.system_matrices.shape=}, '
                f'{self.control_matrices.shape=}.'
            )

        if not (
            self.states.shape[2]
            == self.system_matrices.shape[2]
            == self.control_matrices.shape[2]
        ):
            raise ValueError(
                f'State dimension (dimension 2) of states, system_matrices '
                f'and control_matrices needs to match, but does not match: '
                f'{self.states.shape=}, {self.system_matrices.shape=}, '
                f'{self.control_matrices.shape=}.'
            )


class SwitchingBaseLSTM(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(
        self,
        control: torch.Tensor,
        previous_output: torch.Tensor,
        previous_state: Optional[torch.Tensor] = None,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> SwitchingLSTMOutput:
        """
        :control: (batch, time, control)
        :previous_output: (batch, output)
        :previous_state: (batch, state) or None
        :hx: hx = (h0, c0) or None
        :returns: SwitchingLSTMOutput
            with .outputs.shape = (batch, time, output)
                 .states.shape = (batch, time, state)
                 .system_matrices = (batch, time, state, state)
                 .control_matrices = (batch, time, state, control)
        """
        pass

    @property
    @abc.abstractmethod
    def output_matrix(self) -> torch.Tensor:
        """
        :returns: .shape = (output, state)
        """
        pass

    @property
    @abc.abstractmethod
    def control_dimension(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def state_dimension(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def output_dimension(self) -> int:
        pass


class UnconstrainedSwitchingLSTM(SwitchingBaseLSTM):
    def __init__(
        self,
        control_dim: int,
        state_dim: int,
        output_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if not (state_dim >= output_dim):
            raise ValueError(
                'state_dim must be larger or equal to output_dim, '
                f'but {state_dim=} < {output_dim}.'
            )

        self.control_dim = control_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.recurrent_dim = recurrent_dim

        self.lstm = LSTM(
            input_size=control_dim,
            hidden_size=recurrent_dim,
            num_layers=num_recurrent_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.gen_A = nn.Linear(
            in_features=recurrent_dim, out_features=state_dim * state_dim, bias=True
        )
        self.gen_B = nn.Linear(
            in_features=recurrent_dim, out_features=state_dim * control_dim, bias=True
        )
        self.C = nn.Linear(in_features=state_dim, out_features=output_dim, bias=False)

    def forward(
        self,
        control: torch.Tensor,
        previous_output: torch.Tensor,
        previous_state: Optional[torch.Tensor] = None,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> SwitchingLSTMOutput:
        batch_size = control.shape[0]
        sequence_length = control.shape[1]

        x, (h0, c0) = self.lstm.forward(control, hx=hx)  # type: ignore
        x = torch.reshape(x, (batch_size * sequence_length, self.recurrent_dim))

        A = torch.reshape(
            self.gen_A.forward(x),
            (batch_size, sequence_length, self.state_dim, self.state_dim),
        )
        B = torch.reshape(
            self.gen_B.forward(x),
            (batch_size, sequence_length, self.state_dim, self.control_dim),
        )

        states = torch.zeros(
            size=(batch_size, sequence_length, self.state_dim), device=control.device
        )
        if previous_state is None:
            state = torch.zeros(
                size=(batch_size, self.state_dim), device=control.device
            )
            state[:, : self.output_dim] = previous_output
        else:
            state = previous_state

        for time in range(sequence_length):
            state = (
                A[:, time] @ state.unsqueeze(-1)
                + B[:, time] @ control[:, time].unsqueeze(-1)
            ).squeeze(-1)
            state = state
            states[:, time] = state

        outputs = self.C.forward(states)

        return SwitchingLSTMOutput(
            outputs=outputs,
            states=states,
            hx=(h0, c0),
            system_matrices=A,
            control_matrices=B,
        )

    @property
    def output_matrix(self) -> torch.Tensor:
        return self.C.weight

    @property
    def control_dimension(self) -> int:
        return self.control_dim

    @property
    def state_dimension(self) -> int:
        return self.state_dim

    @property
    def output_dimension(self) -> int:
        return self.output_dim


class UnconstrainedIdentityOutputSwitchingLSTM(UnconstrainedSwitchingLSTM):
    def __init__(
        self,
        control_dim: int,
        output_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(
            control_dim=control_dim,
            state_dim=output_dim,
            output_dim=output_dim,
            recurrent_dim=recurrent_dim,
            num_recurrent_layers=num_recurrent_layers,
            dropout=dropout,
        )

        self.C.weight = nn.Parameter(
            torch.eye(output_dim, output_dim), requires_grad=False
        )


class StableSwitchingLSTM(SwitchingBaseLSTM):
    def __init__(
        self,
        control_dim: int,
        state_dim: int,
        output_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()

        if not (state_dim >= output_dim):
            raise ValueError(
                'state_dim must be larger or equal to output_dim, '
                f'but {state_dim=} < {output_dim}.'
            )

        self.control_dim = control_dim
        self.state_dim = state_dim
        self.output_dim = output_dim
        self.recurrent_dim = recurrent_dim

        self.lstm = LSTM(
            input_size=control_dim,
            hidden_size=recurrent_dim,
            num_layers=num_recurrent_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.T = nn.Parameter(
            torch.from_numpy(np.random.normal(0, 1, (state_dim, state_dim))).float(),
            requires_grad=True,
        )

        self.gen_A = nn.Linear(
            in_features=recurrent_dim, out_features=state_dim, bias=True
        )
        self.gen_B = nn.Linear(
            in_features=recurrent_dim, out_features=state_dim * control_dim, bias=True
        )
        self.C = nn.Linear(in_features=state_dim, out_features=output_dim, bias=False)

    def forward(
        self,
        control: torch.Tensor,
        previous_output: torch.Tensor,
        previous_state: Optional[torch.Tensor] = None,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> SwitchingLSTMOutput:
        batch_size = control.shape[0]
        sequence_length = control.shape[1]

        x, (h0, c0) = self.lstm.forward(control, hx=hx)  # type: ignore
        x = torch.reshape(x, (batch_size * sequence_length, self.recurrent_dim))

        A = (
            torch.linalg.inv(self.T).unsqueeze(0).unsqueeze(0)
            @ torch.reshape(
                torch.diag_embed(torch.tanh(self.gen_A.forward(x))),
                (batch_size, sequence_length, self.state_dim, self.state_dim),
            )
            @ self.T.unsqueeze(0).unsqueeze(0)
        )
        B = torch.reshape(
            self.gen_B.forward(x),
            (batch_size, sequence_length, self.state_dim, self.control_dim),
        )

        states = torch.zeros(
            size=(batch_size, sequence_length, self.state_dim), device=control.device
        )
        if previous_state is None:
            state = torch.zeros(
                size=(batch_size, self.state_dim), device=control.device
            )
            state[:, : self.output_dim] = previous_output
        else:
            state = previous_state

        for time in range(sequence_length):
            state = A[:, time] @ state.unsqueeze(-1) + B[:, time] @ control[
                :, time
            ].unsqueeze(-1)
            state = state.squeeze(-1)
            states[:, time] = state

        outputs = self.C.forward(states)

        return SwitchingLSTMOutput(
            outputs=outputs,
            states=states,
            hx=(h0, c0),
            system_matrices=A,
            control_matrices=B,
        )

    @property
    def output_matrix(self) -> torch.Tensor:
        return self.C.weight

    @property
    def control_dimension(self) -> int:
        return self.control_dim

    @property
    def state_dimension(self) -> int:
        return self.state_dim

    @property
    def output_dimension(self) -> int:
        return self.output_dim


class StableIdentityOutputSwitchingLSTM(StableSwitchingLSTM):
    def __init__(
        self,
        control_dim: int,
        output_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        dropout: float,
    ) -> None:
        super().__init__(
            control_dim=control_dim,
            state_dim=output_dim,
            output_dim=output_dim,
            recurrent_dim=recurrent_dim,
            num_recurrent_layers=num_recurrent_layers,
            dropout=dropout,
        )

        self.C.weight = nn.Parameter(
            torch.eye(output_dim, output_dim), requires_grad=False
        )
