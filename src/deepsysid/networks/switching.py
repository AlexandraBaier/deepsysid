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
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.nn import LSTM


class SwitchingBaseLSTM(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(
        self,
        control: torch.Tensor,
        state: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        pass


class UnconstrainedSwitchingLSTM(SwitchingBaseLSTM):
    def __init__(
        self,
        control_dim: int,
        state_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.control_dim = control_dim
        self.state_dim = state_dim
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

    def forward(
        self,
        control: torch.Tensor,
        state: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        """
        :control: (batch, time, control)
        :state: (batch, state)
        :returns: (batch, time, state),
                hx:=(h,c),
                A.shape = (batch, time, state, state),
                B.shape = (batch, time, state, control)
        """
        batch_size = control.shape[0]
        sequence_length = control.shape[1]

        x, (h0, c0) = self.lstm.forward(control, hx=hx)
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
            size=(batch_size, sequence_length, self.state_dim), device=state.device
        )
        for time in range(sequence_length):
            state = (
                A[:, time] @ state.unsqueeze(-1)
                + B[:, time] @ control[:, time].unsqueeze(-1)
            ).squeeze(-1)
            state = state
            states[:, time] = state

        return states, (h0, c0), A, B


class StableSwitchingLSTM(SwitchingBaseLSTM):
    def __init__(
        self,
        control_dim: int,
        state_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.control_dim = control_dim
        self.state_dim = state_dim
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

    def forward(
        self,
        control: torch.Tensor,
        state: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[
        torch.Tensor, Tuple[torch.Tensor, torch.Tensor], torch.Tensor, torch.Tensor
    ]:
        """
        :control: (batch, time, control)
        :state: (batch, state)
        :returns: (batch, time, state),
                hx:=(h,c),
                A.shape = (batch, time, state, state),
                B.shape = (batch, time, state, control)
        """
        batch_size = control.shape[0]
        sequence_length = control.shape[1]

        x, (h0, c0) = self.lstm.forward(control, hx=hx)
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
            size=(batch_size, sequence_length, self.state_dim), device=state.device
        )
        for time in range(sequence_length):
            state = A[:, time] @ state.unsqueeze(-1) + B[:, time] @ control[
                :, time
            ].unsqueeze(-1)
            state = state.squeeze(-1)
            states[:, time] = state

        return states, (h0, c0), A, B
