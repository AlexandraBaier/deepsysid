import logging
from typing import List, Optional, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class BasicLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        output_dim: List[int],
        dropout: float,
    ):
        super().__init__()

        self.num_recurrent_layers = num_recurrent_layers
        self.recurrent_dim = recurrent_dim

        self.predictor_lstm = nn.LSTM(  # type: ignore
            input_size=input_dim,
            hidden_size=recurrent_dim,
            num_layers=num_recurrent_layers,
            dropout=dropout,
            batch_first=True,
        )

        layer_dim = [recurrent_dim] + output_dim
        self.out = nn.ModuleList(
            [
                nn.Linear(in_features=layer_dim[i - 1], out_features=layer_dim[i])
                for i in range(1, len(layer_dim))
            ]
        )

        for name, param in self.predictor_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        for layer in self.out:
            nn.init.xavier_normal_(layer.weight)

    def forward(
        self,
        x_pred: torch.Tensor,
        y_init: Optional[torch.Tensor] = None,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        if y_init is not None:
            h0 = y_init.new_zeros(
                (self.num_recurrent_layers, y_init.shape[0], self.recurrent_dim)
            )
            for i in range(self.num_recurrent_layers):
                h0[i, :, : y_init.shape[1]] = y_init
            c0 = h0.new_zeros(h0.shape)
            hx = (h0, c0)

        x, (h0, c0) = self.predictor_lstm(x_pred, hx)
        for layer in self.out[:-1]:
            x = F.relu(layer(x))
        x = self.out[-1](x)

        if return_state:
            return x, (h0, c0)
        else:
            return x  # type: ignore


class LinearOutputLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.num_recurrent_layers = num_recurrent_layers
        self.recurrent_dim = recurrent_dim

        self.predictor_lstm = nn.LSTM(  # type: ignore
            input_size=input_dim,
            hidden_size=recurrent_dim,
            num_layers=num_recurrent_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.out = nn.Linear(
            in_features=recurrent_dim, out_features=output_dim, bias=False
        )

        for name, param in self.predictor_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        nn.init.xavier_normal_(self.out.weight)

    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        return_state: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        x, (h0, c0) = self.predictor_lstm.forward(x_pred, hx)
        x = self.out.forward(x)

        if return_state:
            return x, (h0, c0)
        else:
            return x


class LTIRnn(nn.Module):
    def __init__(
        self, nx: int, nu: int, ny: int, nw: int, gamma: float, beta: float
    ) -> None:
        super(LTIRnn, self).__init__()

        self.nx = nx  # number of states
        self.nu = nu  # number of performance (external) input
        self.nw = nw  # number of disturbance input
        self.ny = ny  # number of performance output
        self.nz = nw  # number of disturbance output, always equal to size of w

        self.ga = gamma
        self.beta = beta
        self.nl = torch.tanh

        self.Y = torch.nn.Parameter(torch.zeros((self.nx, self.nx)))  # type: ignore
        self.A_tilde = torch.nn.Linear(self.nx, self.nx, bias=False)
        self.B1_tilde = torch.nn.Linear(self.nu, self.nx, bias=False)
        self.B2_tilde = torch.nn.Linear(self.nw, self.nx, bias=False)
        self.C1 = torch.nn.Linear(self.nx, self.ny, bias=False)
        self.D11 = torch.nn.Linear(self.nu, self.ny, bias=False)
        self.D12 = torch.nn.Linear(self.nw, self.ny, bias=False)
        self.C2_tilde = torch.nn.Linear(self.nx, self.nz, bias=False)
        self.D21_tilde = torch.nn.Linear(self.nu, self.nz, bias=False)
        self.lambdas = torch.nn.Parameter(torch.zeros((self.nw, 1)))  # type: ignore

    def initialize_lmi(self) -> None:
        # storage function
        Y = cp.Variable((self.nx, self.nx), 'Y')  # type: ignore
        # hidden state
        A_tilde = cp.Variable((self.nx, self.nx), 'A_tilde')  # type: ignore
        B1_tilde = cp.Variable((self.nx, self.nu), 'B1_tilde')  # type: ignore
        B2_tilde = cp.Variable((self.nx, self.nw), 'B2_tilde')  # type: ignore
        # output
        C1 = cp.Variable((self.ny, self.nx), 'C1')  # type: ignore
        D11 = cp.Variable((self.ny, self.nu), 'D11')  # type: ignore
        D12 = cp.Variable((self.ny, self.nw), 'D12')  # type: ignore
        # disturbance
        C2 = np.random.normal(0, 1 / np.sqrt(self.nw), size=(self.nz, self.nx))
        D21 = np.random.normal(0, 1 / np.sqrt(self.nw), size=(self.nz, self.nu))
        # multipliers
        lambdas = cp.Variable((self.nw, 1), 'tau', nonneg=True)  # type: ignore
        T = cp.diag(lambdas)  # type: ignore

        C2_tilde = T @ C2
        D21_tilde = T @ D21

        if self.ga == 0:
            # lmi that ensures finite l2 gain
            M = cp.bmat(  # type: ignore
                [
                    [-Y, self.beta * C2_tilde.T, A_tilde.T],
                    [self.beta * C2_tilde, -2 * T, B2_tilde.T],
                    [A_tilde, B2_tilde, -Y],
                ]
            )
        else:
            # lmi that ensures l2 gain gamma
            M = cp.bmat(  # type: ignore
                [
                    [
                        -Y,
                        np.zeros((self.nx, self.nu)),
                        self.beta * C2_tilde.T,
                        A_tilde.T,
                        C1.T,
                    ],
                    [
                        np.zeros((self.nu, self.nx)),
                        -self.ga**2 * np.eye(self.nu),
                        self.beta * D21_tilde.T,
                        B1_tilde.T,
                        D11.T,
                    ],
                    [
                        self.beta * C2_tilde,
                        self.beta * D21_tilde,
                        -2 * T,
                        B2_tilde.T,
                        D12.T,
                    ],
                    [A_tilde, B1_tilde, B2_tilde, -Y, np.zeros((self.nx, self.ny))],
                    [C1, D11, D12, np.zeros((self.ny, self.nx)), -np.eye(self.ny)],
                ]
            )

        # setup optimization problem, objective might change,
        # any feasible solution works as initialization for the parameters
        nM = M.shape[0]
        tol = 1e-4
        # rand_matrix = np.random.normal(0,1/np.sqrt(self.nx), (self.nx,self.nw))
        # objective = cp.Minimize(cp.norm(Y @ rand_matrix - B2_tilde))
        objective = cp.Minimize(None)  # type: ignore
        problem = cp.Problem(objective, [M << -tol * np.eye(nM)])  # type: ignore

        logger.info(
            'Initialize Parameter by values that satisfy LMI constraints, solve SDP ...'
        )
        problem.solve(solver=cp.SCS)  # type: ignore
        # check if t is negative
        # max_eig_lmi = np.max(np.real(np.linalg.eig(M.value)[0]))

        if problem.status == 'optimal':
            logger.info(
                f'Found negative semidefinite LMI, problem status: '
                f'\t {problem.status}'
            )
        else:
            raise Exception(
                "Neural network could not be initialized "
                "since no solution to the SDP problem was found."
            )

        logger.info('Write back Parameters values ...')
        dtype = torch.get_default_dtype()

        self.Y.data = torch.tensor(Y.value, dtype=dtype)
        self.A_tilde.weight.data = torch.tensor(A_tilde.value, dtype=dtype)

        self.B2_tilde.weight.data = torch.tensor(B2_tilde.value, dtype=dtype)
        if not self.ga == 0:
            self.C1.weight.data = torch.tensor(C1.value, dtype=dtype)
            self.D11.weight.data = torch.tensor(D11.value, dtype=dtype)
            self.D12.weight.data = torch.tensor(D12.value, dtype=dtype)
            self.B1_tilde.weight.data = torch.tensor(B1_tilde.value, dtype=dtype)
            self.D21_tilde.weight.data = torch.tensor(D21_tilde.value, dtype=dtype)
        self.C2_tilde.weight.data = torch.tensor(C2_tilde.value, dtype=dtype)
        self.lambdas.data = torch.tensor(lambdas.value, dtype=dtype)

    def forward(
        self, u_tilde: torch.Tensor, hx: Tuple[torch.Tensor, torch.Tensor]
    ) -> torch.Tensor:
        n_batch, n_sample, _ = u_tilde.shape

        Y_inv = self.Y.inverse()
        T_inv = torch.diag(1 / torch.squeeze(self.lambdas))
        # initialize output
        y = torch.zeros((n_batch, n_sample, self.ny))

        x = hx[0][1]
        for k in range(n_sample):
            z = (self.C2_tilde(x) + self.D21_tilde(u_tilde[:, k, :])) @ T_inv
            w = self.nl(z)
            y[:, k, :] = self.C1(x) + self.D11(u_tilde[:, k, :]) + self.D12(w)
            x = (
                self.A_tilde(x) + self.B1_tilde(u_tilde[:, k, :]) + self.B2_tilde(w)
            ) @ Y_inv

        return y

    def get_constraints(self) -> torch.Tensor:
        # state sizes
        nx = self.nx
        nu = self.nu
        ny = self.ny

        beta = self.beta
        # storage function
        Y = self.Y
        device = Y.device

        # state
        A_tilde = self.A_tilde.weight
        B1_tilde = self.B1_tilde.weight
        B2_tilde = self.B2_tilde.weight
        # output
        C1 = self.C1.weight
        D11 = self.D11.weight
        D12 = self.D12.weight
        # disturbance
        D21_tilde = self.D21_tilde.weight
        C2_tilde = self.C2_tilde.weight

        T = torch.diag(torch.squeeze(self.lambdas))
        ga = self.ga

        # M << 0
        if self.ga == 0:
            M = torch.cat(
                [
                    torch.cat(
                        [-Y, beta * C2_tilde.T, A_tilde.T],
                        dim=1,
                    ),
                    torch.cat(
                        [beta * C2_tilde, -2 * T, B2_tilde.T],
                        dim=1,
                    ),
                    torch.cat(
                        [A_tilde, B2_tilde, -Y],
                        dim=1,
                    ),
                ]
            )
        else:

            M = torch.cat(
                [
                    torch.cat(
                        (
                            -Y,
                            torch.zeros((nx, nu), device=device),
                            beta * C2_tilde.T,
                            A_tilde.T,
                            C1.T,
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            torch.zeros((nu, nx), device=device),
                            -(ga**2) * torch.eye(nu, device=device),
                            beta * D21_tilde.T,
                            B1_tilde.T,
                            D11.T,
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (beta * C2_tilde, beta * D21_tilde, -2 * T, B2_tilde.T, D12.T),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            A_tilde,
                            B1_tilde,
                            B2_tilde,
                            -Y,
                            torch.zeros((nx, ny), device=device),
                        ),
                        dim=1,
                    ),
                    torch.cat(
                        (
                            C1,
                            D11,
                            D12,
                            torch.zeros((ny, nx), device=device),
                            -torch.eye(ny, device=device),
                        ),
                        dim=1,
                    ),
                ]
            )

        return 0.5 * (M + M.T)

    def get_barrier(self, t: float) -> torch.Tensor:
        M = self.get_constraints()
        barrier = -t * (-M).logdet()

        _, info = torch.linalg.cholesky_ex(-M.cpu())

        if info > 0:
            barrier += torch.tensor(float('inf'))

        return barrier

    def check_constr(self) -> bool:
        with torch.no_grad():
            M = self.get_constraints()

            _, info = torch.linalg.cholesky_ex(-M.cpu())

            if info > 0:
                b_satisfied = False
            else:
                b_satisfied = True

        return b_satisfied


class InitLSTM(nn.Module):
    def __init__(
        self,
        input_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        output_dim: int,
        dropout: float,
    ):
        super().__init__()

        self.num_recurrent_layers = num_recurrent_layers
        self.recurrent_dim = recurrent_dim

        self.init_lstm = nn.LSTM(  # type: ignore
            input_size=input_dim + output_dim,
            hidden_size=recurrent_dim,
            num_layers=num_recurrent_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.predictor_lstm = nn.LSTM(  # type: ignore
            input_size=input_dim,
            hidden_size=recurrent_dim,
            num_layers=num_recurrent_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.output_layer = torch.nn.Linear(
            in_features=recurrent_dim, out_features=output_dim, bias=False
        )
        self.init_layer = torch.nn.Linear(
            in_features=recurrent_dim, out_features=output_dim, bias=False
        )

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        for name, param in self.init_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(
        self, input: torch.Tensor, x0: torch.Tensor, return_init: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        h_init, (h0_init, c0_init) = self.init_lstm(x0)
        h, (_, _) = self.predictor_lstm(input, (h0_init, c0_init))
        if return_init:
            return self.output_layer(h), self.init_layer(h_init)
        else:
            return self.output_layer(h)  # type: ignore
