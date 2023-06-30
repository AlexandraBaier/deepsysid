import abc
import logging
import warnings
from typing import Callable, List, Optional, Tuple, Union

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn

from . import utils

logger = logging.getLogger(__name__)


class HiddenStateForwardModule(nn.Module, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass


class ConstrainedForwardModule(HiddenStateForwardModule):
    @abc.abstractmethod
    def get_initial_parameters(
        self,
    ) -> Union[
        NDArray[np.float64],
        Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ]:
        pass

    @abc.abstractmethod
    def get_constraints(self) -> torch.Tensor:
        pass

    @abc.abstractmethod
    def check_constraints(self) -> bool:
        pass


class BasicLSTM(HiddenStateForwardModule):
    def __init__(
        self,
        input_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        output_dim: Union[List[int], int],
        dropout: float,
        bias: bool = True,
    ):
        super().__init__()

        self.num_recurrent_layers = num_recurrent_layers
        self.recurrent_dim = recurrent_dim

        with warnings.catch_warnings():
            self.predictor_lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=recurrent_dim,
                num_layers=num_recurrent_layers,
                dropout=dropout,
                batch_first=True,
                bias=bias,
            )

        if isinstance(output_dim, int):
            self.out = nn.ModuleList(
                [
                    nn.Linear(
                        in_features=recurrent_dim, out_features=output_dim, bias=bias
                    )
                ]
            )
        else:
            layer_dim = [recurrent_dim] + output_dim
            self.out = nn.ModuleList(
                [
                    nn.Linear(
                        in_features=layer_dim[i - 1],
                        out_features=layer_dim[i],
                        bias=bias,
                    )
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
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, (h0, c0) = self.predictor_lstm(x_pred, hx)
        for layer in self.out[:-1]:
            x = F.relu(layer(x))
        x = self.out[-1](x)

        return x, (h0, c0)


class BasicRnn(HiddenStateForwardModule):
    def __init__(
        self,
        input_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        output_dim: int,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()

        self.predictor_rnn = nn.RNN(
            input_size=input_dim,
            hidden_size=recurrent_dim,
            num_layers=num_recurrent_layers,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )

        self.out = nn.Linear(
            in_features=recurrent_dim, out_features=output_dim, bias=bias
        )

    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, _, _ = x_pred.shape

        if hx is not None:
            h = hx[0]
        else:
            h = None

        x, h = self.predictor_rnn.forward(x_pred, h)
        x = self.out.forward(x)

        return x, (h, h)


class BasicGRU(HiddenStateForwardModule):
    def __init__(
        self,
        input_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        output_dim: int,
        dropout: float,
        bias: bool,
    ) -> None:
        super().__init__()

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=recurrent_dim,
            num_layers=num_recurrent_layers,
            dropout=dropout,
            bias=bias,
            batch_first=True,
        )

        self.out = nn.Linear(
            in_features=recurrent_dim, out_features=output_dim, bias=bias
        )

    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if hx is not None:
            h = hx[0]
        else:
            h = None

        x, h = self.gru.forward(x_pred, h)
        x = self.out.forward(x)

        return x, (h, h)


class FixedDepthRnnFlexibleNonlinearity(HiddenStateForwardModule):
    def __init__(
        self,
        input_dim: int,
        recurrent_dim: int,
        output_dim: int,
        bias: bool,
        nonlinearity: str,
    ) -> None:
        super().__init__()

        self.recurrent_dim = recurrent_dim
        self.output_dim = output_dim

        # h = sigma(W_h * h^{k-1} + b_h + U_h * x^{k})
        self.W_h = torch.nn.Linear(
            in_features=recurrent_dim, out_features=recurrent_dim, bias=bias
        )
        self.U_h = torch.nn.Linear(
            in_features=input_dim, out_features=recurrent_dim, bias=False
        )
        try:
            self.nl = eval(nonlinearity)
        except SyntaxError:
            raise Exception('Nonlinearity could not be evaluated.')

        self.out = nn.Linear(
            in_features=recurrent_dim, out_features=output_dim, bias=bias
        )

    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, n_sample, _ = x_pred.shape

        # init output
        y = torch.zeros((n_batch, n_sample, self.output_dim))
        if hx is not None:
            x = hx[0][1]
        else:
            x = torch.zeros((n_batch, self.recurrent_dim))

        for k in range(n_sample):
            x = self.nl(self.W_h(x) + self.U_h(x_pred[:, k, :]))
            y[:, k, :] = self.out(x)

        return y, (x, x)


class LinearOutputLSTM(HiddenStateForwardModule):
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

        with warnings.catch_warnings():
            self.predictor_lstm = nn.LSTM(
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
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x, (h0, c0) = self.predictor_lstm.forward(x_pred, hx)
        x = self.out.forward(x)

        return x, (h0, c0)


class LtiRnn(HiddenStateForwardModule):
    def __init__(self, nx: int, nu: int, ny: int, nw: int, nonlinearity: str) -> None:
        super(LtiRnn, self).__init__()

        self.nx = nx  # number of states
        self.nu = nu  # number of performance (external) input
        self.nw = nw  # number of disturbance input
        self.ny = ny  # number of performance output
        self.nz = nw  # number of disturbance output, always equal to size of w

        try:
            self.nl = eval(nonlinearity)
        except SyntaxError:
            raise Exception('Nonlinearity could not be evaluated.')

        self.Y = torch.nn.Parameter(torch.eye(self.nx))

        self.A_tilde = torch.nn.Linear(self.nx, self.nx, bias=False)
        self.B1_tilde = torch.nn.Linear(self.nu, self.nx, bias=False)
        self.B2_tilde = torch.nn.Linear(self.nw, self.nx, bias=False)
        self.C1 = torch.nn.Linear(self.nx, self.ny, bias=False)
        self.D11 = torch.nn.Linear(self.nu, self.ny, bias=False)
        self.D12 = torch.nn.Linear(self.nw, self.ny, bias=False)
        self.C2_tilde = torch.nn.Linear(self.nx, self.nz, bias=False)
        self.D21_tilde = torch.nn.Linear(self.nu, self.nz, bias=False)

        self.lambdas = torch.nn.Parameter(torch.ones((self.nw, 1)))

    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, n_sample, _ = x_pred.shape

        Y_inv = self.Y.inverse()
        T_inv = torch.diag(1 / torch.squeeze(self.lambdas))

        # initialize output
        y = torch.zeros((n_batch, n_sample, self.ny))
        if hx is not None:
            x = hx[0][1]
        else:
            x = torch.zeros((n_batch, self.nx))

        for k in range(n_sample):
            z = (self.C2_tilde(x) + self.D21_tilde(x_pred[:, k, :])) @ T_inv
            w = self.nl(z)
            y[:, k, :] = self.C1(x) + self.D11(x_pred[:, k, :]) + self.D12(w)
            x = (
                self.A_tilde(x) + self.B1_tilde(x_pred[:, k, :]) + self.B2_tilde(w)
            ) @ Y_inv

        return y, (x, x)


class LtiRnnConvConstr(HiddenStateForwardModule):
    def __init__(
        self,
        nx: int,
        nu: int,
        ny: int,
        nw: int,
        gamma: float,
        beta: float,
        bias: bool,
        nonlinearity: nn.Module,
        device: torch.device = torch.device('cpu'),
    ) -> None:
        super(LtiRnnConvConstr, self).__init__()

        self.device = device

        # torch.set_default_dtype(torch.float64)

        self.nx = nx  # number of states
        self.nu = nu  # number of performance (external) input
        self.nw = nw  # number of disturbance input
        self.ny = ny  # number of performance output
        self.nz = nw  # number of disturbance output, always equal to size of w

        self.ga = gamma
        self.beta = beta
        self.nl = nonlinearity

        self.Y = torch.nn.Parameter(torch.zeros((self.nx, self.nx)))
        self.A_tilde = torch.nn.Linear(self.nx, self.nx, bias=False)
        self.B1_tilde = torch.nn.Linear(self.nu, self.nx, bias=False)
        self.B2_tilde = torch.nn.Linear(self.nw, self.nx, bias=False)
        self.C1 = torch.nn.Linear(self.nx, self.ny, bias=False)
        self.D11 = torch.nn.Linear(self.nu, self.ny, bias=False)
        self.D12 = torch.nn.Linear(self.nw, self.ny, bias=False)
        self.C2_tilde = torch.nn.Linear(self.nx, self.nz, bias=False)
        self.D21_tilde = torch.nn.Linear(self.nu, self.nz, bias=False)
        self.lambdas = torch.nn.Parameter(torch.zeros((self.nw, 1)))
        self.b_z = torch.nn.Parameter(torch.zeros((self.nz)), requires_grad=bias)
        self.b_y = torch.nn.Parameter(torch.zeros((self.ny)), requires_grad=bias)
        self.b_x = torch.nn.Parameter(torch.zeros((self.nx)), requires_grad=bias)

        # self.h0 = None

    def initialize_lmi(self) -> None:
        # np.random.seed = 2023
        # storage function
        Y = cp.Variable((self.nx, self.nx), 'Y')
        # hidden state
        A_tilde = cp.Variable((self.nx, self.nx), 'A_tilde')
        B1_tilde = cp.Variable((self.nx, self.nu), 'B1_tilde')
        B2_tilde = cp.Variable((self.nx, self.nw), 'B2_tilde')
        # output
        C1 = cp.Variable((self.ny, self.nx), 'C1')
        D11 = cp.Variable((self.ny, self.nu), 'D11')
        D12 = cp.Variable((self.ny, self.nw), 'D12')
        # disturbance
        C2 = np.random.normal(0, 1 / np.sqrt(self.nw), size=(self.nz, self.nx))
        D21 = np.random.normal(0, 1 / np.sqrt(self.nw), size=(self.nz, self.nu))
        # multipliers
        lambdas = cp.Variable((self.nw, 1), 'tau', nonneg=True)
        T = cp.diag(lambdas)

        C2_tilde = T @ C2
        D21_tilde = T @ D21

        if self.ga == 0:
            # lmi that ensures finite l2 gain
            M = cp.bmat(
                [
                    [-Y, self.beta * C2_tilde.T, A_tilde.T],
                    [self.beta * C2_tilde, -2 * T, B2_tilde.T],
                    [A_tilde, B2_tilde, -Y],
                ]
            )
        else:
            # lmi that ensures l2 gain gamma
            M = cp.bmat(
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

        rand_matrix = np.random.normal(0, 1 / np.sqrt(self.nx), (self.nx, self.nw))
        objective = cp.Minimize(cp.norm(Y @ rand_matrix - B2_tilde))
        # nu = cp.Variable((1, nM))
        # objective = cp.Minimize(nu @ np.ones((nM, 1)))
        # objective = cp.Minimize(None)
        problem = cp.Problem(objective, [M << -tol * np.eye(nM)])

        logger.info(
            'Initialize Parameter by values that satisfy LMI constraints, solve SDP ...'
        )
        problem.solve(solver=cp.SCS)
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
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, n_sample, _ = x_pred.shape

        Y_inv = self.Y.inverse()
        T_inv = torch.diag(1 / torch.squeeze(self.lambdas))
        # initialize output
        y = torch.zeros((n_batch, n_sample, self.ny))

        if hx is not None:
            x = hx[0][
                -1
            ]  # take the hidden state of the last layer from the initializer
        else:
            x = torch.zeros((n_batch, self.nx))

        for k in range(n_sample):
            z = (self.C2_tilde(x) + self.D21_tilde(x_pred[:, k, :]) + self.b_z) @ T_inv
            w = self.nl(z)
            y[:, k, :] = self.C1(x) + self.D11(x_pred[:, k, :]) + self.D12(w) + self.b_y
            x = (
                self.A_tilde(x)
                + self.B1_tilde(x_pred[:, k, :])
                + self.B2_tilde(w)
                + self.b_x
            ) @ Y_inv

        return y, (x, x)

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

        return (0.5 * (M + M.T)).to(self.device)

    def get_logdet(self, mat: torch.Tensor) -> torch.Tensor:
        # return logdet of matrix mat, if it is not positive semi-definite, return inf
        _, info = torch.linalg.cholesky_ex(mat.cpu())

        if info > 0:
            logdet = torch.tensor(float('inf')).to(self.device)
        else:
            logdet = (mat.logdet()).to(self.device)

        return logdet

    def get_barriers(self, t: float) -> torch.Tensor:
        constraints = [
            -self.get_constraints(),
            self.Y,
            torch.diag(torch.squeeze(self.lambdas)).to(self.device),
        ]
        barrier = torch.tensor(0.0).to(self.device)
        for constraint in constraints:
            barrier += -t * self.get_logdet(constraint)

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

    def get_min_max_real_eigenvalues(self) -> Tuple[np.float64, np.float64]:
        M = self.get_constraints()
        return (
            torch.min(torch.real(torch.linalg.eig(M)[0])).cpu().detach().numpy(),
            torch.max(torch.real(torch.linalg.eig(M)[0])).cpu().detach().numpy(),
        )

    def write_flat_parameters(self, flat_param: torch.Tensor) -> None:
        idx = 0
        for p in self.parameters():
            p.data = flat_param[idx : idx + p.numel()].view_as(p.data)
            idx = p.numel()

    def write_parameters(self, params: List[torch.Tensor]) -> None:
        for old_par, new_par in zip(params, self.parameters()):
            new_par.data = old_par.clone()

    def get_linear_combination(
        self, old_pars: List[torch.Tensor], new_pars: List[torch.Tensor]
    ) -> Tuple[List[float], List[float]]:
        alphas = np.linspace(0, 1, 100)
        barriers: List[float] = []
        for alpha in alphas:
            par = [
                (1 - alpha) * old_par + alpha * new_par
                for old_par, new_par in zip(old_pars, new_pars)
            ]
            self.write_parameters(par)
            barriers.append(float(self.get_barriers(1.0).cpu().detach().numpy()))
        return (barriers, list(alphas))


class Linear(nn.Module):
    def __init__(
        self, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, D: torch.Tensor
    ) -> None:
        super().__init__()
        self._nx = A.shape[0]
        self._nu = B.shape[1]
        self._ny = C.shape[0]

        self.A = A
        self.B = B
        self.C = C
        self.D = D

        # initialize
        # self._init_weights()

    def _init_weights(self) -> None:
        for p in self.parameters():
            torch.nn.init.uniform_(
                tensor=p, a=-np.sqrt(1 / self._nu), b=np.sqrt(1 / self._nu)
            )

    def state_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.A @ x + self.B @ u

    def output_dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return self.C @ x + self.D @ u

    def forward(
        self, x0: torch.Tensor, us: torch.Tensor, return_state: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        device = self.A.device
        n_batch, N, _, _ = us.shape
        x = torch.zeros(size=(n_batch, N + 1, self._nx, 1)).to(device)
        y = torch.zeros(size=(n_batch, N, self._ny, 1)).to(device)
        x[:, 0, :, :] = x0

        for k in range(N):
            x[:, k + 1, :, :] = self.state_dynamics(x=x[:, k, :, :], u=us[:, k, :, :])
            y[:, k, :, :] = self.output_dynamics(x=x[:, k, :, :], u=us[:, k, :, :])
        if return_state:
            return (y, x)
        else:
            return y


class LureSystem(Linear):
    def __init__(
        self,
        A: torch.Tensor,
        B1: torch.Tensor,
        B2: torch.Tensor,
        C1: torch.Tensor,
        D11: torch.Tensor,
        D12: torch.Tensor,
        C2: torch.Tensor,
        D21: torch.Tensor,
        Delta: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device,
    ) -> None:
        super().__init__(A=A, B=B1, C=C1, D=D11)
        self._nw = B2.shape[1]
        self._nz = C2.shape[0]
        assert self._nw == self._nz
        self.B2 = B2
        self.C2 = C2
        self.D12 = D12
        self.D21 = D21
        self.Delta = Delta  # static nonlinearity
        self.device = device

    def forward(
        self, x0: torch.Tensor, us: torch.Tensor, return_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, N, _, _ = us.shape
        x = torch.zeros(size=(n_batch, N + 1, self._nx, 1)).to(self.device)
        y = torch.zeros(size=(n_batch, N, self._ny, 1)).to(self.device)
        w = torch.zeros(size=(n_batch, N, self._nw, 1)).to(self.device)
        x[:, 0, :, :] = x0

        for k in range(N):
            w[:, k, :, :] = self.Delta(
                self.C2 @ x[:, k, :, :] + self.D21 @ us[:, k, :, :]
            )
            x[:, k + 1, :, :] = (
                super().state_dynamics(x=x[:, k, :, :], u=us[:, k, :, :])
                + self.B2 @ w[:, k, :, :]
            )
            y[:, k, :, :] = (
                super().output_dynamics(x=x[:, k, :, :], u=us[:, k, :, :])
                + self.D12 @ w[:, k, :, :]
            )
        if return_states:
            return (y, x)
        else:
            return y

    def _detach_matrices(self) -> None:
        matrices = [
            self.A,
            self.B,
            self.B2,
            self.C,
            self.D,
            self.D12,
            self.C2,
            self.D21,
        ]
        for matrix in matrices:
            matrix = matrix.detach()


class HybridLinearizationRnn(ConstrainedForwardModule):
    def __init__(
        self,
        A_lin: NDArray[np.float64],
        B_lin: NDArray[np.float64],
        C_lin: NDArray[np.float64],
        alpha: float,
        beta: float,
        nwu: int,
        nzu: int,
        gamma: float,
        nonlinearity: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device = torch.device('cpu'),
    ) -> None:
        super().__init__()
        self.nx = A_lin.shape[0]  # state size
        self.nu = self.nx  # input size of linearization
        self.ny = self.nx  # output size of linearization
        self.nwp = B_lin.shape[1]  # input size of performance channel
        self.nzp = C_lin.shape[0]  # output size of performance channel
        assert nzu == nwu
        self.nzu = nzu  # output size of uncertainty channel
        self.nwu = nwu  # input size of uncertainty channel

        self._x_mean: Optional[torch.Tensor] = None
        self._x_std: Optional[torch.Tensor] = None
        self._wp_mean: Optional[torch.Tensor] = None
        self._wp_std: Optional[torch.Tensor] = None

        self.alpha = alpha
        self.beta = beta

        self.device = device

        self.A_lin = torch.tensor(A_lin, dtype=torch.float32).to(device)
        self.B_lin = torch.tensor(B_lin, dtype=torch.float32).to(device)
        self.C_lin = torch.tensor(C_lin, dtype=torch.float32).to(device)

        self.nl = nonlinearity

        self.gamma = gamma
        epsilon = 1e-3

        L_flat_size = self.extract_vector_from_lower_triangular_matrix(
            np.zeros(shape=(self.nx, self.nx))
        ).shape[0]
        self.L_x_flat = torch.nn.Parameter(
            epsilon * torch.ones(size=(L_flat_size,)).float().to(device)
        )
        self.L_y_flat = torch.nn.Parameter(
            -epsilon * torch.ones(size=(L_flat_size,)).float().to(device)
        )

        self.lam = torch.nn.Parameter(
            epsilon * torch.ones(size=(self.nzu,)).float().to(device)
        )

        self.K = torch.nn.Parameter(epsilon * torch.eye(self.nx).float().to(device))
        self.L1 = torch.nn.Parameter(
            torch.zeros(size=(self.nx, self.nwp)).float().to(device)
        )
        self.L2 = torch.nn.Parameter(
            torch.zeros(size=(self.nx, self.ny)).float().to(device)
        )
        self.L3 = torch.nn.Parameter(
            torch.zeros(size=(self.nx, self.nwu)).float().to(device)
        )

        self.M1 = torch.nn.Parameter(
            torch.zeros(size=(self.nu, self.nx)).float().to(device)
        )
        self.N11 = torch.nn.Parameter(
            torch.zeros(size=(self.nu, self.nwp)).float().to(device)
        )
        self.N12 = torch.nn.Parameter(
            torch.zeros(size=(self.nu, self.ny)).float().to(device)
        )
        self.N13 = torch.nn.Parameter(
            torch.zeros(size=(self.nu, self.nwu)).float().to(device)
        )

        self.M2 = torch.nn.Parameter(
            torch.zeros(size=(self.nzu, self.nx)).float().to(device)
        )
        self.N21 = torch.nn.Parameter(
            torch.zeros(size=(self.nzu, self.nwp)).float().to(device)
        )
        self.N22 = torch.nn.Parameter(
            torch.zeros(size=(self.nzu, self.ny)).float().to(device)
        )
        # self.N23 = torch.nn.Parameter(
        #     torch.zeros(size=(self.nzu, self.nwu)).float().to(device)
        # )

        B_lin_hat = np.hstack(
            (B_lin, np.zeros(shape=(B_lin.shape[0], self.nwp + self.nx)))
        )
        self.nwp_hat = self.nwp + self.nwp + self.nx

        self.S_s = torch.from_numpy(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            A_lin,
                            np.zeros(shape=(self.nx, self.nx)),
                            B_lin_hat,
                            np.zeros(shape=(self.nx, self.nwu)),
                        ],
                        axis=1,
                    ),
                    np.zeros(
                        shape=(self.nx, self.nx + self.nx + self.nwp_hat + self.nwu)
                    ),
                    np.concatenate(
                        [
                            C_lin,
                            np.zeros(shape=(self.nzp, self.nx)),
                            np.zeros(shape=(self.nzp, self.nwp_hat)),
                            np.zeros(shape=(self.nzp, self.nwu)),
                        ],
                        axis=1,
                    ),
                    np.zeros(
                        shape=(self.nzu, self.nx + self.nx + self.nwp_hat + self.nwu)
                    ),
                ],
                axis=0,
                dtype=np.float32,
            )
        ).to(device)
        self.S_l = torch.from_numpy(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nx, self.nx)),
                            A_lin,
                            np.zeros(shape=(self.nx, self.nzu)),
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            np.eye(self.nx),
                            np.zeros(shape=(self.nx, self.nu)),
                            np.zeros(shape=(self.nx, self.nzu)),
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nzp, self.nx)),
                            C_lin,
                            np.zeros(shape=(self.nzp, self.nzu)),
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nzu, self.nx)),
                            np.zeros(shape=(self.nzu, self.nu)),
                            np.eye(self.nzu),
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
                dtype=np.float32,
            )
        ).to(device)
        self.S_r = torch.from_numpy(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nx, self.nx)),
                            np.eye(self.nx),
                            np.zeros(shape=(self.nx, self.nwp_hat)),
                            np.zeros(shape=(self.nx, self.nwu)),
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nwp_hat, self.nx)),
                            np.zeros(shape=(self.nwp_hat, self.nx)),
                            np.eye(self.nwp_hat),
                            np.zeros(shape=(self.nwp_hat, self.nwu)),
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            np.eye(self.ny),
                            np.zeros(shape=(self.ny, self.nx)),
                            np.zeros(shape=(self.ny, self.nwp_hat)),
                            np.zeros(shape=(self.ny, self.nwu)),
                        ],
                        axis=1,
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nwu, self.nx)),
                            np.zeros(shape=(self.nwu, self.nx)),
                            np.zeros(shape=(self.nwu, self.nwp_hat)),
                            np.eye(self.nwu),
                        ],
                        axis=1,
                    ),
                ],
                axis=0,
                dtype=np.float32,
            )
        ).to(device)

        # self.set_lure_system()

    def initialize_lmi(self) -> None:
        # 1. solve synthesis inequalities to find feasible parameter set
        omega_tilde_0_numpy, X, Y, Lambda = self.get_initial_parameters()

        (K, L1, L2, L3, M1, N11, N12, N13, M2, N21, N22, N23) = self.get_matrices(
            omega_tilde_0_numpy
        )

        L_x = np.linalg.cholesky(a=X)
        L_x_flat = self.extract_vector_from_lower_triangular_matrix(L_x)
        self.L_x_flat.data = torch.tensor(L_x_flat).float()

        L_y = np.linalg.cholesky(a=Y)
        L_y_flat = self.extract_vector_from_lower_triangular_matrix(L_y)
        self.L_y_flat.data = torch.tensor(L_y_flat).float()

        self.lam.data = torch.tensor(np.diag(Lambda)).float()

        self.K.data = torch.tensor(K).float()
        self.L1.data = torch.tensor(L1).float()
        self.L2.data = torch.tensor(L2).float()
        self.L3.data = torch.tensor(L3).float()

        self.M1.data = torch.tensor(M1).float()
        self.N11.data = torch.tensor(N11).float()
        self.N12.data = torch.tensor(N12).float()
        self.N13.data = torch.tensor(N13).float()

        self.M2.data = torch.tensor(M2).float()
        self.N21.data = torch.tensor(N21).float()
        self.N22.data = torch.tensor(N22).float()
        # self.N23.data = torch.tensor(N23).float()

    def construct_lower_triangular_matrix(
        self, L_flat: torch.Tensor, diag_length: int
    ) -> torch.Tensor:
        device = L_flat.device
        flat_idx = 0
        L = torch.zeros(
            size=(diag_length, diag_length), dtype=torch.float32, device=device
        )
        for diag_idx, diag_size in zip(
            range(0, -diag_length, -1), range(diag_length, 0, -1)
        ):
            L += torch.diag(L_flat[flat_idx : flat_idx + diag_size], diagonal=diag_idx)
            flat_idx += diag_size

        return L

    def extract_vector_from_lower_triangular_matrix(
        self, L: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        diag_length = L.shape[0]
        vector_list = []
        for diag_idx in range(0, -diag_length, -1):
            vector_list.append(np.diag(L, k=diag_idx))

        return np.hstack(vector_list)

    def get_T(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        U: torch.Tensor,
        V: torch.Tensor,
        Lambda: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        T_l = torch.concat(
            [
                torch.concat(
                    [
                        U,
                        X @ self.A_lin,
                        torch.zeros((self.nx, self.nwu)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros((self.nu, self.nx)),
                        torch.eye(self.nu),
                        torch.zeros((self.nu, self.nwu)),
                    ],
                    dim=1,
                ).to(self.device),
                torch.concat(
                    [
                        torch.zeros((self.nzu, self.nx)).to(self.device),
                        torch.zeros((self.nzu, self.nu)).to(self.device),
                        Lambda,
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        ).float()
        T_r = torch.concat(
            [
                torch.concat(
                    [
                        torch.zeros((self.nx, self.nx)).to(self.device),
                        torch.zeros(size=(self.nx, self.nwp)).to(self.device),
                        V.T,
                        torch.zeros((self.nx, self.nwu)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros(size=(self.nwp, self.nx)),
                        torch.eye(self.nwp),
                        torch.zeros(size=(self.nwp, self.nx)),
                        torch.zeros(size=(self.nwp, self.nwu)),
                    ],
                    dim=1,
                ).to(self.device),
                torch.concat(
                    [
                        torch.eye(self.nx).to(self.device),
                        torch.zeros(size=(self.nx, self.nwp)).to(self.device),
                        Y,
                        torch.zeros((self.nu, self.nwu)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros((self.nzu, self.nx)),
                        torch.zeros(size=(self.nzu, self.nwp)),
                        torch.zeros((self.nzu, self.ny)),
                        torch.eye(self.nwu),
                    ],
                    dim=1,
                ).to(self.device),
            ],
            dim=0,
        ).float()
        T_s = torch.concat(
            [
                torch.concat(
                    [
                        torch.zeros(size=(self.nx, self.nx + self.nwp)).to(self.device),
                        X @ self.A_lin @ Y,
                        torch.zeros(size=(self.nx, self.nwu)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.zeros(size=(self.nu, self.nx + self.nwp + self.ny + self.nwu)).to(
                    self.device
                ),
                torch.zeros(
                    size=(self.nzu, self.nx + self.nwp + self.ny + self.nwu)
                ).to(self.device),
            ],
            dim=0,
        ).float()

        return (T_l, T_r, T_s)

    def get_coupling_matrices(
        self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        L_x = self.construct_lower_triangular_matrix(
            L_flat=self.L_x_flat, diag_length=self.nx
        ).to(self.device)
        L_y = self.construct_lower_triangular_matrix(
            L_flat=self.L_y_flat, diag_length=self.nx
        ).to(self.device)

        X = L_x @ L_x.T
        Y = L_y @ L_y.T

        # 2. Determine non-singular U,V with V U^T = I - Y X
        # U = torch.linalg.inv(Y) - X
        # V = Y
        U = torch.eye(self.nx).to(self.device)
        V = torch.eye(self.nx).to(self.device) - Y @ X

        return (X, Y, U, V)

    def set_lure_system(self) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

        if (
            self._x_mean is None
            or self._x_std is None
            or self._wp_mean is None
            or self._wp_std is None
        ):
            raise ValueError('Mean and std values of linear models are not set.')

        device = self.device
        omega_tilde_0 = self.get_omega_tilde().to(device)

        # construct X, Y, and Lambda
        X, Y, U, V = self.get_coupling_matrices()

        Lambda = torch.diag(input=self.lam).to(device)

        # transform to original parameters
        T_l, T_r, T_s = self.get_T(X, Y, U, V, Lambda)
        omega_0 = (
            torch.linalg.inv(T_l).float().to(device)
            @ (omega_tilde_0 - T_s.float().to(device))
            @ torch.linalg.inv(T_r).float().to(device)
        )

        (
            A_tilde,
            B1_tilde,
            B2_tilde,
            B3_tilde,
            C1_tilde,
            D11_tilde,
            D12_tilde,
            D13_tilde,
            C2,
            D21,
            D22,
        ) = self.get_controller_matrices(omega_0)

        B1_hat = B1_tilde @ torch.diag(1 / self._wp_std)
        B2_hat = B2_tilde @ torch.diag(1 / self._x_std)
        C1_hat = torch.diag(self._x_std) @ C1_tilde
        D11_hat = torch.diag(self._x_std) @ D11_tilde @ torch.diag(1 / self._wp_std)
        D12_hat = torch.diag(self._x_std) @ D12_tilde @ torch.diag(1 / self._x_std)
        D13_hat = torch.diag(self._x_std) @ D13_tilde
        D21_hat = D21 @ torch.diag(1 / self._wp_std)
        D22_hat = D22 @ torch.diag(1 / self._x_std)

        omega = self.get_omega(
            A_tilde,
            torch.hstack((B1_hat, -B1_hat, -B2_hat)),
            B2_hat,
            B3_tilde,
            C1_hat,
            torch.hstack(
                (D11_hat, -D11_hat, torch.eye(self.nx).to(self.device) - D12_hat)
            ),
            D12_hat,
            D13_hat,
            C2,
            torch.hstack((D21_hat, -D21_hat, -D22_hat)),
            D22_hat,
        )

        sys_block_matrix = self.S_s + self.S_l @ omega @ self.S_r

        (
            A_cal,
            B1_cal,
            B2_cal,
            C1_cal,
            D11_cal,
            D12_cal,
            C2_cal,
            D21_cal,
            D22_cal,
        ) = self.get_interconnected_matrices(sys_block_matrix)
        # print(A_cal.requires_grad)
        # print(f'norm D_22 cal {torch.linalg.norm(D22_cal)}')
        # assert torch.linalg.norm(D22_cal) - 0 < 1e-2
        D22_cal = torch.zeros_like(D22_cal)

        def Delta_tilde(z: torch.Tensor) -> torch.Tensor:
            return torch.tensor((2 / (self.beta - self.alpha))) * (
                self.nl(z) - torch.tensor(((self.alpha + self.beta) / 2)) * z
            )

        self.lure = LureSystem(
            A=A_cal,
            B1=B1_cal,
            B2=B2_cal,
            C1=C1_cal,
            D11=D11_cal,
            D12=D12_cal,
            C2=C2_cal,
            D21=D21_cal,
            Delta=Delta_tilde,
            device=self.device,
        ).to(device)

        return (omega_0.cpu().detach().numpy(), sys_block_matrix.cpu().detach().numpy())

    def get_controller_matrices(
        self, omega: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        A_tilde = omega[: self.nx, : self.nx]
        B1_tilde = omega[: self.nx, self.nx : self.nx + self.nwp]
        B2_tilde = omega[: self.nx, self.nx + self.nwp : self.nx + self.nwp + self.ny]
        B3_tilde = omega[: self.nx, self.nx + self.nwp + self.ny :]

        C1_tilde = omega[self.nx : self.nx + self.nu, : self.nx]
        D11_tilde = omega[self.nx : self.nx + self.nu, self.nx : self.nx + self.nwp]
        D12_tilde = omega[
            self.nx : self.nx + self.nu,
            self.nx + self.nwp : self.nx + self.nwp + self.ny,
        ]
        D13_tilde = omega[self.nx : self.nx + self.nu, self.nx + self.nwp + self.ny :]

        C2 = omega[self.nx + self.nu :, : self.nx]
        D21 = omega[self.nx + self.nu :, self.nx : self.nx + self.nwp]
        D22 = omega[
            self.nx + self.nu :, self.nx + self.nwp : self.nx + self.nwp + self.ny
        ]

        return (
            A_tilde,
            B1_tilde,
            B2_tilde,
            B3_tilde,
            C1_tilde,
            D11_tilde,
            D12_tilde,
            D13_tilde,
            C2,
            D21,
            D22,
        )

    def get_omega_tilde(self) -> torch.Tensor:

        return torch.concat(
            [
                torch.concat([self.K, self.L1, self.L2, self.L3], dim=1),
                torch.concat([self.M1, self.N11, self.N12, self.N13], dim=1),
                torch.concat(
                    [
                        self.M2,
                        self.N21,
                        self.N22,
                        torch.zeros(size=(self.nwu, self.nzu), device=self.device),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )

    def get_omega(
        self,
        A_tilde: torch.Tensor,
        B1_tilde: torch.Tensor,
        B2_tilde: torch.Tensor,
        B3_tilde: torch.Tensor,
        C1_tilde: torch.Tensor,
        D11_tilde: torch.Tensor,
        D12_tilde: torch.Tensor,
        D13_tilde: torch.Tensor,
        C2: torch.Tensor,
        D21: torch.Tensor,
        D22: torch.Tensor,
    ) -> torch.Tensor:
        J = torch.tensor(2 / (self.alpha - self.beta), device=self.device)
        L = torch.tensor((self.alpha + self.beta) / 2, device=self.device)
        B3 = J * B3_tilde
        A = A_tilde - L * B3 @ C2
        B1 = B1_tilde - L * B3 @ D21
        B2 = B2_tilde - L * B3 @ D22

        D13 = J * D13_tilde
        C1 = C1_tilde - L * D13 @ C2
        D11 = D11_tilde - L * D13 @ D21
        D12 = D12_tilde - L * D13 @ D22

        return torch.concat(
            [
                torch.concat([A, B1, B2, B3], dim=1),
                torch.concat([C1, D11, D12, D13], dim=1),
                torch.concat(
                    [
                        C2,
                        D21,
                        D22,
                        torch.zeros(size=(self.nwu, self.nzu), device=self.device),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        )

    def get_matrices(
        self, block_matrix: NDArray[np.float64]
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        K = block_matrix[: self.nx, : self.nx]
        L1 = block_matrix[: self.nx, self.nx : self.nx + self.nwp]
        L2 = block_matrix[: self.nx, self.nx + self.nwp : self.nx + self.nwp + self.ny]
        L3 = block_matrix[: self.nx, self.nx + self.nwp + self.ny :]

        M1 = block_matrix[self.nx : self.nx + self.nu, : self.nx]
        N11 = block_matrix[self.nx : self.nx + self.nu, self.nx : self.nx + self.nwp]
        N12 = block_matrix[
            self.nx : self.nx + self.nu,
            self.nx + self.nwp : self.nx + self.nwp + self.ny,
        ]
        N13 = block_matrix[self.nx : self.nx + self.nu, self.nx + self.nwp + self.ny :]

        M2 = block_matrix[self.nx + self.nu :, : self.nx]
        N21 = block_matrix[self.nx + self.nu :, self.nx : self.nx + self.nwp]
        N22 = block_matrix[
            self.nx + self.nu :, self.nx + self.nwp : self.nx + self.nwp + self.ny
        ]
        N23 = block_matrix[self.nx + self.nu :, self.nx + self.nwp + self.ny :]
        # assert np.linalg.norm(D23) < 1e-4

        return (K, L1, L2, L3, M1, N11, N12, N13, M2, N21, N22, N23)

    def get_interconnected_matrices(
        self, interconnected_block_matrix: torch.Tensor
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        A_cal = interconnected_block_matrix[: self.nx + self.nx, : self.nx + self.nx]
        B1_cal = interconnected_block_matrix[
            : self.nx + self.nx, self.nx + self.nx : self.nx + self.nx + self.nwp_hat
        ]
        B2_cal = interconnected_block_matrix[
            : self.nx + self.nx,
            self.nx
            + self.nx
            + self.nwp_hat : self.nx
            + self.nx
            + self.nwp_hat
            + self.nwu,
        ]

        C1_cal = interconnected_block_matrix[
            self.nx + self.nx : self.nx + self.nx + self.nzp, : self.nx + self.nx
        ]
        D11_cal = interconnected_block_matrix[
            self.nx + self.nx : self.nx + self.nx + self.nzp,
            self.nx + self.nx : self.nx + self.nx + self.nwp_hat,
        ]
        D12_cal = interconnected_block_matrix[
            self.nx + self.nx : self.nx + self.nx + self.nzp,
            self.nx
            + self.nx
            + self.nwp_hat : self.nx
            + self.nx
            + self.nwp_hat
            + self.nwu,
        ]

        C2_cal = interconnected_block_matrix[
            self.nx + self.nx + self.nzp : self.nx + self.nx + self.nzp + self.nzu,
            : self.nx + self.nx,
        ]
        D21_cal = interconnected_block_matrix[
            self.nx + self.nx + self.nzp : self.nx + self.nx + self.nzp + self.nzu,
            self.nx + self.nx : self.nx + self.nx + self.nwp_hat,
        ]
        D22_cal = interconnected_block_matrix[
            self.nx + self.nx + self.nzp : self.nx + self.nx + self.nzp + self.nzu,
            self.nx
            + self.nx
            + self.nwp_hat : self.nx
            + self.nx
            + self.nwp_hat
            + self.nwu,
        ]

        return (
            A_cal,
            B1_cal,
            B2_cal,
            C1_cal,
            D11_cal,
            D12_cal,
            C2_cal,
            D21_cal,
            D22_cal,
        )

    def get_initial_parameters(
        self,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        X = cp.Variable(shape=(self.nx, self.nx), PSD=True)
        Y = cp.Variable(shape=(self.nx, self.nx), PSD=True)
        lam = cp.Variable(shape=(self.nzu, 1), pos=True, name='lam')
        Lambda = cp.diag(lam)

        K = cp.Variable(shape=(self.nx, self.nx))
        # K = np.zeros(shape=(self.nx,self.nx))
        L1 = cp.Variable(shape=(self.nx, self.nwp))
        # L1 = np.zeros(shape=(self.nx, self.nwp))
        L2 = cp.Variable(shape=(self.nx, self.ny))
        # L2 = np.zeros(shape=(self.nx, self.ny))
        L3 = cp.Variable(shape=(self.nx, self.nwu))

        M1 = cp.Variable(shape=(self.nu, self.nx))
        # M1 = np.zeros(shape=(self.nu, self.nx))
        N11 = cp.Variable(shape=(self.nu, self.nwp))
        # N11 = np.zeros(shape=(self.nu, self.nwp))
        N12 = cp.Variable(shape=(self.nu, self.ny))
        # N12 = np.zeros(shape=(self.nu, self.ny))
        N13 = cp.Variable(shape=(self.nu, self.nwu))

        M2 = cp.Variable(shape=(self.nzu, self.nx))
        N21 = cp.Variable(shape=(self.nzu, self.nwp))
        N22 = cp.Variable(shape=(self.nzu, self.ny))
        N23 = np.zeros(shape=(self.nzu, self.nwu))
        # N23 = cp.Variable(shape=(self.nzu, self.nwu))

        P_21_1 = cp.bmat(
            [
                [
                    self.A_lin.cpu() @ Y,
                    self.A_lin.cpu(),
                    self.B_lin.cpu(),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nu, self.nx)),
                    X @ self.A_lin.cpu(),
                    X @ self.B_lin.cpu(),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    self.C_lin.cpu() @ Y,
                    self.C_lin.cpu(),
                    np.zeros(shape=(self.nzp, self.nwp)),
                    np.zeros(shape=(self.nzp, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nwp)),
                    np.zeros(shape=(self.nzu, self.nwu)),
                ],
            ]
        )
        P_21_2 = cp.bmat(
            [
                [
                    np.zeros(shape=(self.nx, self.nx)),
                    self.A_lin.cpu(),
                    np.zeros(shape=(self.nx, self.nzu)),
                ],
                [
                    np.eye(self.nx),
                    np.zeros(shape=(self.nx, self.nu)),
                    np.zeros(shape=(self.nx, self.nzu)),
                ],
                [
                    np.zeros(shape=(self.nzp, self.nx)),
                    self.C_lin.cpu(),
                    np.zeros(shape=(self.nzp, self.nzu)),
                ],
                [
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nu)),
                    np.eye(self.nzu),
                ],
            ]
        )
        Omega_tilde = cp.bmat(
            [
                [K, L1, L2, L3],
                [M1, N11, N12, N13],
                [M2, N21, N22, N23],
            ]
        )
        P_21_4 = cp.bmat(
            [
                [
                    np.zeros(shape=(self.nx, self.nx)),
                    np.eye(self.nx),
                    np.zeros(shape=(self.nx, self.nwp)),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nwp, self.nx)),
                    np.zeros(shape=(self.nwp, self.nx)),
                    np.eye(self.nwp),
                    np.zeros(shape=(self.nwp, self.nwu)),
                ],
                [
                    np.eye(self.ny),
                    np.zeros(shape=(self.ny, self.nx)),
                    np.zeros(shape=(self.ny, self.nwp)),
                    np.zeros(shape=(self.ny, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nwu, self.nx)),
                    np.zeros(shape=(self.nwu, self.nx)),
                    np.zeros(shape=(self.nwu, self.nwp)),
                    np.eye(self.nwu),
                ],
            ]
        )

        P_21 = P_21_1 + P_21_2 @ Omega_tilde @ P_21_4
        P_11 = -cp.bmat(
            [
                [
                    Y,
                    np.eye(self.nx),
                    np.zeros(shape=(self.nx, self.nwp)),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    np.eye(self.nx),
                    X,
                    np.zeros(shape=(self.nx, self.nwp)),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nwp, self.nx)),
                    np.zeros(shape=(self.nwp, self.nx)),
                    self.gamma**2 * np.eye(self.nwp),
                    np.zeros(shape=(self.nwp, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nwp)),
                    Lambda,
                ],
            ]
        )
        P_22 = -cp.bmat(
            [
                [
                    Y,
                    np.eye(self.nx),
                    np.zeros(shape=(self.nx, self.nzp)),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    np.eye(self.nx),
                    X,
                    np.zeros(shape=(self.nx, self.nzp)),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nzp, self.nx)),
                    np.zeros(shape=(self.nzp, self.nx)),
                    np.eye(self.nzp),
                    np.zeros(shape=(self.nzp, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nzp)),
                    Lambda,
                ],
            ]
        )
        P = cp.bmat([[P_11, P_21.T], [P_21, P_22]])

        # 1. run: use nontrivial objective
        t = cp.Variable(shape=(1))

        problem = cp.Problem(
            cp.Minimize(expr=t), utils.get_feasibility_constraint(P, t)
        )
        problem.solve(solver=cp.MOSEK)
        logger.info(
            f'1. run: feasibility, problem status {problem.status}, t_star = {t.value}'
        )

        assert t.value < 0
        t_fixed = np.float64(t.value + 0.1)

        # 2. run: parameter bounds
        alpha = cp.Variable(shape=(1))

        problem = cp.Problem(
            objective=cp.Minimize(expr=alpha),
            constraints=utils.get_feasibility_constraint(P, t_fixed)
            + utils.get_bounding_inequalities(X, Y, Omega_tilde, alpha),
        )
        problem.solve(solver=cp.MOSEK)

        logger.info(
            f'2. run: parameter bounds, '
            f'problem status {problem.status},'
            f'alpha_star = {alpha.value}'
        )

        alpha_fixed = np.float64(alpha.value + 0.5)

        # 3. run: make U and V well conditioned
        beta = cp.Variable(shape=(1))

        problem = cp.Problem(
            objective=cp.Maximize(expr=beta),
            constraints=utils.get_feasibility_constraint(P, t_fixed)
            + utils.get_bounding_inequalities(X, Y, Omega_tilde, alpha_fixed)
            + utils.get_conditioning_constraints(X, Y, beta),
        )
        problem.solve(solver=cp.MOSEK)
        logger.info(
            f'3. run: coupling condition,'
            f'problem status {problem.status},'
            f'beta_star = {beta.value}'
        )

        logger.info(
            f'Max real eigenvalue of P: {np.max(np.real(np.linalg.eig(P.value)[0]))}'
        )

        omega_tilde_0 = np.concatenate(
            [
                np.concatenate([K.value, L1.value, L2.value, L3.value], axis=1),
                np.concatenate(
                    [
                        M1.value,
                        N11.value,
                        N12.value,
                        N13.value,
                    ],
                    axis=1,
                ),
                np.concatenate([M2.value, N21.value, N22.value, N23], axis=1),
            ],
            axis=0,
        )

        return (
            omega_tilde_0,
            np.array(X.value, dtype=np.float64),
            np.array(Y.value, dtype=np.float64),
            np.array(Lambda.value, dtype=np.float64),
        )

    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, N, nu = x_pred.shape
        assert self.lure._nu == nu
        assert hx is not None
        x0_lin, x0_rnn = hx
        x0 = torch.concat((x0_lin, x0_rnn), dim=1).reshape(
            shape=(n_batch, self.nx * 2, 1)
        )
        us = x_pred.reshape(shape=(n_batch, N, nu, 1))
        y, x = self.lure.forward(x0=x0, us=us, return_states=True)

        # self.get_barrier(1e-3)

        return y.reshape(n_batch, N, self.lure._ny), (
            x[:, :, : self.nx].reshape(n_batch, N + 1, self.nx),
            x[:, :, self.nx :].reshape(n_batch, N + 1, self.nx),
        )

    def get_constraints(self) -> torch.Tensor:
        L_x = self.construct_lower_triangular_matrix(
            L_flat=self.L_x_flat, diag_length=self.nx
        ).to(self.device)
        L_y = self.construct_lower_triangular_matrix(
            L_flat=self.L_y_flat, diag_length=self.nx
        ).to(self.device)
        X = L_x @ L_x.T
        Y = L_y @ L_y.T
        # U = torch.linalg.inv(Y) - X
        # V = Y
        A_lin = self.A_lin
        B_lin = self.B_lin
        C_lin = self.C_lin
        Lambda = torch.diag(self.lam).to(self.device)
        omega_tilde = self.get_omega_tilde().float().to(self.device)

        P_21_1 = (
            torch.concat(
                [
                    torch.concat(
                        [
                            A_lin @ Y,
                            A_lin,
                            B_lin,
                            torch.zeros(size=(self.nx, self.nwu)).to(self.device),
                        ],
                        dim=1,
                    ),
                    torch.concat(
                        [
                            torch.zeros(size=(self.nu, self.nx)).to(self.device),
                            X @ A_lin,
                            X @ B_lin,
                            torch.zeros(size=(self.nx, self.nwu)).to(self.device),
                        ],
                        dim=1,
                    ),
                    torch.concat(
                        [
                            C_lin @ Y,
                            C_lin,
                            torch.zeros(size=(self.nzp, self.nwp)).to(self.device),
                            torch.zeros(size=(self.nzp, self.nwu)).to(self.device),
                        ],
                        dim=1,
                    ),
                    torch.concat(
                        [
                            torch.zeros(size=(self.nzu, self.nx)).to(self.device),
                            torch.zeros(size=(self.nzu, self.nx)).to(self.device),
                            torch.zeros(size=(self.nzu, self.nwp)).to(self.device),
                            torch.zeros(size=(self.nzu, self.nwu)).to(self.device),
                        ],
                        dim=1,
                    ),
                ],
                dim=0,
            )
            .float()
            .to(self.device)
        )
        P_21_2 = (
            torch.concat(
                [
                    torch.concat(
                        [
                            torch.zeros(size=(self.nx, self.nx)).to(self.device),
                            A_lin,
                            torch.zeros(size=(self.nx, self.nzu)).to(self.device),
                        ],
                        dim=1,
                    ),
                    torch.concat(
                        [
                            torch.eye(self.nx),
                            torch.zeros(size=(self.nx, self.nu)),
                            torch.zeros(size=(self.nx, self.nzu)),
                        ],
                        dim=1,
                    ).to(self.device),
                    torch.concat(
                        [
                            torch.zeros(size=(self.nzp, self.nx)).to(self.device),
                            C_lin,
                            torch.zeros(size=(self.nzp, self.nzu)).to(self.device),
                        ],
                        dim=1,
                    ),
                    torch.concat(
                        [
                            torch.zeros(size=(self.nzu, self.nx)),
                            torch.zeros(size=(self.nzu, self.nu)),
                            torch.eye(self.nzu),
                        ],
                        dim=1,
                    ).to(self.device),
                ],
                dim=0,
            )
            .float()
            .to(self.device)
        )

        P_21_4 = (
            torch.concat(
                [
                    torch.concat(
                        [
                            torch.zeros(size=(self.nx, self.nx)),
                            torch.eye(self.nx),
                            torch.zeros(size=(self.nx, self.nwp)),
                            torch.zeros(size=(self.nx, self.nwu)),
                        ],
                        dim=1,
                    ).to(self.device),
                    torch.concat(
                        [
                            torch.zeros(size=(self.nwp, self.nx)),
                            torch.zeros(size=(self.nwp, self.nx)),
                            torch.eye(self.nwp),
                            torch.zeros(size=(self.nwp, self.nwu)),
                        ],
                        dim=1,
                    ).to(self.device),
                    torch.concat(
                        [
                            torch.eye(self.ny),
                            torch.zeros(size=(self.ny, self.nx)),
                            torch.zeros(size=(self.ny, self.nwp)),
                            torch.zeros(size=(self.ny, self.nwu)),
                        ],
                        dim=1,
                    ).to(self.device),
                    torch.concat(
                        [
                            torch.zeros(size=(self.nwu, self.nx)),
                            torch.zeros(size=(self.nwu, self.nx)),
                            torch.zeros(size=(self.nwu, self.nwp)),
                            torch.eye(self.nwu),
                        ],
                        dim=1,
                    ).to(self.device),
                ],
                dim=0,
            )
            .float()
            .to(self.device)
        )

        P_21 = P_21_1.to(self.device) + P_21_2.to(self.device) @ omega_tilde.to(
            self.device
        ) @ P_21_4.to(self.device)
        P_11 = -torch.concat(
            [
                torch.concat(
                    [
                        Y,
                        torch.eye(self.nx).to(self.device),
                        torch.zeros(size=(self.nx, self.nwp)).to(self.device),
                        torch.zeros(size=(self.nx, self.nwu)).to(self.device),
                    ],
                    dim=1,
                ).to(self.device),
                torch.concat(
                    [
                        torch.eye(self.nx).to(self.device),
                        X,
                        torch.zeros(size=(self.nx, self.nwp)).to(self.device),
                        torch.zeros(size=(self.nx, self.nwu)).to(self.device),
                    ],
                    dim=1,
                ).to(self.device),
                torch.concat(
                    [
                        torch.zeros(size=(self.nwp, self.nx)).to(self.device),
                        torch.zeros(size=(self.nwp, self.nx)).to(self.device),
                        self.gamma**2 * torch.eye(self.nwp).to(self.device),
                        torch.zeros(size=(self.nwp, self.nwu)).to(self.device),
                    ],
                    dim=1,
                ).to(self.device),
                torch.concat(
                    [
                        torch.zeros(size=(self.nzu, self.nx)).to(self.device),
                        torch.zeros(size=(self.nzu, self.nx)).to(self.device),
                        torch.zeros(size=(self.nzu, self.nwp)).to(self.device),
                        Lambda,
                    ],
                    dim=1,
                ).to(self.device),
            ],
            dim=0,
        ).float()
        P_22 = -torch.concat(
            [
                torch.concat(
                    [
                        Y,
                        torch.eye(self.nx).to(self.device),
                        torch.zeros(size=(self.nx, self.nzp)).to(self.device),
                        torch.zeros(size=(self.nx, self.nwu)).to(self.device),
                    ],
                    dim=1,
                ).to(self.device),
                torch.concat(
                    [
                        torch.eye(self.nx).to(self.device),
                        X,
                        torch.zeros(size=(self.nx, self.nzp)).to(self.device),
                        torch.zeros(size=(self.nx, self.nwu)).to(self.device),
                    ],
                    dim=1,
                ).to(self.device),
                torch.concat(
                    [
                        torch.zeros(size=(self.nzp, self.nx)),
                        torch.zeros(size=(self.nzp, self.nx)),
                        torch.eye(self.nzp),
                        torch.zeros(size=(self.nzp, self.nwu)),
                    ],
                    dim=1,
                ).to(self.device),
                torch.concat(
                    [
                        torch.zeros(size=(self.nzu, self.nx)).to(self.device),
                        torch.zeros(size=(self.nzu, self.nx)).to(self.device),
                        torch.zeros(size=(self.nzu, self.nzp)).to(self.device),
                        Lambda,
                    ],
                    dim=1,
                ).to(self.device),
            ],
            dim=0,
        ).float()
        P = torch.concat(
            [torch.concat([P_11, P_21.T], dim=1), torch.concat([P_21, P_22], dim=1)],
            dim=0,
        ).to(self.device)

        return P

    def check_constraints(self) -> bool:
        P = self.get_constraints()
        _, info = torch.linalg.cholesky_ex(-P)
        return True if info == 0 else False

    def get_logdet(self, mat: torch.Tensor) -> torch.Tensor:
        # return logdet of matrix mat, if it is not positive semi-definite, return inf
        _, info = torch.linalg.cholesky_ex(mat.cpu())

        if info > 0:
            logdet = torch.tensor(float('inf')).to(self.device)
        else:
            logdet = (mat.logdet()).to(self.device)

        return logdet

    def get_barriers(self, t: torch.Tensor) -> torch.Tensor:
        L_x = self.construct_lower_triangular_matrix(
            L_flat=self.L_x_flat, diag_length=self.nx
        ).to(self.device)
        L_y = self.construct_lower_triangular_matrix(
            L_flat=self.L_y_flat, diag_length=self.nx
        ).to(self.device)

        X = L_x @ L_x.T
        Y = L_y @ L_y.T
        constraints = [
            -self.get_constraints(),
            torch.diag(torch.squeeze(self.lam)),
            torch.concat(
                (
                    torch.concat((Y, torch.eye(self.nx).to(self.device)), dim=1),
                    torch.concat((torch.eye(self.nx).to(self.device), X), dim=1),
                ),
                dim=0,
            ),
        ]

        barrier = torch.tensor(0.0).to(self.device)
        for constraint in constraints:
            barrier += -t * self.get_logdet(constraint)

        return barrier

    def project_parameters(self, write_parameter: bool = True) -> np.float64:
        if self.check_constraints():
            logger.info('No projection necessary, constraints are satisfied.')
            return np.float64(0.0)
        X = cp.Variable(shape=(self.nx, self.nx), symmetric=True)
        Y = cp.Variable(shape=(self.nx, self.nx), symmetric=True)
        lam = cp.Variable(shape=(self.nzu, 1), pos=True, name='lam')
        Lambda = cp.diag(lam)

        K = cp.Variable(shape=(self.nx, self.nx))
        L1 = cp.Variable(shape=(self.nx, self.nwp))
        L2 = cp.Variable(shape=(self.nx, self.ny))
        L3 = cp.Variable(shape=(self.nx, self.nwu))

        M1 = cp.Variable(shape=(self.nu, self.nx))
        N11 = cp.Variable(shape=(self.nu, self.nwp))
        N12 = cp.Variable(shape=(self.nu, self.ny))
        N13 = cp.Variable(shape=(self.nu, self.nwu))

        M2 = cp.Variable(shape=(self.nzu, self.nx))
        N21 = cp.Variable(shape=(self.nzu, self.nwp))
        N22 = cp.Variable(shape=(self.nzu, self.ny))
        N23 = np.zeros(shape=(self.nzu, self.nwu))
        # N23 = cp.Variable(shape=(self.nzu, self.nwu))

        P_21_1 = cp.bmat(
            [
                [
                    self.A_lin.cpu() @ Y,
                    self.A_lin.cpu(),
                    self.B_lin.cpu(),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nu, self.nx)),
                    X @ self.A_lin.cpu(),
                    X @ self.B_lin.cpu(),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    self.C_lin.cpu() @ Y,
                    self.C_lin.cpu(),
                    np.zeros(shape=(self.nzp, self.nwp)),
                    np.zeros(shape=(self.nzp, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nwp)),
                    np.zeros(shape=(self.nzu, self.nwu)),
                ],
            ]
        )
        P_21_2 = cp.bmat(
            [
                [
                    np.zeros(shape=(self.nx, self.nx)),
                    self.A_lin.cpu(),
                    np.zeros(shape=(self.nx, self.nzu)),
                ],
                [
                    np.eye(self.nx),
                    np.zeros(shape=(self.nx, self.nu)),
                    np.zeros(shape=(self.nx, self.nzu)),
                ],
                [
                    np.zeros(shape=(self.nzp, self.nx)),
                    self.C_lin.cpu(),
                    np.zeros(shape=(self.nzp, self.nzu)),
                ],
                [
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nu)),
                    np.eye(self.nzu),
                ],
            ]
        )
        P_21_4 = cp.bmat(
            [
                [
                    np.zeros(shape=(self.nx, self.nx)),
                    np.eye(self.nx),
                    np.zeros(shape=(self.nx, self.nwp)),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nwp, self.nx)),
                    np.zeros(shape=(self.nwp, self.nx)),
                    np.eye(self.nwp),
                    np.zeros(shape=(self.nwp, self.nwu)),
                ],
                [
                    np.eye(self.ny),
                    np.zeros(shape=(self.ny, self.nx)),
                    np.zeros(shape=(self.ny, self.nwp)),
                    np.zeros(shape=(self.ny, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nwu, self.nx)),
                    np.zeros(shape=(self.nwu, self.nx)),
                    np.zeros(shape=(self.nwu, self.nwp)),
                    np.eye(self.nwu),
                ],
            ]
        )
        Omega_tilde = cp.bmat(
            [
                [K, L1, L2, L3],
                [M1, N11, N12, N13],
                [M2, N21, N22, N23],
            ]
        )
        P_21 = P_21_1 + P_21_2 @ Omega_tilde @ P_21_4
        P_11 = -cp.bmat(
            [
                [
                    Y,
                    np.eye(self.nx),
                    np.zeros(shape=(self.nx, self.nwp)),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    np.eye(self.nx),
                    X,
                    np.zeros(shape=(self.nx, self.nwp)),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nwp, self.nx)),
                    np.zeros(shape=(self.nwp, self.nx)),
                    self.gamma**2 * np.eye(self.nwp),
                    np.zeros(shape=(self.nwp, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nwp)),
                    Lambda,
                ],
            ]
        )
        P_22 = -cp.bmat(
            [
                [
                    Y,
                    np.eye(self.nx),
                    np.zeros(shape=(self.nx, self.nzp)),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    np.eye(self.nx),
                    X,
                    np.zeros(shape=(self.nx, self.nzp)),
                    np.zeros(shape=(self.nx, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nzp, self.nx)),
                    np.zeros(shape=(self.nzp, self.nx)),
                    np.eye(self.nzp),
                    np.zeros(shape=(self.nzp, self.nwu)),
                ],
                [
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nx)),
                    np.zeros(shape=(self.nzu, self.nzp)),
                    Lambda,
                ],
            ]
        )
        P = cp.bmat([[P_11, P_21.T], [P_21, P_22]])
        nP = P.shape[0]

        device = self.get_omega_tilde().device
        Omega_tilde_0 = self.get_omega_tilde().cpu().detach().numpy()
        L_x = (
            self.construct_lower_triangular_matrix(self.L_x_flat, self.nx)
            .cpu()
            .detach()
            .numpy()
        )
        L_y = (
            self.construct_lower_triangular_matrix(self.L_y_flat, self.nx)
            .cpu()
            .detach()
            .numpy()
        )
        X_0 = L_x @ L_x.T
        Y_0 = L_y @ L_y.T
        Lambda_0 = torch.diag(self.lam).cpu().detach().numpy()

        feasibility_constraint = [P << -1e-4 * np.eye(nP)]

        problem = cp.Problem(
            cp.Minimize(
                cp.norm(Omega_tilde - Omega_tilde_0)
                + cp.norm(Lambda - Lambda_0)
                + cp.norm(X - X_0)
                + cp.norm(Y - Y_0)
            ),
            feasibility_constraint,
        )
        problem.solve(solver=cp.MOSEK)

        d = np.linalg.norm(Omega_tilde.value - Omega_tilde_0)
        +np.linalg.norm(Lambda.value - Lambda_0)
        +np.linalg.norm(X.value - X_0)
        +np.linalg.norm(Y.value - Y_0)

        logger.info(
            f'1. run: projection. '
            f'problem status {problem.status},'
            f'||X-X_0|| + ||Y-Y_0|| + ||Omega - Omega_0|| + ||Lambda-Lambda_0|| = {d}'
        )

        alpha = cp.Variable(shape=(1,))
        problem = cp.Problem(
            cp.Minimize(expr=alpha),
            feasibility_constraint
            + utils.get_bounding_inequalities(X, Y, Omega_tilde, alpha),
        )
        problem.solve(solver=cp.MOSEK)
        logger.info(
            f'2. run: parameter bounds. '
            f'problem status {problem.status},'
            f'alpha_star = {alpha.value}'
        )

        alpha_fixed = np.float64(alpha.value + 0.1)

        beta = cp.Variable(shape=(1,))
        problem = cp.Problem(
            cp.Maximize(expr=beta),
            feasibility_constraint
            + utils.get_bounding_inequalities(X, Y, Omega_tilde, alpha_fixed)
            + utils.get_conditioning_constraints(Y, X, beta),
        )
        problem.solve(solver=cp.MOSEK)
        logger.info(
            f'3. run: coupling conditions. '
            f'problem status {problem.status},'
            f'beta_star = {beta.value}'
        )

        if not write_parameter:
            logger.info('Return distance.')
            return np.float64(d)

        logger.info('Write back projected parameters.')
        self.L_x_flat.data = (
            torch.tensor(
                self.extract_vector_from_lower_triangular_matrix(
                    np.linalg.cholesky(np.array(X.value))
                )
            )
            .float()
            .to(device)
        )
        self.L_y_flat.data = (
            torch.tensor(
                self.extract_vector_from_lower_triangular_matrix(
                    np.linalg.cholesky(np.array(Y.value))
                )
            )
            .float()
            .to(device)
        )
        self.lam.data = torch.tensor(np.diag(np.array(Lambda.value))).float().to(device)

        self.K.data = torch.tensor(K.value).float().to(device)
        self.L1.data = torch.tensor(L1.value).float().to(device)
        self.L2.data = torch.tensor(L2.value).float().to(device)
        self.L3.data = torch.tensor(L3.value).float().to(device)
        self.M1.data = torch.tensor(M1.value).float().to(device)
        self.N11.data = torch.tensor(N11.value).float().to(device)
        self.N12.data = torch.tensor(N12.value).float().to(device)
        self.N13.data = torch.tensor(N13.value).float().to(device)
        self.M2.data = torch.tensor(M2.value).float().to(device)
        self.N21.data = torch.tensor(N21.value).float().to(device)
        self.N22.data = torch.tensor(N22.value).float().to(device)
        # self.N23.data = torch.tensor(N23).float().to(device)

        return np.float64(d)

    def write_parameters(self, params: List[torch.Tensor]) -> None:
        for old_par, new_par in zip(params, self.parameters()):
            new_par.data = old_par.clone()


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

        with warnings.catch_warnings():
            self.init_lstm = nn.LSTM(
                input_size=input_dim + output_dim,
                hidden_size=recurrent_dim,
                num_layers=num_recurrent_layers,
                dropout=dropout,
                batch_first=True,
            )

            self.predictor_lstm = nn.LSTM(
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
        self,
        input: torch.Tensor,
        x0: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h_init, (h0_init, c0_init) = self.init_lstm(x0)
        h, (_, _) = self.predictor_lstm(input, (h0_init, c0_init))

        return self.output_layer(h), self.init_layer(h_init)


class InitializerPredictorLSTM(nn.Module):
    """
    Variation of InitLSTM.
    InitLSTM is somewhat broken, because it assumes that the initializer
    receives input_dim + output_dim as input but this might actually differ
    for some models.
    """

    def __init__(
        self,
        predictor_input_dim: int,
        initializer_input_dim: int,
        output_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        dropout: float,
    ):
        super().__init__()

        self.num_recurrent_layers = num_recurrent_layers
        self.recurrent_dim = recurrent_dim

        with warnings.catch_warnings():
            self.init_lstm = nn.LSTM(
                input_size=initializer_input_dim,
                hidden_size=recurrent_dim,
                num_layers=num_recurrent_layers,
                dropout=dropout,
                batch_first=True,
            )

            self.predictor_lstm = nn.LSTM(
                input_size=predictor_input_dim,
                hidden_size=recurrent_dim,
                num_layers=num_recurrent_layers,
                dropout=dropout,
                batch_first=True,
            )

        self.output_layer = torch.nn.Linear(
            in_features=recurrent_dim, out_features=output_dim, bias=False
        )

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        for name, param in self.init_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

    def forward(
        self,
        predictor_input: torch.Tensor,
        initializer_input: torch.Tensor,
    ) -> torch.Tensor:
        _, (h0_init, c0_init) = self.init_lstm.forward(initializer_input)
        h, (_, _) = self.predictor_lstm.forward(predictor_input, (h0_init, c0_init))
        y = self.output_layer.forward(h)

        return y
