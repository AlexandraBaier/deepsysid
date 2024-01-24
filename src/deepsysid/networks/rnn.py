import abc
import logging
import warnings
from typing import Callable, List, Optional, Tuple, Union, Literal

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn

from ..models.utils import SimAbcdParameter
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

        # https://yalmip.github.io/faq/semidefiniteelementwise/
        # symmetrize variable
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

    # def forward(
    #     self, x0: torch.Tensor, us: torch.Tensor, return_states: bool = False
    # ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    #     n_batch, N, _, _ = us.shape
    #     x_int = torch.zeros(size=(n_batch, N + 1, self._nx, 1)).to(self.device)
    #     y = torch.zeros(size=(n_batch, N, self._ny, 1)).to(self.device)
    #     w = torch.zeros(size=(n_batch, N, self._nw, 1)).to(self.device)
    #     x_int[:, 0, :, :] = x0

    #     for k in range(N):
    #         w[:, k, :, :] = self.Delta(
    #             self.C2 @ x_int[:, k, :, :] + self.D21 @ us[:, k, :, :]
    #         )
    #         x_int[:, k + 1, :, :] = (
    #             super().state_dynamics(x=x_int[:, k, :, :], u=us[:, k, :, :])
    #             + self.B2 @ w[:, k, :, :]
    #         )
    #         y[:, k, :, :] = (
    #             super().output_dynamics(x=x_int[:, k, :, :], u=us[:, k, :, :])
    #             + self.D12 @ w[:, k, :, :]
    #         )
    #     x = x_int[:,:N,:,:]
    #     if return_states:
    #         return (y, x)
    #     else:
    #         return y

    def forward(
        self, x0: torch.Tensor, us: torch.Tensor, return_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, N, _, _ = us.shape
        # x = torch.zeros(size=(n_batch, N + 1, self._nx, 1)).to(self.device)
        y = torch.zeros(size=(n_batch, N, self._ny, 1)).to(self.device)
        # w = torch.zeros(size=(n_batch, self._nw, 1)).to(self.device)
        x = x0.reshape(n_batch, self._nx, 1)

        for k in range(N):
            w = self.Delta(self.C2 @ x + self.D21 @ us[:, k, :, :])
            x = super().state_dynamics(x=x, u=us[:, k, :, :]) + self.B2 @ w
            y[:, k, :, :] = (
                super().output_dynamics(x=x, u=us[:, k, :, :]) + self.D12 @ w
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
            matrix = matrix.cpu().detach()


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
        normalize_rnn: Optional[bool] = False,
        optimizer: str = cp.SCS,
        enforce_constraints_method: Optional[str] = 'barrier',
        controller_feedback: Optional[Literal['cf', 'nf', 'signal']] = 'cf',
        multiplier_type: Optional[str] = 'diag',
        loop_trafo: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.nx = A_lin.shape[0]  # state size
        self.ny = self.nx  # output size of linearization
        self.nwp = B_lin.shape[1]  # input size of performance channel
        self.nzp = C_lin.shape[0]  # output size of performance channel
        if controller_feedback == 'cf':
            self.nu = self.nx  # input size of linearization
        elif controller_feedback == 'nf':
            self.nu = self.nzp  # rnn should only predict the output
        elif controller_feedback == 'signal':
            self.nu = self.nx + self.nzp

        assert nzu == nwu
        self.nzu = nzu  # output size of uncertainty channel
        self.nwu = nwu  # input size of uncertainty channel

        self.optimizer = optimizer
        self.multiplier_type = multiplier_type
        self.loop_trafo = loop_trafo

        self._x_mean: Optional[torch.Tensor] = None
        self._x_std: Optional[torch.Tensor] = None
        self._wp_mean: Optional[torch.Tensor] = None
        self._wp_std: Optional[torch.Tensor] = None

        self.normalize_rnn = normalize_rnn

        self.alpha = alpha
        self.beta = beta

        self.device = device

        self.A_lin = torch.tensor(A_lin, dtype=torch.float64).to(device)
        self.B_lin = torch.tensor(B_lin, dtype=torch.float64).to(device)
        self.C_lin = torch.tensor(C_lin, dtype=torch.float64).to(device)

        self.nl = nonlinearity

        self.gamma = gamma

        L_flat_size = self.extract_vector_from_lower_triangular_matrix(
            torch.zeros(size=(self.nx, self.nx))
        ).shape[0]

        if enforce_constraints_method is None:
            self.L_x_flat = (
                torch.normal(0, 1 / self.nx, size=(L_flat_size,)).double().to(device)
            )
            self.L_y_flat = (
                torch.normal(0, 1 / self.nx, size=(L_flat_size,)).double().to(device)
            )
        else:
            self.L_x_flat = torch.nn.Parameter(
                torch.normal(0, 1 / self.nx, size=(L_flat_size,)).double().to(device)
            )
            self.L_y_flat = torch.nn.Parameter(
                torch.normal(0, 1 / self.nx, size=(L_flat_size,)).double().to(device)
            )

        if self.multiplier_type == 'diagonal':
            self.lam = torch.nn.Parameter(
                torch.ones(size=(self.nzu,)).double().to(device)
            )
        elif self.multiplier_type == 'static_zf':
            self.lam = torch.nn.Parameter(torch.eye(self.nzu).double().to(device))
        else:
            raise ValueError(f'Multiplier type {self.multiplier_type} not supported.')
        
        self.K = torch.nn.Parameter(torch.zeros(size=(self.nx,self.nx))).to(device)
        self.L1 = torch.nn.Parameter(torch.zeros(size=(self.nx, self.nwp))).to(device)
        self.L2 = torch.nn.Parameter(torch.zeros(size=(self.nx, self.ny))).to(device)
        self.L3 = torch.nn.Parameter(torch.zeros(size=(self.nx, self.nwu))).to(device)

        self.M1 = torch.nn.Parameter(torch.zeros(size=(self.nx,self.nx))).to(device)
        self.N11 = torch.nn.Parameter(torch.zeros(size=(self.nx, self.nwp))).to(device)
        self.N12 = torch.nn.Parameter(torch.zeros(size=(self.nx, self.ny))).to(device)
        self.N13 = torch.nn.Parameter(torch.zeros(size=(self.nx,self.nwu))).to(device)
        
        self.M2 = torch.nn.Parameter(torch.zeros(size=(self.nzp,self.nx))).to(device)
        self.N21 = torch.nn.Parameter(torch.zeros(size=(self.nzp, self.nwp))).to(device)
        self.N22 = torch.nn.Parameter(torch.zeros(size=(self.nzp, self.ny))).to(device)
        self.N23 = torch.nn.Parameter(torch.zeros(size=(self.nzp,self.nwu))).to(device)

        self.M3 = torch.nn.Parameter(torch.zeros(size=(self.nzu, self.nx))).to(device)
        self.N31 = torch.nn.Parameter(torch.zeros(size=(self.nzu, self.nwp))).to(device)
        self.N32 = torch.nn.Parameter(torch.zeros(size=(self.nzu, self.ny))).to(device)
        self.N33 = torch.zeros(size=(self.nzu, self.nwu),requires_grad=False).to(device)

        # self.klmn = torch.cat(
        #     [
        #         torch.cat([self.K,self.L1,self.L2], dim=1),
        #         torch.cat([self.M1,self.N11,self.N12], dim=1),
        #         torch.cat([self.M2,self.N21,self.N22], dim=1)
        #     ], dim=0
        # ).to(device)

        # self.klmn = torch.nn.Parameter(
        #     torch.zeros(
        #         size=(
        #             self.nx + self.nu + self.nzu,
        #             self.nx + self.nwp + self.ny + self.nwu,
        #         )
        #     )
        # ).to(device)

        if self.normalize_rnn:
            B_lin_hat = np.hstack(
                (B_lin, np.zeros(shape=(B_lin.shape[0], self.nwp + self.nx)))
            )
            self.nwp_hat = self.nwp + self.nwp + self.nx
        else:
            B_lin_hat = B_lin
            self.nwp_hat = self.nwp

        self.controller_feedback = controller_feedback

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
                dtype=np.float64,
            )
        ).to(device)
        if controller_feedback == 'cf':
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
                    dtype=np.float64,
                )
            ).to(device)
        elif controller_feedback=='nf':
            self.S_l = torch.from_numpy(
                np.concatenate(
                    [
                        np.concatenate(
                            [
                                np.zeros(shape=(self.nx, self.nx)),
                                np.zeros(shape=(self.nx, self.nu)),
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
                                -np.eye(self.nzp),
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
                    dtype=np.float64,
                )
            ).to(device)
        elif controller_feedback=='signal':
            self.S_l = torch.from_numpy(
                np.concatenate(
                    [
                        np.concatenate(
                            [
                                np.zeros(shape=(self.nx, self.nx)),
                                np.eye(self.nx),
                                np.zeros(shape=(self.nx, self.nzp)),
                                np.zeros(shape=(self.nx, self.nzu)),
                            ],
                            axis=1,
                        ),
                        np.concatenate(
                            [
                                np.eye(self.nx),
                                np.zeros(shape=(self.nx, self.nx)),
                                np.zeros(shape=(self.nx, self.nzp)),
                                np.zeros(shape=(self.nx, self.nzu)),
                            ],
                            axis=1,
                        ),
                        np.concatenate(
                            [
                                np.zeros(shape=(self.nzp, self.nx)),
                                np.zeros(shape=(self.nzp, self.nx)),
                                np.eye(self.nzp),
                                np.zeros(shape=(self.nzp, self.nzu)),
                            ],
                            axis=1,
                        ),
                        np.concatenate(
                            [
                                np.zeros(shape=(self.nzu, self.nx)),
                                np.zeros(shape=(self.nzu, self.nx)),
                                np.zeros(shape=(self.nzu, self.nzp)),
                                np.eye(self.nzu),
                            ],
                            axis=1,
                        ),
                    ],
                    axis=0,
                    dtype=np.float64,
                )
            )
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
                dtype=np.float64,
            )
        ).to(device)

    def initialize_lmi(self) -> None:
        # 1. solve synthesis inequalities to find feasible parameter set
        klmn_0, X, Y, Lambda = self.get_initial_parameters()

        L_x = np.linalg.cholesky(a=X)
        L_x_flat = self.extract_vector_from_lower_triangular_matrix(L_x)
        self.L_x_flat.data = torch.tensor(L_x_flat).double()

        L_y = np.linalg.cholesky(a=Y)
        L_y_flat = self.extract_vector_from_lower_triangular_matrix(L_y)
        self.L_y_flat.data = torch.tensor(L_y_flat).double()

        self.lam.data = torch.tensor(np.diag(Lambda)).double()

        # self.klmn.data = torch.tensor(klmn_0).double()
        self.K.data = torch.tensor(klmn_0[:self.nx,:self.nx]).double()
        self.L1.data = torch.tensor(klmn_0[:self.nx,self.nx:self.nx+self.nwp+self.nu]).double()
        self.L2.data = torch.tensor(klmn_0[:self.nx,self.nx+self.nwp+self.nu:]).double()
        
        self.M1.data = torch.tensor(klmn_0[self.nx:self.nx+self.nu,:self.nx]).double()
        self.N11.data = torch.tensor(klmn_0[self.nx:self.nx+self.nu,self.nx:self.nx+self.nwp+self.nu]).double()
        self.N12.data = torch.tensor(klmn_0[self.nx:self.nx+self.nu,self.nx+self.nwp+self.nu:]).double()

        self.M2.data = torch.tensor(klmn_0[self.nx+self.nu:,:self.nx]).double()
        self.N21.data = torch.tensor(klmn_0[self.nx+self.nu:,self.nx:self.nx+self.nwp+self.nu]).double()




        # self.N22.data = torch.tensor(klmn_0[self.nx+self.nu:,self.nx+self.nwp+self.nu:]).double()

    def construct_lower_triangular_matrix(
        self, L_flat: torch.Tensor, diag_length: int
    ) -> torch.Tensor:
        device = L_flat.device
        flat_idx = 0
        L = torch.zeros(
            size=(diag_length, diag_length), dtype=torch.float64, device=device
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

        if self.controller_feedback == 'cf':
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
            ).double()
        elif self.controller_feedback == 'nf':
            T_l = torch.concat(
                [
                    torch.concat(
                        [
                            U,
                            torch.zeros((self.nx, self.nzp)).to(self.device),
                            torch.zeros((self.nx, self.nzu)).to(self.device),
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
            ).double()
        elif self.controller_feedback == 'signal':
            T_l = torch.concat(
                [
                    torch.concat(
                        [
                            U,
                            X,
                            torch.zeros((self.nx, self.nzp)).to(self.device),
                            torch.zeros((self.nx, self.nzu)).to(self.device),
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
                            torch.zeros((self.nzu, self.nx)).to(self.device),
                            torch.zeros((self.nzu, self.nzp)).to(self.device),
                            Lambda,
                        ],
                        dim=1,
                    ),
                ],
                dim=0,
            ).double()


        T_r = torch.concat(
            [
                torch.concat(
                    [
                        torch.zeros(size=(self.nx, self.nx)).to(self.device),
                        torch.zeros(size=(self.nx, self.nwp)).to(self.device),
                        V.T,
                        torch.zeros(size=(self.nx, self.nwu)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros(size=(self.nwp, self.nx)),
                        torch.eye(self.nwp),
                        torch.zeros(size=(self.nwp, self.ny)),
                        torch.zeros(size=(self.nwp, self.nwu)),
                    ],
                    dim=1,
                ).to(self.device),
                torch.concat(
                    [
                        torch.eye(self.ny).to(self.device),
                        torch.zeros(size=(self.ny, self.nwp)).to(self.device),
                        Y,
                        torch.zeros(size=(self.ny, self.nwu)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros(size=(self.nwu, self.nx)),
                        torch.zeros(size=(self.nwu, self.nwp)),
                        torch.zeros(size=(self.nwu, self.ny)),
                        torch.eye(self.nwu),
                    ],
                    dim=1,
                ).to(self.device),
            ],
            dim=0,
        ).double()
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
        ).double()

        return (T_l, T_r, T_s)
    
    def get_klmn(self) -> torch.Tensor:
        return torch.cat(
            [
                torch.cat([self.K,self.L1,self.L2, self.L3], dim=1),
                torch.cat([self.M1,self.N11,self.N12, self.N13], dim=1),
                torch.cat([self.M2,self.N21,self.N22, self.N23], dim=1),
                torch.cat([self.M3,self.N31,self.N32, self.N33], dim=1)
            ], dim=0
        ).to(self.device)
        # return self.klmn.to(self.device)


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
        U = torch.linalg.inv(Y) - X
        V = Y

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
        # klmn_0 = self.klmn.to(device)

        if self.multiplier_type == 'diagonal':
            Lambda = torch.diag(input=self.lam).to(device)
        elif self.multiplier_type == 'static_zf':
            Lambda = self.lam.to(device)

        if not self.loop_trafo:
            Lambda_inv = torch.linalg.inv(Lambda)
            self.M3.value = Lambda_inv @ self.M3
            self.N31.value = Lambda_inv @ self.N31
            self.N32.value = Lambda_inv @ self.N32

        klmn_0 = self.get_klmn()

        # construct X, Y, and Lambda
        X, Y, U, V = self.get_coupling_matrices()


        # transform to original parameters
        T_l, T_r, T_s = self.get_T(X, Y, U, V, Lambda)
        abcd_0 = (
            torch.linalg.inv(T_l).double().to(device)
            @ (klmn_0 - T_s.double().to(device))
            @ torch.linalg.inv(T_r).double().to(device)
        )

        if self.loop_trafo:
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
            ) = self.get_controller_matrices(abcd_0)

            if self.normalize_rnn:
                B1_hat = B1_tilde @ torch.diag(1 / self._wp_std)
                B2_hat = B2_tilde @ torch.diag(1 / self._x_std)
                C1_hat = torch.diag(self._x_std) @ C1_tilde
                D11_hat = torch.diag(self._x_std) @ D11_tilde @ torch.diag(1 / self._wp_std)
                D12_hat = torch.diag(self._x_std) @ D12_tilde @ torch.diag(1 / self._x_std)
                D13_hat = torch.diag(self._x_std) @ D13_tilde
                D21_hat = D21 @ torch.diag(1 / self._wp_std)
                D22_hat = D22 @ torch.diag(1 / self._x_std)

                abcd = self.get_abcd(
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
            else:
                abcd = self.get_abcd(
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
        else:
            abcd = abcd_0

        sys_block_matrix = self.S_s + self.S_l @ abcd @ self.S_r

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
        # assert torch.linalg.norm(D22_cal) - 0 < 1e-2
        # D22_cal = torch.zeros_like(D22_cal)

        self.lure = LureSystem(
            A=A_cal,
            B1=B1_cal,
            B2=B2_cal,
            C1=C1_cal,
            D11=D11_cal,
            D12=D12_cal,
            C2=C2_cal,
            D21=D21_cal,
            Delta=self.nl,
            device=self.device,
        ).to(device)

        return (abcd_0.cpu().detach().numpy(), sys_block_matrix.cpu().detach().numpy())

    def get_controller_matrices(
        self, abcd: torch.Tensor
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
        A_tilde = abcd[: self.nx, : self.nx]
        B1_tilde = abcd[: self.nx, self.nx : self.nx + self.nwp]
        B2_tilde = abcd[: self.nx, self.nx + self.nwp : self.nx + self.nwp + self.ny]
        B3_tilde = abcd[: self.nx, self.nx + self.nwp + self.ny :]

        C1_tilde = abcd[self.nx : self.nx + self.nu, : self.nx]
        D11_tilde = abcd[self.nx : self.nx + self.nu, self.nx : self.nx + self.nwp]
        D12_tilde = abcd[
            self.nx : self.nx + self.nu,
            self.nx + self.nwp : self.nx + self.nwp + self.ny,
        ]
        D13_tilde = abcd[self.nx : self.nx + self.nu, self.nx + self.nwp + self.ny :]

        C2 = abcd[self.nx + self.nu :, : self.nx]
        D21 = abcd[self.nx + self.nu :, self.nx : self.nx + self.nwp]
        D22 = abcd[
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

    def get_abcd(
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
        # J = torch.tensor(2 / (self.alpha - self.beta), device=self.device)
        J = torch.tensor(2 / (self.beta - self.alpha), device=self.device)
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
        if self.controller_feedback:
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
        else:
            P_21_2 = cp.bmat(
                [
                    [
                        np.zeros(shape=(self.nx, self.nx)),
                        np.zeros(shape=(self.nx, self.nu)),
                        np.zeros(shape=(self.nx, self.nzu)),
                    ],
                    [
                        np.eye(self.nx),
                        np.zeros(shape=(self.nx, self.nu)),
                        np.zeros(shape=(self.nx, self.nzu)),
                    ],
                    [
                        np.zeros(shape=(self.nzp, self.nx)),
                        -np.eye(self.nu),
                        np.zeros(shape=(self.nzp, self.nzu)),
                    ],
                    [
                        np.zeros(shape=(self.nzu, self.nx)),
                        np.zeros(shape=(self.nzu, self.nu)),
                        np.eye(self.nzu),
                    ],
                ]
            )

        klmn = cp.Variable(
            shape=(
                self.nx + self.nu + self.nzu,
                self.nx + self.nwp + self.ny + self.nwu,
            )
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

        P_21 = P_21_1 + P_21_2 @ klmn @ P_21_4
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
                    2*Lambda,
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
                    1/2*Lambda,
                ],
            ]
        )
        P = cp.bmat([[P_11, P_21.T], [P_21, P_22]])

        # 1. run: use nontrivial objective
        t = cp.Variable(shape=(1))

        problem = cp.Problem(
            cp.Minimize(expr=t), utils.get_feasibility_constraint(P, t)
        )
        problem.solve(solver=self.optimizer)
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
            + utils.get_bounding_inequalities(X, Y, klmn, alpha),
        )
        problem.solve(solver=self.optimizer)

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
            + utils.get_bounding_inequalities(X, Y, klmn, alpha_fixed)
            + utils.get_conditioning_constraints(X, Y, beta),
        )
        problem.solve(solver=self.optimizer)
        logger.info(
            f'3. run: coupling condition,'
            f'problem status {problem.status},'
            f'beta_star = {beta.value}'
        )

        logger.info(
            f'Max real eigenvalue of P: {np.max(np.real(np.linalg.eig(P.value)[0]))}'
        )

        return (
            klmn.value,
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
        # assert hx is not None
        if hx is None:
            x0_lin = torch.zeros(size=(n_batch,self.nx)).to(self.device)
            x0_rnn = torch.zeros(size=(n_batch,self.nx)).to(self.device)
        else:
            x0_lin, x0_rnn = hx
        x0 = torch.concat((x0_lin, x0_rnn), dim=1).reshape(
            shape=(n_batch, self.nx * 2, 1)
        )
        us = x_pred.reshape(shape=(n_batch, N, nu, 1))
        y, x = self.lure.forward(x0=x0, us=us, return_states=True)

        # self.get_barrier(1e-3)

        return y.reshape(n_batch, N, self.lure._ny), (
            x[:, : self.nx].reshape(n_batch, self.nx),
            x[:, self.nx :].reshape(n_batch, self.nx),
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

        if self.multiplier_type == 'diagonal':
            Lambda = torch.diag(self.lam).to(self.device)
        elif self.multiplier_type == 'static_zf':
            Lambda = self.lam.to(self.device)
        # klmn = self.get_klmn().double().to(self.device)
        # klmn = self.klmn.double().to(self.device)

        

        if self.loop_trafo:
            klmn = self.get_klmn().double()
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
                                torch.zeros(size=(self.nx, self.nx)).to(self.device),
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
                .double()
                .to(self.device)
            )
            if self.controller_feedback == 'cf':
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
                    .double()
                    .to(self.device)
                )
            elif self.controller_feedback == 'nf':
                P_21_2 = (
                    torch.concat(
                        [
                            torch.concat(
                                [
                                    torch.zeros(size=(self.nx, self.nx)).to(self.device),
                                    torch.zeros(size=(self.nx, self.nu)).to(self.device),
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
                                    -torch.eye(self.nzp).to(self.device),
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
                    .double()
                    .to(self.device)
                )
            elif self.controller_feedback =='signal':
                P_21_2 = (
                    torch.concat(
                        [
                            torch.concat(
                                [
                                    torch.zeros(size=(self.nx, self.nx)).to(self.device),
                                    torch.eye(self.nx).to(self.device),
                                    torch.zeros(size=(self.nx, self.nzp)).to(self.device),
                                    torch.zeros(size=(self.nx, self.nzu)).to(self.device),
                                ],
                                dim=1,
                            ),
                            torch.concat(
                                [
                                    torch.eye(self.nx),
                                    torch.zeros(size=(self.nx, self.nx)),
                                    torch.zeros(size=(self.nx, self.nzp)),
                                    torch.zeros(size=(self.nx, self.nzu)),
                                ],
                                dim=1,
                            ).to(self.device),
                            torch.concat(
                                [
                                    torch.zeros(size=(self.nzp, self.nx)).to(self.device),
                                    torch.zeros(size=(self.nzp, self.nx)).to(self.device),
                                    torch.eye(self.nzp).to(self.device),
                                    torch.zeros(size=(self.nzp, self.nzu)).to(self.device),
                                ],
                                dim=1,
                            ),
                            torch.concat(
                                [
                                    torch.zeros(size=(self.nzu, self.nx)),
                                    torch.zeros(size=(self.nzu, self.nx)),
                                    torch.zeros(size=(self.nzu, self.nzp)),
                                    torch.eye(self.nzu),
                                ],
                                dim=1,
                            ).to(self.device),
                        ],
                        dim=0,
                    )
                    .double()
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
                .double()
                .to(self.device)
            )

            P_21 = P_21_1 + P_21_2 @ klmn @ P_21_4
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
                            2*Lambda,
                        ],
                        dim=1,
                    ).to(self.device),
                ],
                dim=0,
            ).double()
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
                            1/2*Lambda,
                        ],
                        dim=1,
                    ).to(self.device),
                ],
                dim=0,
            ).double()
            P = torch.concat(
                [
                    torch.concat([P_11, P_21.T], dim=1), 
                    torch.concat([P_21, P_22], dim=1)
                ],dim=0,
            ).to(self.device)

        else:
            M3_tilde = self.M3
            N31_tilde = self.N31
            N32_tilde = self.N32

            P_11 = torch.concat([
                torch.concat([-Y, -torch.eye(self.nx), torch.zeros(size=(self.nx,self.nwp)), self.beta * N32_tilde.T], dim=1),
                torch.concat([-torch.eye(self.nx), -X, torch.zeros(size=(self.nx,self.nwp)), self.beta * M3_tilde.T], dim=1),
                torch.concat([torch.zeros(size=(self.nwp, self.nx)), torch.zeros(size=(self.nwp, self.nx)),-self.gamma**2*torch.eye(self.nwp), self.beta * N31_tilde.T ],dim=1 ),
                torch.concat([self.beta *N32_tilde, self.beta*M3_tilde, self.beta*N31_tilde, -(Lambda.T + Lambda)], dim=1)
            ], dim =0).to(self.device)

            P_21 = torch.concat([
                torch.concat([A_lin @ Y + self.N12, A_lin + self.M1, B_lin + self.N11, self.N13], dim=1),
                torch.concat([self.L2, X@A_lin+self.K, X@B_lin+self.L1, self.L3],dim=1),
                torch.concat([C_lin@Y+self.N22, C_lin + self.M2, self.N21, self.N23], dim=1)
            ], dim=0).to(self.device)
            P_22 = torch.concat([
                torch.concat([-Y, -torch.eye(self.nx), torch.zeros(size=(self.nx,self.nzp))], dim =1),
                torch.concat([-torch.eye(self.nx), -X, torch.zeros(size=(self.nx,self.nzp))], dim = 1),
                torch.concat([torch.zeros(size=(self.nzp, self.nx)), torch.zeros(size=(self.nzp,self.nx)), -torch.eye(self.nzp)], dim=1)
            ], dim=0).to(self.device)
            P = torch.concat([
                torch.concat([P_11, P_21.T], dim=1),
                torch.concat([P_21, P_22], dim=1)
            ], dim=0).to(self.device)

        if self.multiplier_type == 'diagonal':
            # https://yalmip.github.io/faq/semidefiniteelementwise/
            # symmetrize variable
            return 0.5 * (P + P.T)
        else:
            return P

    def check_constraints(self) -> bool:
        with torch.no_grad():
            P = self.get_constraints()
            _, info = torch.linalg.cholesky_ex(-P)
        return True if info == 0 else False

    def get_logdet(self, mat: torch.Tensor) -> torch.Tensor:
        # return logdet of matrix mat, if it is not positive semi-definite, return inf
        if len(mat.shape) < 2:
            return torch.log(mat).to(self.device)

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

        multiplier_constraints = []
        if self.multiplier_type == 'diagonal':
            multiplier_constraints.append(torch.diag(torch.squeeze(self.lam)))
        elif self.multiplier_type == 'static_zf':
            multiplier_constraints.extend(
                list(
                    torch.squeeze(
                        torch.ones(size=(self.nwu, 1)).double().to(self.device).T
                        @ self.lam
                    )
                )
            ),
            multiplier_constraints.extend(
                list(
                    torch.squeeze(
                        self.lam
                        @ torch.ones(size=(self.nwu, 1)).double().to(self.device)
                    )
                )
            )
            for col_idx in range(self.nwu):
                for row_idx in range(self.nwu):
                    if not (row_idx == col_idx):
                        multiplier_constraints.append(-self.lam[col_idx, row_idx])

        constraints = [
            -self.get_constraints(),
            *multiplier_constraints,
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

        multiplier_constraints = []
        logger.info(f'Multiplier type: {self.multiplier_type}')
        if self.multiplier_type == 'diagonal':
            # diagonal multiplier, elements need to be positive
            lam = cp.Variable(shape=(self.nzu, 1))
            Lambda = cp.diag(lam)

        elif self.multiplier_type == 'static_zf':
            # static zames falb multiplier, Lambda must be double hyperdominant
            Lambda = cp.Variable(shape=(self.nzu, self.nwu))
            multiplier_constraints.extend(
                [
                    np.ones(shape=(self.nwu, 1)).T @ Lambda >= 0,
                    Lambda @ np.ones(shape=(self.nwu, 1)) >= 0,
                ]
            )
            for col_idx in range(self.nwu):
                for row_idx in range(self.nwu):
                    if not (col_idx == row_idx):
                        multiplier_constraints.append(Lambda[col_idx, row_idx] <= 0)

        # klmn = cp.Variable(
        #     shape=(
        #         self.nx + self.nu + self.nzu,
        #         self.nx + self.nwp + self.ny + self.nwu,
        #     )
        # )
        K = cp.Variable(shape=(self.nx,self.nx))
        L1 = cp.Variable(shape=(self.nx, self.nwp))
        L2 = cp.Variable(shape=(self.nx, self.ny))
        L3 = cp.Variable(shape=(self.nx, self.nwu))

        M1 = cp.Variable(shape=(self.nx,self.nx))
        N11 = cp.Variable(shape=(self.nx, self.nwp))
        N12 = cp.Variable(shape=(self.nx, self.ny))
        N13 = cp.Variable(shape=(self.nx,self.nwu))
        
        M2 = cp.Variable(shape=(self.nzp,self.nx))
        N21 = cp.Variable(shape=(self.nzp, self.nwp))
        N22 = cp.Variable(shape=(self.nzp, self.ny))
        N23 = cp.Variable(shape=(self.nzp,self.nwu))
        
        M3 = cp.Variable(shape=(self.nzu, self.nx))
        N31 = cp.Variable(shape=(self.nzu, self.nwp))
        N32 = cp.Variable(shape=(self.nzu, self.ny))
        N33 = np.zeros(shape=(self.nzu, self.nwu))

        klmn = cp.bmat(
            [
                [K, L1, L2, L3],
                [M1,N11,N12, N13],
                [M2,N21,N22, N23],
                [M3, N31, N32, N33]
            ]
        )


        if self.loop_trafo:
            P_21_1 = cp.bmat(
                [
                    [
                        self.A_lin.cpu() @ Y,
                        self.A_lin.cpu(),
                        self.B_lin.cpu(),
                        np.zeros(shape=(self.nx, self.nwu)),
                    ],
                    [
                        np.zeros(shape=(self.nx, self.nx)),
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
            if self.controller_feedback == 'cf':
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
            elif self.controller_feedback == 'nf':
                P_21_2 = cp.bmat(
                    [
                        [
                            np.zeros(shape=(self.nx, self.nx)),
                            np.zeros(shape=(self.nx, self.nu)),
                            np.zeros(shape=(self.nx, self.nzu)),
                        ],
                        [
                            np.eye(self.nx),
                            np.zeros(shape=(self.nx, self.nu)),
                            np.zeros(shape=(self.nx, self.nzu)),
                        ],
                        [
                            np.zeros(shape=(self.nzp, self.nx)),
                            -np.eye(self.nu),
                            np.zeros(shape=(self.nzp, self.nzu)),
                        ],
                        [
                            np.zeros(shape=(self.nzu, self.nx)),
                            np.zeros(shape=(self.nzu, self.nu)),
                            np.eye(self.nzu),
                        ],
                    ]
                )
            elif self.controller_feedback == 'signal':
                P_21_2 = cp.bmat(
                    [
                        [
                            np.zeros(shape=(self.nx, self.nx)),
                            np.eye(self.nx),
                            np.zeros(shape=(self.nx, self.nzp)),
                            np.zeros(shape=(self.nx, self.nzu)),
                        ],
                        [
                            np.eye(self.nx),
                            np.zeros(shape=(self.nx, self.nx)),
                            np.zeros(shape=(self.nx, self.nzp)),
                            np.zeros(shape=(self.nx, self.nzu)),
                        ],
                        [
                            np.zeros(shape=(self.nzp, self.nx)),
                            np.zeros(shape=(self.nzp, self.nx)),
                            np.eye(self.nzp),
                            np.zeros(shape=(self.nzp, self.nzu)),
                        ],
                        [
                            np.zeros(shape=(self.nzu, self.nx)),
                            np.zeros(shape=(self.nzu, self.nx)),
                            np.zeros(shape=(self.nzu, self.nzp)),
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

            P_21 = P_21_1 + P_21_2 @ klmn @ P_21_4
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
                        2*Lambda,
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
                        1/2*Lambda,
                    ],
                ]
            )
            P = cp.bmat([[P_11, P_21.T], [P_21, P_22]])

        else:
            M3_tilde = M3
            N31_tilde = N31
            N32_tilde = N32

            P_11 = cp.bmat([
                [-Y, -np.eye(self.nx), np.zeros(shape=(self.nx,self.nwp)), self.beta * N32_tilde.T],
                [-np.eye(self.nx), -X, np.zeros(shape=(self.nx,self.nwp)), self.beta * M3_tilde.T],
                [np.zeros(shape=(self.nwp, self.nx)), np.zeros(shape=(self.nwp, self.nx)),-self.gamma**2*np.eye(self.nwp), self.beta * N31_tilde.T ],
                [self.beta *N32_tilde, self.beta*M3_tilde, self.beta*N31_tilde, -(Lambda.T + Lambda)]
            ])

            P_21 = cp.bmat([
                [self.A_lin.cpu() @ Y + N12, self.A_lin.cpu() + M1, self.B_lin + N11, N13],
                [L2, X@self.A_lin.cpu()+K, X@self.B_lin.cpu()+L1, L3],
                [self.C_lin.cpu()@Y+N22, self.C_lin + M2, N21, N23]
            ])
            P_22 = cp.bmat([
                [-Y, -np.eye(self.nx), np.zeros(shape=(self.nx,self.nzp))],
                [-np.eye(self.nx), -X, np.zeros(shape=(self.nx,self.nzp))],
                [np.zeros(shape=(self.nzp, self.nx)), np.zeros(shape=(self.nzp,self.nx)), -np.eye(self.nzp)]
            ])
            P = cp.bmat([
                [P_11, P_21.T],
                [P_21, P_22]
            ])
        nP = P.shape[0]

        device = self.K.device

        if self.multiplier_type == 'diagonal':
            Lambda_0_torch = torch.diag(self.lam)
        elif self.multiplier_type == 'static_zf':
            Lambda_0_torch = self.lam

        X_0_torch, Y_0_torch, U_0_torch, V_0_torch = self.get_coupling_matrices()
        # Ts = (
        #     T.cpu().detach().numpy()
        #     for T in self.get_T(
        #         X_0_torch, Y_0_torch, U_0_torch, V_0_torch, Lambda_0_torch
        #     )
        # )
        # T_l, T_r, T_s = Ts

        A_0 = np.zeros(shape=(self.nx, self.nx))
        B1_0 = np.zeros(shape=(self.nx, self.nwp))
        B2_0 = np.zeros(shape=(self.nx, self.ny))
        B3_0 = np.zeros(shape=(self.nx, self.nwu))

        C1_0 = np.zeros(shape=(self.nu, self.nx))
        D11_0 = np.zeros(shape=(self.nu, self.nwp))
        D12_0 = np.zeros(shape=(self.nu, self.ny))
        D13_0 = np.zeros(shape=(self.nu, self.nwu))

        C2_0 = np.zeros(shape=(self.nzu, self.nx))
        D21_0 = np.zeros(shape=(self.nzu, self.nwp))
        D22_0 = np.zeros(shape=(self.nzu, self.ny))
        D23_0 = np.zeros(shape=(self.nzu, self.nwu))

        abcd_0 = np.concatenate(
            [
                np.concatenate([A_0, B1_0, B2_0, B3_0], axis=1),
                np.concatenate([C1_0, D11_0, D12_0, D13_0], axis=1),
                np.concatenate([C2_0, D21_0, D22_0, D23_0], axis=1),
            ],
            axis=0,
        )

        # klmn_0 = T_l @ abcd_0 @ T_r + T_s
        # klmn_0 = self.klmn.cpu().detach().numpy()
        klmn_0 = self.get_klmn().cpu().detach().numpy()

        feasibility_constraint = [
            P << -1e-3 * np.eye(nP),
            cp.bmat([[Y, np.eye(self.nx)], [np.eye(self.nx), X]])
            >> 1e-3 * np.eye(self.nx * 2),
            *multiplier_constraints,
        ]

        d = cp.Variable(shape=(1,))

        problem = cp.Problem(
            cp.Minimize(d),
            feasibility_constraint + utils.get_distance_constraints(klmn_0, klmn, d),
        )
        problem.solve(solver=self.optimizer)
        d_fixed = np.float64(d.value + 1e-1)

        logger.info(
            f'1. run: projection. '
            f'problem status {problem.status},'
            f'||Omega - Omega_0|| = {d.value}'
        )

        alpha = cp.Variable(shape=(1,))
        problem = cp.Problem(
            cp.Minimize(expr=alpha),
            feasibility_constraint
            + utils.get_distance_constraints(klmn_0, klmn, d_fixed)
            + utils.get_bounding_inequalities(X, Y, klmn, alpha),
        )
        problem.solve(solver=self.optimizer)
        logger.info(
            f'2. run: parameter bounds. '
            f'problem status {problem.status},'
            f'alpha_star = {alpha.value}'
        )

        alpha_fixed = np.float64(alpha.value + 1e-1)

        beta = cp.Variable(shape=(1,))
        problem = cp.Problem(
            cp.Maximize(expr=beta),
            feasibility_constraint
            + utils.get_conditioning_constraints(Y, X, beta)
            + utils.get_distance_constraints(klmn_0, klmn, d_fixed)
            + utils.get_bounding_inequalities(X, Y, klmn, alpha_fixed),
        )
        problem.solve(solver=self.optimizer)
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
            .double()
            .to(device)
        )
        self.L_y_flat.data = (
            torch.tensor(
                self.extract_vector_from_lower_triangular_matrix(
                    np.linalg.cholesky(np.array(Y.value))
                )
            )
            .double()
            .to(device)
        )
        if self.multiplier_type == 'diagonal':
            self.lam.data = (
                torch.tensor(np.diag(np.array(Lambda.value))).double().to(device)
            )
        elif self.multiplier_type == 'static_zf':
            self.lam.data = torch.tensor(Lambda.value).double().to(device)

        self.K.data = torch.tensor(klmn.value[:self.nx,:self.nx]).double().to(device)
        self.L1.data = torch.tensor(klmn.value[:self.nx,self.nx:self.nx+self.nwp]).double().to(device)
        self.L2.data = torch.tensor(klmn.value[:self.nx,self.nx+self.nwp:self.nx+self.nwp+self.ny]).double().to(device)
        self.L3.data = torch.tensor(klmn.value[:self.nx,self.nx+self.nwp+self.ny:]).double().to(device)

        self.M1.data = torch.tensor(klmn.value[self.nx:self.nx+self.nx,:self.nx]).double().to(device)
        self.N11.data = torch.tensor(klmn.value[self.nx:self.nx+self.nx, self.nx:self.nx+self.nwp]).double().to(device)
        self.N12.data = torch.tensor(klmn.value[self.nx:self.nx+self.nx, self.nx+self.nwp:self.nx+self.nwp+self.ny]).double().to(device)
        self.N13.data = torch.tensor(klmn.value[self.nx:self.nx+self.nx,self.nx+self.nwp+self.ny:]).double().to(device)

        self.M2.data = torch.tensor(klmn.value[self.nx+self.nx:self.nx+self.nx+self.nzp,:self.nx]).double().to(device)
        self.N21.data = torch.tensor(klmn.value[self.nx+self.nx:self.nx+self.nx+self.nzp, self.nx:self.nx+self.nwp]).double().to(device)
        self.N22.data = torch.tensor(klmn.value[self.nx+self.nx:self.nx+self.nx+self.nzp, self.nx+self.nwp:self.nx+self.nwp+self.ny]).double().to(device)
        self.N23.data = torch.tensor(klmn.value[self.nx+self.nx:self.nx+self.nx+self.nzp,self.nx+self.nwp+self.ny:]).double().to(device)

        self.M3.data = torch.tensor(klmn.value[self.nx+self.nu:,:self.nx]).double().to(device)
        self.N31.data = torch.tensor(klmn.value[self.nx+self.nu:,self.nx:self.nx+self.nwp]).double().to(device)
        self.N32.data = torch.tensor(klmn.value[self.nx+self.nu:,self.nx+self.nwp:self.nx+self.nwp+self.ny]).double().to(device)

        return np.float64(d.value)
    


    def write_parameters(self, params: List[torch.Tensor]) -> None:
        for old_par, new_par in zip(params, self.parameters()):
            new_par.data = old_par.clone()

    def get_regularization(self, decay_param: torch.Tensor) -> torch.Tensor:
        device = self.device
        # klmn_0 = self.klmn.to(device)
        klmn_0 = self.get_klmn()
        # klmn_0 = self.get_klmn().to(device)

        # construct X, Y, and Lambda
        X, Y, U, V = self.get_coupling_matrices()

        Lambda = torch.diag(input=self.lam).to(device)

        # transform to original parameters
        T_l, T_r, T_s = self.get_T(X, Y, U, V, Lambda)
        abcd_0 = (
            torch.linalg.inv(T_l).double().to(device)
            @ (klmn_0 - T_s.double().to(device))
            @ torch.linalg.inv(T_r).double().to(device)
        )

        _, _, _, B3, _, _, _, D13, C2, D21, D22 = self.get_controller_matrices(abcd_0)

        # par_norm = (torch.linalg.norm(B3)+torch.linalg.norm(D13)+torch.linalg.norm(C2)+torch.linalg.norm(D21)+torch.linalg.norm(D22))
        par_norm = (
            torch.linalg.norm(C2) + torch.linalg.norm(D21) + torch.linalg.norm(D22)
        )

        return decay_param / torch.max(torch.tensor(1e-5), par_norm)


class InputLinearizationNonConvexRnn(ConstrainedForwardModule):
    def __init__(        
        self,
        A_lin: NDArray[np.float64],
        B_lin: NDArray[np.float64],
        C_lin: NDArray[np.float64],
        nd: int,
        alpha: float,
        beta: float,
        nwu: int,
        nzu: int,
        gamma: float,
        init_pars: SimAbcdParameter,
        nonlinearity: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device = torch.device('cpu'),
        multiplier_type: Optional[str] = 'diag',
    ) -> None:
        super().__init__()
        self.nx = A_lin.shape[0]  # state size
        self.ny = self.nx  # output size of linearization
        self.nh = B_lin.shape[1]  # input size of performance channel
        self.nd = nd
        self.ne = C_lin.shape[0]  # output size of performance channel
        self.nr = self.nx + self.ne # output of controller is residual of state and performance output
        self.nz = nzu  # output size of uncertainty channel
        self.nw = nwu  # input size of uncertainty channel
        self.multiplier_type = multiplier_type

        self.alpha = alpha
        self.beta = beta

        self.device = device

        self.A_lin = torch.tensor(A_lin, dtype=torch.float64).to(device)
        self.B_lin = torch.tensor(B_lin, dtype=torch.float64).to(device)
        self.C_lin = torch.tensor(C_lin, dtype=torch.float64).to(device)

        self.nl = nonlinearity

        self.gamma = gamma

        self.Lam = torch.nn.Parameter(
            torch.tensor(init_pars.Lambda).to(device)
        )

        self.theta = torch.nn.Parameter(
            torch.tensor(init_pars.theta).to(device)
        )

        self.X_cal = torch.nn.Parameter(
            torch.tensor(init_pars.X_cal).to(device)
        )   
        self.S_s = torch.from_numpy(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            A_lin,
                            np.zeros(shape=(self.nx, self.nx)),
                            np.zeros(shape=(self.nx, self.nd)),
                            np.zeros(shape=(self.nx, self.nw)),
                        ],
                        axis=1,
                    ),
                    np.zeros(
                        shape=(self.nx, self.nx + self.nx + self.nd + self.nw)
                    ),
                    np.concatenate(
                        [
                            C_lin,
                            np.zeros(shape=(self.ne, self.nx)),
                            np.zeros(shape=(self.ne, self.nd)),
                            np.zeros(shape=(self.ne, self.nw)),
                        ],
                        axis=1,
                    ),
                    np.zeros(
                        shape=(self.nz, self.nx + self.nx + self.nd + self.nw)
                    ),
                ],
                axis=0,
                dtype=np.float64,
            )
        ).to(device)
        self.S_l = torch.from_numpy(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nx, self.nx)),
                            B_lin,
                            np.eye(self.nx),
                            np.zeros(shape=(self.nx,self.ne)),
                            np.zeros(shape=(self.nx, self.nz))
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.eye(self.nx),
                            np.zeros(shape=(self.nx, self.nh+self.nr+self.nz)),
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.ne, self.nx+self.nh+self.nx)),
                            np.eye(self.ne),
                            np.zeros(shape=(self.ne, self.nz))

                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nz,self.nx+self.nh+self.nr)),
                            np.eye(self.nz)
                        ], axis=1
                    )
                ], axis=0
            )
        ).double().to(device)
        self.S_r = torch.from_numpy(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nx, self.nx)),
                            np.eye(self.nx),
                            np.zeros(shape=(self.nx, self.nd+self.nw))
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.eye(self.ny),
                            np.zeros(shape=(self.ny, self.nx+self.nd+self.nw))
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nd, self.nx+self.nx)),
                            np.eye(self.nd),
                            np.zeros(shape=(self.nd,self.nw))
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nw, self.nx+self.nx+self.nd)),
                            np.eye(self.nw)
                        ], axis=1
                    )
                ],axis=0
            )
        ).double().to(device)
   

    def set_lure_system(self) -> Tuple[SimAbcdParameter, NDArray[np.float64]]:
        generalized_plant = self.S_s + self.S_l @ self.theta @ self.S_r

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
        ) = utils.get_cal_matrices(
            generalized_plant,
            self.nx+self.nx,
            self.nd,
            self.ne,
            self.nz
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
            Delta=self.nl,
            device=self.device,
        ).to(self.device)

        sim_parameter = SimAbcdParameter(
            self.theta.cpu().detach().numpy(),
            self.X_cal.cpu().detach().numpy(),
            self.Lam.cpu().detach().numpy()
        )

        return (sim_parameter, generalized_plant.cpu().detach().numpy())


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

    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, N, nu = x_pred.shape
        assert self.lure._nu == nu
        # assert hx is not None
        if hx is None:
            x0_lin = torch.zeros(size=(n_batch,self.nx)).to(self.device)
            x0_rnn = torch.zeros(size=(n_batch,self.nx)).to(self.device)
        else:
            x0_lin, x0_rnn = hx
        x0 = torch.concat((x0_lin, x0_rnn), dim=1).reshape(
            shape=(n_batch, self.nx * 2, 1)
        )
        us = x_pred.reshape(shape=(n_batch, N, nu, 1))
        y, x = self.lure.forward(x0=x0, us=us, return_states=True)

        return y.reshape(n_batch, N, self.lure._ny), (
            x[:, : self.nx].reshape(n_batch, self.nx),
            x[:, self.nx :].reshape(n_batch, self.nx),
        )

    def get_barriers(self, t: torch.Tensor) -> torch.Tensor:
        multiplier_constraints = []
        if self.multiplier_type == 'diagonal':
            multiplier_constraints.append(self.Lam)
        elif self.multiplier_type == 'static_zf':
            multiplier_constraints.extend(
                list(
                    torch.squeeze(
                        torch.ones(size=(self.nw, 1)).double().to(self.device).T
                        @ self.Lam
                    )
                )
            ),
            multiplier_constraints.extend(
                list(
                    torch.squeeze(
                        self.Lam
                        @ torch.ones(size=(self.nw, 1)).double().to(self.device)
                    )
                )
            )
            for col_idx in range(self.nw):
                for row_idx in range(self.nw):
                    if not (row_idx == col_idx):
                        multiplier_constraints.append(-self.Lam[col_idx, row_idx])

        constraints = [
            -self.get_constraints(),
            *multiplier_constraints,
            self.X_cal
        ]

        barrier = torch.tensor(0.0).to(self.device)
        for constraint in constraints:
            barrier += -t * utils.get_logdet(constraint).to(self.device)

        return barrier

    def get_constraints(self) -> torch.Tensor:

        generalized_plant = self.S_s + self.S_l @ self.theta @ self.S_r

        nxi = self.nx+self.nx
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
        ) = utils.get_cal_matrices(
            generalized_plant,
            nxi,
            self.nd,
            self.ne,
            self.nz
        )

        L1 = torch.concat(
            [
                torch.concat([torch.eye(nxi), torch.zeros((nxi,self.nd)), torch.zeros((nxi,self.nw))], dim=1),
                torch.concat([A_cal, B1_cal, B2_cal], dim=1)
            ], dim=0
        )
        L2 = torch.concat(
            [
                torch.concat([torch.zeros((self.nd,nxi)), torch.eye(self.nd), torch.zeros((self.nd,self.nw))], dim=1),
                torch.concat([C1_cal, D11_cal, D12_cal], dim=1)
            ], dim=0
        )
        L3 = torch.concat(
            [
                torch.concat([torch.zeros((self.nw,nxi)), torch.zeros((self.nw,self.nd)), torch.eye(self.nw)], dim=1),
                torch.concat([C2_cal, D21_cal, D22_cal], dim=1)
            ], dim=0
        )

        stab_cond = torch.concat(
            [
                torch.concat([-self.X_cal, torch.zeros((nxi,nxi))],dim=1),
                torch.concat([torch.zeros(nxi,nxi), self.X_cal],dim=1)
            ], dim=0
        )
        perf_cond = torch.concat(
            [
                torch.concat([-self.gamma**2*torch.eye(self.nd), torch.zeros((self.nd,self.ne))], dim=1),
                torch.concat([torch.zeros((self.ne,self.nd)), torch.eye(self.ne)],dim=1)
            ],dim=0
        )
        uncertain_cond = torch.concat(
            [
                torch.concat([-(self.Lam+self.Lam.T),self.beta*self.Lam],dim=1),
                torch.concat([self.beta*self.Lam, torch.zeros((self.nz, self.nw))],dim=1)
            ], dim=0
        )
        P = L1.T @ stab_cond @ L1 + L2.T @ perf_cond @ L2 + L3.T @ uncertain_cond @ L3

        if self.multiplier_type == 'diagonal':
            # https://yalmip.github.io/faq/semidefiniteelementwise/
            # symmetrize variable
            return 0.5 * (P + P.T)
        else:
            return P
    
    def write_parameters(self, params: List[torch.Tensor]) -> None:
        for old_par, new_par in zip(params, self.parameters()):
            new_par.data = old_par.clone()


    def check_constraints(self) -> bool:
        with torch.no_grad():
            P = self.get_constraints()
            _, info = torch.linalg.cholesky_ex(-P)
        return True if info == 0 else False


class InputLinearizationRnn(ConstrainedForwardModule):
    def __init__(        
        self,
        A_lin: NDArray[np.float64],
        B_lin: NDArray[np.float64],
        C_lin: NDArray[np.float64],
        nd: int,
        alpha: float,
        beta: float,
        nwu: int,
        nzu: int,
        gamma: float,
        nonlinearity: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device = torch.device('cpu'),
        optimizer: str = cp.SCS,
        multiplier_type: Optional[str] = 'diag',
        init_omega: Optional[str]='zero',
    ) -> None:
        super().__init__()
        self.nx = A_lin.shape[0]  # state size
        self.ny = self.nx  # output size of linearization
        self.nh = B_lin.shape[1]  # input size of performance channel
        self.nd = nd
        self.ne = C_lin.shape[0]  # output size of performance channel
        self.nr = self.nx + self.ne # output of controller is residual of state and performance output
        self.nz = nzu  # output size of uncertainty channel
        self.nw = nwu  # input size of uncertainty channel
        self.optimizer = optimizer
        self.multiplier_type = multiplier_type
        self.init_omega = init_omega

        self.alpha = alpha
        self.beta = beta

        self.device = device

        self.A_lin = torch.tensor(A_lin, dtype=torch.float64).to(device)
        self.B_lin = torch.tensor(B_lin, dtype=torch.float64).to(device)
        self.C_lin = torch.tensor(C_lin, dtype=torch.float64).to(device)

        self.nl = nonlinearity

        self.gamma = gamma

        self.Lam = torch.nn.Parameter(torch.eye(self.nz).double().to(device))
            
        if self.init_omega == 'zero':
            self.Omega_tilde = torch.nn.Parameter(
                torch.zeros(
                    size=(
                        self.nx + self.nh + self.nr + self.nz,
                        self.nx + self.ny + self.nd + self.nw,
                    )
                )
            ).to(device)
        elif self.init_omega == 'rand':
            self.Omega_tilde = torch.nn.Parameter(
                torch.normal(0,1/self.nx, size=(
                    self.nx + self.nh + self.nr + self.nz,
                    self.nx + self.ny + self.nd + self.nw,
                )).double().to(device)
            )
        else:
            raise ValueError(f'Initialization method {self.init_omega} is not supported.')

        L_flat_size = utils.extract_vector_from_lower_triangular_matrix(
            torch.zeros(size=(self.nx, self.nx))
        ).shape[0]
        self.L_x_flat = torch.nn.Parameter(
                torch.normal(0, 1 / self.nx, size=(L_flat_size,)).double().to(device)
            )
        self.L_y_flat = torch.nn.Parameter(
            torch.normal(0, 1 / self.nx, size=(L_flat_size,)).double().to(device)
        )

        self.S_s = torch.from_numpy(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            A_lin,
                            np.zeros(shape=(self.nx, self.nx)),
                            np.zeros(shape=(self.nx, self.nd)),
                            np.zeros(shape=(self.nx, self.nw)),
                        ],
                        axis=1,
                    ),
                    np.zeros(
                        shape=(self.nx, self.nx + self.nx + self.nd + self.nw)
                    ),
                    np.concatenate(
                        [
                            C_lin,
                            np.zeros(shape=(self.ne, self.nx)),
                            np.zeros(shape=(self.ne, self.nd)),
                            np.zeros(shape=(self.ne, self.nw)),
                        ],
                        axis=1,
                    ),
                    np.zeros(
                        shape=(self.nz, self.nx + self.nx + self.nd + self.nw)
                    ),
                ],
                axis=0,
                dtype=np.float64,
            )
        ).to(device)
        self.S_l = torch.from_numpy(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nx, self.nx)),
                            B_lin,
                            np.eye(self.nx),
                            np.zeros(shape=(self.nx,self.ne)),
                            np.zeros(shape=(self.nx, self.nz))
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.eye(self.nx),
                            np.zeros(shape=(self.nx, self.nh+self.nr+self.nz)),
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.ne, self.nx+self.nh+self.nx)),
                            np.eye(self.ne),
                            np.zeros(shape=(self.ne, self.nz))

                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nz,self.nx+self.nh+self.nr)),
                            np.eye(self.nz)
                        ], axis=1
                    )
                ], axis=0
            )
        ).double().to(device)
        self.S_r = torch.from_numpy(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nx, self.nx)),
                            np.eye(self.nx),
                            np.zeros(shape=(self.nx, self.nd+self.nw))
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.eye(self.ny),
                            np.zeros(shape=(self.ny, self.nx+self.nd+self.nw))
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nd, self.nx+self.nx)),
                            np.eye(self.nd),
                            np.zeros(shape=(self.nd,self.nw))
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nw, self.nx+self.nx+self.nd)),
                            np.eye(self.nw)
                        ], axis=1
                    )
                ],axis=0
            )
        ).double().to(device)

    def set_lure_system(self) -> Tuple[SimAbcdParameter, NDArray[np.float64]]:
        device = self.device
        Lambda = self.Lam

        L = torch.concat(
            [
                torch.concat(
                    [
                        torch.eye(self.nx),
                        torch.zeros((self.nx,self.nh+self.nr+self.nz))
                    ], dim=1
                ),
                torch.concat(
                    [
                        torch.zeros(self.nh, self.nx),
                        torch.eye(self.nh),
                        torch.zeros(self.nh, self.nr+self.nz)
                    ], dim=1
                ),
                torch.concat(
                    [
                        torch.zeros(self.nr,self.nx+self.nh),
                        torch.eye(self.nr),
                        torch.zeros(self.nr, self.nz),
                    ], dim=1
                ),
                torch.concat(
                    [
                        torch.zeros(self.nz, self.nx+self.nh+self.nr),
                        torch.linalg.inv(Lambda)
                    ], dim=1
                )
            ],dim=0
        )

        # transform from Omega_tilde (optimization parameters) to Omega
        Omega = L @ self.Omega_tilde

        X,Y,U,V = utils.get_coupling_matrices(
            self.L_x_flat,
            self.L_y_flat,
            self.nx
        )

        T_l,T_r,T_s = self.get_T(X,Y,U,V,Lambda)

        theta = (
            torch.linalg.inv(T_l).double().to(device)
            @ (Omega - T_s.double().to(device))
            @ torch.linalg.inv(T_r).double().to(device)
        )

        generalized_plant = self.S_s + self.S_l @ theta @ self.S_r

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
        ) = utils.get_cal_matrices(
            generalized_plant,
            self.nx+self.nx,
            self.nd,
            self.ne,
            self.nz
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
            Delta=self.nl,
            device=self.device,
        ).to(device)

        X_cal = torch.concat(
            [
                torch.concat([X,U],dim=1),
                torch.concat([U,-V@torch.linalg.inv(Y)@U],dim=1)
            ]
        )

        sim_parameter = SimAbcdParameter(
            theta.cpu().detach().numpy(),
            X_cal.cpu().detach().numpy(),
            Lambda.cpu().detach().numpy()
        )

        return (sim_parameter, generalized_plant.cpu().detach().numpy())


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
                        X@self.B_lin,
                        X,
                        torch.zeros((self.nx, self.ne)).to(self.device),
                        torch.zeros((self.nx, self.nz)).to(self.device)
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros((self.nh, self.nx)),
                        torch.eye(self.nh),
                        torch.zeros((self.nh, self.nr+self.nz)),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros((self.nr, self.nx+self.nh)),
                        torch.eye(self.nr),
                        torch.zeros((self.nr,self.nz))
                    ], dim=1
                ),
                torch.concat(
                    [
                        torch.zeros((self.nz, self.nx)),
                        torch.zeros((self.nz, self.nh)),
                        torch.zeros((self.nz, self.nr)),
                        Lambda,
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        ).double().to(self.device)
        T_r = torch.concat(
            [
                torch.concat(
                    [
                        torch.zeros(size=(self.nx, self.nx)).to(self.device),
                        V.T,
                        torch.zeros(size=(self.nx, self.nd)).to(self.device),
                        torch.zeros(size=(self.nx, self.nw)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.eye(self.nx).to(self.device),
                        Y,
                        torch.zeros(size=(self.nx, self.nd)).to(self.device),
                        torch.zeros(size=(self.nx, self.nw)).to(self.device),
                    ],
                    dim=1,
                ).to(self.device),
                torch.concat(
                    [
                        torch.zeros((self.nd,self.nx)).to(self.device),
                        torch.zeros(size=(self.nd, self.ny)).to(self.device),
                        torch.eye(self.nd),
                        torch.zeros(size=(self.nd, self.nw)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros(size=(self.nw, self.nx)),
                        torch.zeros(size=(self.nw, self.ny)),
                        torch.zeros(size=(self.nw, self.nd)),
                        torch.eye(self.nw),
                    ],
                    dim=1,
                ).to(self.device),
            ],
            dim=0,
        ).double()
        T_s = torch.concat(
            [
                torch.concat(
                    [
                        torch.zeros(size=(self.nx, self.nx)).to(self.device),
                        X @ self.A_lin @ Y,
                        torch.zeros(size=(self.nx, self.nd+self.nw)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.zeros(size=(self.nh, self.nx + self.ny + self.nd + self.nw)),
                torch.zeros(size=(self.nr, self.nx + self.ny + self.nd + self.nw)),
                torch.zeros(size=(self.nz, self.nx + self.ny + self.nd + self.nw))
            ],
            dim=0,
        ).double().to(self.device)

        return (T_l, T_r, T_s)


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

    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, N, nu = x_pred.shape
        assert self.lure._nu == nu
        # assert hx is not None
        if hx is None:
            x0_lin = torch.zeros(size=(n_batch,self.nx)).to(self.device)
            x0_rnn = torch.zeros(size=(n_batch,self.nx)).to(self.device)
        else:
            x0_lin, x0_rnn = hx
        x0 = torch.concat((x0_lin, x0_rnn), dim=1).reshape(
            shape=(n_batch, self.nx * 2, 1)
        )
        us = x_pred.reshape(shape=(n_batch, N, nu, 1))
        y, x = self.lure.forward(x0=x0, us=us, return_states=True)

        return y.reshape(n_batch, N, self.lure._ny), (
            x[:, : self.nx].reshape(n_batch, self.nx),
            x[:, self.nx :].reshape(n_batch, self.nx),
        )

    def get_barriers(self, t: torch.Tensor) -> torch.Tensor:
        L_x = utils.construct_lower_triangular_matrix(
            L_flat=self.L_x_flat, diag_length=self.nx
        ).to(self.device)
        L_y = utils.construct_lower_triangular_matrix(
            L_flat=self.L_y_flat, diag_length=self.nx
        ).to(self.device)

        X = L_x @ L_x.T
        Y = L_y @ L_y.T

        multiplier_constraints = []
        if self.multiplier_type == 'diagonal':
            multiplier_constraints.append(self.Lam)
        elif self.multiplier_type == 'static_zf':
            multiplier_constraints.extend(
                list(
                    torch.squeeze(
                        torch.ones(size=(self.nw, 1)).double().to(self.device).T
                        @ self.Lam
                    )
                )
            ),
            multiplier_constraints.extend(
                list(
                    torch.squeeze(
                        self.Lam
                        @ torch.ones(size=(self.nw, 1)).double().to(self.device)
                    )
                )
            )
            for col_idx in range(self.nw):
                for row_idx in range(self.nw):
                    if not (row_idx == col_idx):
                        multiplier_constraints.append(-self.Lam[col_idx, row_idx])

        constraints = [
            -self.get_constraints(),
            *multiplier_constraints,
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
            barrier += -t * utils.get_logdet(constraint).to(self.device)

        return barrier

    def get_constraints(self) -> torch.Tensor:
        L_x = utils.construct_lower_triangular_matrix(
            L_flat=self.L_x_flat, diag_length=self.nx
        ).to(self.device)
        L_y = utils.construct_lower_triangular_matrix(
            L_flat=self.L_y_flat, diag_length=self.nx
        ).to(self.device)
        X = L_x @ L_x.T
        Y = L_y @ L_y.T
        # U = torch.linalg.inv(Y) - X
        # V = Y
        A_lin = self.A_lin
        B_lin = self.B_lin
        C_lin = self.C_lin

        Lambda = self.Lam

        P_21_1 = torch.concat(
            [
                torch.concat(
                    [
                        A_lin @ Y,
                        A_lin,
                        torch.zeros(size=(self.nx, self.nd)).to(self.device),
                        torch.zeros(size=(self.nx, self.nw)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros(size=(self.nx, self.nx)).to(self.device),
                        X @ A_lin,
                        torch.zeros(size=(self.nx, self.nd)).to(self.device),
                        torch.zeros(size=(self.nx, self.nw)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        C_lin @ Y,
                        C_lin,
                        torch.zeros(size=(self.ne, self.nd)).to(self.device),
                        torch.zeros(size=(self.ne, self.nw)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros(size=(self.nz, self.nx)).to(self.device),
                        torch.zeros(size=(self.nz, self.nx)).to(self.device),
                        torch.zeros(size=(self.nz, self.nd)).to(self.device),
                        torch.zeros(size=(self.nz, self.nw)).to(self.device),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        ).double().to(self.device)
        
        P_21_2 = self.S_l
        P_21_4 = self.S_r

        P_21 = P_21_1 + P_21_2 @ self.Omega_tilde @ P_21_4

        # internal state size
        nxi = self.nx+self.nx
        (
            A_bf,
            B1_bf,
            B2_bf,
            C1_bf,
            D11_bf,
            D12_bf,
            C2_bf_tilde,
            D21_bf_tilde,
            D22_bf_tilde,
        ) = utils.get_cal_matrices(
            P_21,
            nxi,
            self.nd,
            self.ne,
            self.nz
        )
        

        X_bf = torch.concat([
                torch.concat([Y, torch.eye(self.nx)], dim=1),
                torch.concat([torch.eye(self.nx), X], dim=1),
        ], dim=0)

        P_11 = torch.concat(
            [
                torch.concat(
                    [
                        -X_bf, torch.zeros(size=(nxi, self.nd)), self.beta*C2_bf_tilde.T
                    ], dim=1
                ),
                torch.concat(
                    [
                        torch.zeros(size=(self.nd,nxi)), -self.gamma**2*torch.eye(self.nd), self.beta*D21_bf_tilde.T
                    ], dim=1
                ),
                torch.concat(
                    [
                        self.beta*C2_bf_tilde, self.beta*D21_bf_tilde, -(Lambda.T+Lambda)
                    ], dim=1
                )
            ], dim=0
        )

        P_21 = torch.concat(
            [
                torch.concat([A_bf, B1_bf, B2_bf], dim=1),
                torch.concat([C1_bf, D11_bf, D12_bf], dim=1),
            ], dim=0
        )

        P_22 = torch.concat(
            [
                torch.concat([-X_bf, torch.zeros(size=(nxi, self.ne))],dim=1),
                torch.concat([torch.zeros(size=(self.ne, nxi)), -torch.eye(self.ne)],dim=1)
            ], dim=0
        )
        P = torch.concat(
            [
                torch.concat([P_11, P_21.T], dim=1), 
                torch.concat([P_21, P_22], dim=1)
            ],dim=0,
        ).to(self.device)

        if self.multiplier_type == 'diagonal':
            # https://yalmip.github.io/faq/semidefiniteelementwise/
            # symmetrize variable
            return 0.5 * (P + P.T)
        else:
            return P

    def project_parameters(self, write_parameter: bool = True) -> np.float64:
        if self.check_constraints():
            logger.info('No projection necessary, constraints are satisfied.')
            return np.float64(0.0)
        X = cp.Variable(shape=(self.nx, self.nx), symmetric=True)
        Y = cp.Variable(shape=(self.nx, self.nx), symmetric=True)

        multiplier_constraints = []
        logger.info(f'Multiplier type: {self.multiplier_type}')
        if self.multiplier_type == 'diagonal':
            # diagonal multiplier, elements need to be positive
            lam = cp.Variable(shape=(self.nz, 1))
            Lambda = cp.diag(lam)

        elif self.multiplier_type == 'static_zf':
            # static zames falb multiplier, Lambda must be double hyperdominant
            Lambda = cp.Variable(shape=(self.nz, self.nw))
            multiplier_constraints.extend(
                [
                    np.ones(shape=(self.nw, 1)).T @ Lambda >= 0,
                    Lambda @ np.ones(shape=(self.nw, 1)) >= 0,
                ]
            )
            for col_idx in range(self.nw):
                for row_idx in range(self.nw):
                    if not (col_idx == row_idx):
                        multiplier_constraints.append(Lambda[col_idx, row_idx] <= 0)

        Omega_tilde = cp.Variable(
            shape=(
                self.nx + self.nh + self.nr + self.nz,
                self.nx + self.ny + self.nd + self.nw,
            )
        )

        A_lin = self.A_lin.detach().numpy()
        B_lin = self.B_lin.detach().numpy()
        C_lin = self.C_lin.detach().numpy()
        P_21_1 = cp.bmat(
            [
                [
                    A_lin @ Y,
                    A_lin,
                    np.zeros(shape=(self.nx, self.nd)),
                    np.zeros(shape=(self.nx, self.nw)),
                ],
                [
                    np.zeros(shape=(self.nx, self.nx)),
                    X @ A_lin,
                    np.zeros(shape=(self.nx, self.nd)),
                    np.zeros(shape=(self.nx, self.nw)),          
                ],
                [
                    C_lin @ Y,
                    C_lin,
                    np.zeros(shape=(self.ne, self.nd)),
                    np.zeros(shape=(self.ne, self.nw)),
                ],
                [
                    np.zeros(shape=(self.nz, self.nx)),
                    np.zeros(shape=(self.nz, self.nx)),
                    np.zeros(shape=(self.nz, self.nd)),
                    np.zeros(shape=(self.nz, self.nw)),
                ]
            ]
        )
        
        P_21_2 = cp.bmat(
            [
                [
                    np.zeros(shape=(self.nx, self.nx)),
                    B_lin,
                    np.eye(self.nx),
                    np.zeros(shape=(self.nx,self.ne)),
                    np.zeros(shape=(self.nx, self.nz))
                ],
                [
                    np.eye(self.nx),
                    np.zeros(shape=(self.nx, self.nh+self.nr+self.nz)),
                ],
                [
                    np.zeros(shape=(self.ne, self.nx+self.nh+self.nx)),
                    np.eye(self.ne),
                    np.zeros(shape=(self.ne, self.nz))
                ],
                [
                    np.zeros(shape=(self.nz,self.nx+self.nh+self.nr)),
                    np.eye(self.nz)
                ]
            ]
        )
        
        P_21_4 = cp.bmat(
            [
                [
                    np.zeros(shape=(self.nx, self.nx)),
                    np.eye(self.nx),
                    np.zeros(shape=(self.nx, self.nd+self.nw))
                ],
                [
                    np.eye(self.ny),
                    np.zeros(shape=(self.ny, self.nx+self.nd+self.nw))
                ],
                [
                    np.zeros(shape=(self.nd, self.nx+self.nx)),
                    np.eye(self.nd),
                    np.zeros(shape=(self.nd,self.nw))
                ],
                [
                    np.zeros(shape=(self.nw, self.nx+self.nx+self.nd)),
                    np.eye(self.nw)
                ]
            ]
        )

        gen_plant = P_21_1 + P_21_2 @ Omega_tilde @ P_21_4

        nxi = self.nx+self.nx
        
        A_bf = gen_plant[: nxi, : nxi]
        B1_bf = gen_plant[:nxi, nxi:nxi+self.nd]
        B2_bf = gen_plant[:nxi, nxi+self.nd:]

        C1_bf = gen_plant[nxi:nxi+self.ne, : nxi]
        D11_bf = gen_plant[nxi:nxi+self.ne, nxi:nxi+self.nd]
        D12_bf = gen_plant[nxi:nxi+self.ne, nxi+self.nd:]

        C2_bf_tilde = gen_plant[nxi+self.ne:, : nxi]
        D21_bf_tilde = gen_plant[nxi+self.ne:, nxi:nxi+self.nd]
        
        X_bf = cp.bmat([
                [Y, np.eye(self.nx)],
                [np.eye(self.nx), X],
        ])

        P_11 = cp.bmat(
            [
                [-X_bf, torch.zeros(size=(nxi, self.nd)), self.beta*C2_bf_tilde.T],
                [torch.zeros(size=(self.nd,nxi)), -self.gamma**2*torch.eye(self.nd), self.beta*D21_bf_tilde.T],
                [self.beta*C2_bf_tilde, self.beta*D21_bf_tilde, -(Lambda.T+Lambda)]
            ]
        )

        P_21 = cp.bmat(
            [
                [A_bf, B1_bf, B2_bf],
                [C1_bf, D11_bf, D12_bf],
            ]
        )

        P_22 = cp.bmat(
            [
                [-X_bf, torch.zeros(size=(nxi, self.ne))],
                [torch.zeros(size=(self.ne, nxi)), -torch.eye(self.ne)]
            ]
        )
        P = cp.bmat(
            [
                [P_11, P_21.T], 
                [P_21, P_22],
            ]
        )       

        nP = P.shape[0]

        device = self.Omega_tilde.device

        Omega_tilde_0 = self.Omega_tilde.cpu().detach().numpy()

        feasibility_constraint = [
            P << -1e-3 * np.eye(nP),
            cp.bmat([[Y, np.eye(self.nx)], [np.eye(self.nx), X]])
            >> 1e-3 * np.eye(self.nx * 2),
            *multiplier_constraints,
        ]

        d = cp.Variable(shape=(1,))

        problem = cp.Problem(
            cp.Minimize(d),
            feasibility_constraint + utils.get_distance_constraints(Omega_tilde_0, Omega_tilde, d),
        )
        problem.solve(solver=self.optimizer)
        d_fixed = np.float64(d.value + 1)

        logger.info(
            f'1. run: projection. '
            f'problem status {problem.status},'
            f'||Omega - Omega_0|| = {d.value}'
        )

        alpha = cp.Variable(shape=(1,))
        problem = cp.Problem(
            cp.Minimize(expr=alpha),
            feasibility_constraint
            + utils.get_distance_constraints(Omega_tilde_0, Omega_tilde, d_fixed)
            + utils.get_bounding_inequalities(X, Y, Omega_tilde, alpha),
        )
        problem.solve(solver=self.optimizer)
        logger.info(
            f'2. run: parameter bounds. '
            f'problem status {problem.status},'
            f'alpha_star = {alpha.value}'
        )

        alpha_fixed = np.float64(alpha.value + 1)

        beta = cp.Variable(shape=(1,))
        problem = cp.Problem(
            cp.Maximize(expr=beta),
            feasibility_constraint
            + utils.get_conditioning_constraints(Y, X, beta)
            + utils.get_distance_constraints(Omega_tilde_0, Omega_tilde, d_fixed)
            + utils.get_bounding_inequalities(X, Y, Omega_tilde, alpha_fixed),
        )
        problem.solve(solver=self.optimizer)
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
                utils.extract_vector_from_lower_triangular_matrix(
                    np.linalg.cholesky(np.array(X.value))
                )
            )
            .double()
            .to(device)
        )
        self.L_y_flat.data = (
            torch.tensor(
                utils.extract_vector_from_lower_triangular_matrix(
                    np.linalg.cholesky(np.array(Y.value))
                )
            )
            .double()
            .to(device)
        )

        self.Lam.data = torch.tensor(Lambda.value).double().to(device)
        self.Omega_tilde.data = torch.tensor(Omega_tilde.value).double().to(device)

        return np.float64(d.value)
    
    def write_parameters(self, params: List[torch.Tensor]) -> None:
        for old_par, new_par in zip(params, self.parameters()):
            new_par.data = old_par.clone()


    def check_constraints(self) -> bool:
        with torch.no_grad():
            P = self.get_constraints()
            _, info = torch.linalg.cholesky_ex(-P)
        return True if info == 0 else False
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
    

class InputLinearizationRnn2(ConstrainedForwardModule):
    def __init__(        
        self,
        A_lin: NDArray[np.float64],
        B_lin: NDArray[np.float64],
        C_lin: NDArray[np.float64],
        D_lin: NDArray[np.float64],
        B_lin_2: NDArray[np.float64],
        D_lin_2: NDArray[np.float64],
        alpha: float,
        beta: float,
        nwu: int,
        gamma: float,
        nonlinearity: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device = torch.device('cpu'),
        optimizer: str = cp.SCS,
        multiplier_type: Optional[str] = 'diag',
        init_omega: Optional[str]='zero',
    ) -> None:
        super().__init__()
        self.nx = A_lin.shape[0]  # state size
        self.nx_rnn = self.nx # controller has same state size
        self.nd = B_lin.shape[1]  # input size of performance channel
        self.ny = self.nx + self.nd  # output size of linearization
        self.ne = C_lin.shape[0]  # output size of performance channel
        self.nu = B_lin_2.shape[1] # output size of controller
        self.nw = nwu  # input size of uncertainty channel
        self.nz = self.nw # output size of uncertainty channel
        
        self.optimizer = optimizer
        self.multiplier_type = multiplier_type
        self.init_omega = init_omega

        self.alpha = alpha
        self.beta = beta

        self.device = device

        self.A_lin = torch.tensor(A_lin, dtype=torch.float64).to(device)
        self.B_lin = torch.tensor(B_lin, dtype=torch.float64).to(device)
        self.C_lin = torch.tensor(C_lin, dtype=torch.float64).to(device)
        self.D_lin = torch.tensor(D_lin, dtype=torch.float64).to(device)
        self.B_lin_2 = torch.tensor(B_lin_2, dtype=torch.float64).to(device)
        self.D_lin_2 = torch.tensor(D_lin_2, dtype=torch.float64).to(device)

        self.nl = nonlinearity

        self.gamma = gamma

        if self.multiplier_type == 'diagonal':
            self.lam = torch.nn.Parameter(
                torch.ones(size=(self.nz,)).double().to(device)
            )
        elif self.multiplier_type == 'static_zf':
            self.lam = torch.nn.Parameter(torch.eye(self.nz).double().to(device))
        else:
            raise ValueError(f'Multiplier type {self.multiplier_type} not supported.')

            
        if self.init_omega == 'zero':
            self.Omega_tilde = torch.nn.Parameter(
                torch.zeros(
                    size=(
                        self.nx + self.nu + self.nz,
                        self.nx + self.ny + self.nw,
                    )
                )
            ).to(device)
        elif self.init_omega == 'rand':
            self.Omega_tilde = torch.nn.Parameter(
                torch.normal(0,1/self.nx, size=(
                    self.nx + self.nu + self.nz,
                    self.nx + self.ny + self.nw,
                )).double().to(device)
            )
        else:
            raise ValueError(f'Initialization method {self.init_omega} is not supported.')

        L_flat_size = utils.extract_vector_from_lower_triangular_matrix(
            torch.zeros(size=(self.nx, self.nx))
        ).shape[0]
        self.L_x_flat = torch.nn.Parameter(
                torch.normal(0, 1 / self.nx, size=(L_flat_size,)).double().to(device)
            )
        self.L_y_flat = torch.nn.Parameter(
            torch.normal(0, 1 / self.nx, size=(L_flat_size,)).double().to(device)
        )



        self.S_s = torch.from_numpy(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            A_lin,
                            np.zeros(shape=(self.nx, self.nx_rnn)),
                            B_lin,
                            np.zeros(shape=(self.nx, self.nw)),
                        ],
                        axis=1,
                    ),
                    np.zeros(
                        shape=(self.nx_rnn, self.nx + self.nx_rnn + self.nd + self.nw)
                    ),
                    np.concatenate(
                        [
                            C_lin,
                            np.zeros(shape=(self.ne, self.nx_rnn)),
                            D_lin,
                            np.zeros(shape=(self.ne, self.nw)),
                        ],
                        axis=1,
                    ),
                    np.zeros(
                        shape=(self.nz, self.nx + self.nx_rnn + self.nd + self.nw)
                    ),
                ],
                axis=0,
                dtype=np.float64,
            )
        ).to(device)
        self.S_l = torch.from_numpy(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nx, self.nx_rnn)),
                            B_lin_2,
                            np.zeros(shape=(self.nx,self.nz))
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.eye(self.nx_rnn),
                            np.zeros(shape=(self.nx_rnn, self.nu + self.nz)),
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.ne, self.nx_rnn)),
                            D_lin_2,
                            np.zeros(shape=(self.ne, self.nz))

                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nz,self.nx_rnn + self.nu)),
                            np.eye(self.nz)
                        ], axis=1
                    )
                ], axis=0
            )
        ).double().to(device)
        self.S_r = torch.from_numpy(
            np.concatenate(
                [
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nx_rnn, self.nx)),
                            np.eye(self.nx_rnn),
                            np.zeros(shape=(self.nx, self.nd+self.nw))
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.vstack((np.eye(self.nx), np.zeros((self.nd,self.nx)))),
                            np.zeros(shape=(self.ny, self.nx_rnn)),
                            np.vstack((np.zeros((self.nx,self.nd)), np.eye(self.nd))),
                            np.zeros(shape=(self.ny, self.nw))
                        ], axis=1
                    ),
                    np.concatenate(
                        [
                            np.zeros(shape=(self.nw, self.nx+self.nx_rnn+self.nd)),
                            np.eye(self.nw),
                        ], axis=1
                    ),
                ],axis=0
            )
        ).double().to(device)

    def set_lure_system(self) -> Tuple[SimAbcdParameter, NDArray[np.float64]]:
        device = self.device
        if self.multiplier_type == 'diagonal':
            Lambda = torch.diag(self.lam).to(self.device)
        elif self.multiplier_type == 'static_zf':
            Lambda = self.lam.to(self.device)

        L = torch.concat(
            [
                torch.concat(
                    [
                        torch.eye(self.nx_rnn),
                        torch.zeros((self.nx,self.nu+self.nz))
                    ], dim=1
                ),
                torch.concat(
                    [
                        torch.zeros(self.nu, self.nx_rnn),
                        torch.eye(self.nu),
                        torch.zeros(self.nu, self.nz)
                    ], dim=1
                ),
                torch.concat(
                    [
                        torch.zeros(self.nz, self.nx+self.nu),
                        torch.linalg.inv(Lambda)
                    ], dim=1
                )
            ],dim=0
        )

        # transform from Omega_tilde (optimization parameters) to Omega
        Omega = L @ self.Omega_tilde

        X,Y,U,V = utils.get_coupling_matrices(
            self.L_x_flat,
            self.L_y_flat,
            self.nx
        )

        T_l,T_r,T_s = self.get_T(X,Y,U,V,Lambda)

        theta = (
            torch.linalg.inv(T_l).double().to(device)
            @ (Omega - T_s.double().to(device))
            @ torch.linalg.inv(T_r).double().to(device)
        )

        generalized_plant = self.S_s + self.S_l @ theta @ self.S_r

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
        ) = utils.get_cal_matrices(
            generalized_plant,
            self.nx+self.nx_rnn,
            self.nd,
            self.ne,
            self.nz
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
            Delta=self.nl,
            device=self.device,
        ).to(device)

        X_cal = torch.concat(
            [
                torch.concat([X,U],dim=1),
                torch.concat([U,-torch.linalg.inv(V)@Y@U],dim=1)
            ]
        )

        sim_parameter = SimAbcdParameter(
            theta.cpu().detach().numpy(),
            X_cal.cpu().detach().numpy(),
            Lambda.cpu().detach().numpy()
        )

        return (sim_parameter, generalized_plant.cpu().detach().numpy())


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
                        X@self.B_lin_2,
                        torch.zeros((self.nx_rnn, self.nz)).to(self.device)
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros((self.nu, self.nx_rnn)),
                        torch.eye(self.nu),
                        torch.zeros((self.nu, self.nz)),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros((self.nz, self.nx_rnn+self.nu)),
                        torch.eye(self.nz),
                    ], dim=1
                ),
            ],
            dim=0,
        ).double().to(self.device)
        T_r = torch.concat(
            [
                torch.concat(
                    [
                        torch.zeros(size=(self.nx_rnn, self.nx_rnn)).to(self.device),
                        torch.hstack((V.T,torch.zeros((self.nx_rnn,self.nd)).to(self.device))),
                        torch.zeros(size=(self.nx_rnn, self.nw)).to(self.device)
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.vstack((torch.eye(self.nx_rnn), torch.zeros((self.nd, self.nx_rnn)))).to(self.device),
                        torch.concat([
                            torch.concat([Y, torch.zeros((self.nx_rnn,self.nd)).to(self.device)],dim=1),
                            torch.concat([torch.zeros((self.nd, self.nx_rnn)), torch.eye(self.nd)],dim=1),
                        ],dim=0),
                        torch.zeros(size=(self.ny, self.nw)).to(self.device)
                    ],
                    dim=1,
                ).to(self.device),
                torch.concat(
                    [
                        torch.zeros(size=(self.nw, self.nx)),
                        torch.zeros(size=(self.nw, self.ny)),
                        torch.eye(self.nw),
                    ],
                    dim=1,
                ).to(self.device),
            ],
            dim=0,
        ).double()
        T_s = torch.concat(
            [
                torch.concat(
                    [
                        torch.zeros(size=(self.nx_rnn, self.nx_rnn)).to(self.device),
                        torch.hstack((X @ self.A_lin @ Y,torch.zeros((self.nx_rnn,self.nd)))),
                        torch.zeros(size=(self.nx_rnn, self.nw)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.zeros(size=(self.nu, self.nx_rnn + self.ny + self.nw)),
                torch.zeros(size=(self.nz, self.nx_rnn + self.ny + self.nw))
            ],
            dim=0,
        ).double().to(self.device)

        return (T_l, T_r, T_s)


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

    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, N, nu = x_pred.shape
        assert self.lure._nu == nu
        # assert hx is not None
        if hx is None:
            x0_lin = torch.zeros(size=(n_batch,self.nx)).to(self.device)
            x0_rnn = torch.zeros(size=(n_batch,self.nx)).to(self.device)
        else:
            x0_lin, x0_rnn = hx
        x0 = torch.concat((x0_lin, x0_rnn), dim=1).reshape(
            shape=(n_batch, self.nx * 2, 1)
        )
        us = x_pred.reshape(shape=(n_batch, N, nu, 1))
        y, x = self.lure.forward(x0=x0, us=us, return_states=True)

        return y.reshape(n_batch, N, self.lure._ny), (
            x[:, : self.nx].reshape(n_batch, self.nx),
            x[:, self.nx :].reshape(n_batch, self.nx),
        )

    def get_barriers(self, t: torch.Tensor) -> torch.Tensor:
        L_x = utils.construct_lower_triangular_matrix(
            L_flat=self.L_x_flat, diag_length=self.nx
        ).to(self.device)
        L_y = utils.construct_lower_triangular_matrix(
            L_flat=self.L_y_flat, diag_length=self.nx
        ).to(self.device)

        X = L_x @ L_x.T
        Y = L_y @ L_y.T

        multiplier_constraints = []
        if self.multiplier_type == 'diagonal':
            multiplier_constraints.append(torch.diag(torch.squeeze(self.lam)))
        elif self.multiplier_type == 'static_zf':
            multiplier_constraints.extend(
                list(
                    torch.squeeze(
                        torch.ones(size=(self.nw, 1)).double().to(self.device).T
                        @ self.lam
                    )
                )
            ),
            multiplier_constraints.extend(
                list(
                    torch.squeeze(
                        self.lam
                        @ torch.ones(size=(self.nw, 1)).double().to(self.device)
                    )
                )
            )
            for col_idx in range(self.nw):
                for row_idx in range(self.nw):
                    if not (row_idx == col_idx):
                        multiplier_constraints.append(-self.lam[col_idx, row_idx])

        constraints = [
            -self.get_constraints(),
            *multiplier_constraints,
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
            barrier += -t * utils.get_logdet(constraint).to(self.device)

        return barrier

    def get_constraints(self) -> torch.Tensor:
        L_x = utils.construct_lower_triangular_matrix(
            L_flat=self.L_x_flat, diag_length=self.nx
        ).to(self.device)
        L_y = utils.construct_lower_triangular_matrix(
            L_flat=self.L_y_flat, diag_length=self.nx
        ).to(self.device)
        X = L_x @ L_x.T
        Y = L_y @ L_y.T
        # U = torch.linalg.inv(Y) - X
        # V = Y
        A_lin = self.A_lin
        B_lin = self.B_lin
        C_lin = self.C_lin
        D_lin = self.D_lin

        if self.multiplier_type == 'diagonal':
            Lambda = torch.diag(self.lam).to(self.device)
        elif self.multiplier_type == 'static_zf':
            Lambda = self.lam.to(self.device)

        P_21_1 = torch.concat(
            [
                torch.concat(
                    [
                        A_lin @ Y,
                        A_lin,
                        B_lin,
                        torch.zeros(size=(self.nx, self.nw)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros(size=(self.nx, self.nx)).to(self.device),
                        X @ A_lin,
                        X @ B_lin,
                        torch.zeros(size=(self.nx, self.nw)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        C_lin @ Y,
                        C_lin,
                        D_lin,
                        torch.zeros(size=(self.ne, self.nw)).to(self.device),
                    ],
                    dim=1,
                ),
                torch.concat(
                    [
                        torch.zeros(size=(self.nz, self.nx)).to(self.device),
                        torch.zeros(size=(self.nz, self.nx)).to(self.device),
                        torch.zeros(size=(self.nz, self.nd)).to(self.device),
                        torch.zeros(size=(self.nz, self.nw)).to(self.device),
                    ],
                    dim=1,
                ),
            ],
            dim=0,
        ).double().to(self.device)
        
        P_21_2 = self.S_l
        P_21_4 = self.S_r

        P_21 = P_21_1 + P_21_2 @ self.Omega_tilde @ P_21_4

        # internal state size
        nxi = self.nx+self.nx
        (
            A_bf,
            B1_bf,
            B2_bf,
            C1_bf,
            D11_bf,
            D12_bf,
            C2_bf_tilde,
            D21_bf_tilde,
            D22_bf_tilde,
        ) = utils.get_cal_matrices(
            P_21,
            nxi,
            self.nd,
            self.ne,
            self.nz
        )
        

        X_bf = torch.concat([
                torch.concat([Y, torch.eye(self.nx)], dim=1),
                torch.concat([torch.eye(self.nx), X], dim=1),
        ], dim=0)

        M_11 = torch.concat(
            [
                torch.concat(
                    [
                        -X_bf, torch.zeros(size=(nxi, self.nd)), self.beta*C2_bf_tilde.T
                    ], dim=1
                ),
                torch.concat(
                    [
                        torch.zeros(size=(self.nd,nxi)), -self.gamma**2*torch.eye(self.nd), self.beta*D21_bf_tilde.T
                    ], dim=1
                ),
                torch.concat(
                    [
                        self.beta*C2_bf_tilde, self.beta*D21_bf_tilde, -(Lambda.T+Lambda)
                    ], dim=1
                )
            ], dim=0
        )

        M_21 = torch.concat(
            [
                torch.concat([A_bf, B1_bf, B2_bf], dim=1),
                torch.concat([C1_bf, D11_bf, D12_bf], dim=1),
            ], dim=0
        )

        M_22 = torch.concat(
            [
                torch.concat([-X_bf, torch.zeros(size=(nxi, self.ne))],dim=1),
                torch.concat([torch.zeros(size=(self.ne, nxi)), -torch.eye(self.ne)],dim=1)
            ], dim=0
        )
        M = torch.concat(
            [
                torch.concat([M_11, M_21.T], dim=1), 
                torch.concat([M_21, M_22], dim=1)
            ],dim=0,
        ).to(self.device)

        if self.multiplier_type == 'diagonal':
            # https://yalmip.github.io/faq/semidefiniteelementwise/
            # symmetrize variable
            return 0.5 * (M + M.T)
        else:
            return M

    def project_parameters(self, write_parameter: bool = True) -> np.float64:
        if self.check_constraints():
            logger.info('No projection necessary, constraints are satisfied.')
            return np.float64(0.0)
        X = cp.Variable(shape=(self.nx, self.nx), symmetric=True)
        Y = cp.Variable(shape=(self.nx_rnn, self.nx_rnn), symmetric=True)

        multiplier_constraints = []
        logger.info(f'Multiplier type: {self.multiplier_type}')
        if self.multiplier_type == 'diagonal':
            # diagonal multiplier, elements need to be positive
            lam = cp.Variable(shape=(self.nz, 1))
            Lambda = cp.diag(lam)

        elif self.multiplier_type == 'static_zf':
            # static zames falb multiplier, Lambda must be double hyperdominant
            Lambda = cp.Variable(shape=(self.nz, self.nw))
            multiplier_constraints.extend(
                [
                    np.ones(shape=(self.nw, 1)).T @ Lambda >= 0,
                    Lambda @ np.ones(shape=(self.nw, 1)) >= 0,
                ]
            )
            for col_idx in range(self.nw):
                for row_idx in range(self.nw):
                    if not (col_idx == row_idx):
                        multiplier_constraints.append(Lambda[col_idx, row_idx] <= 0)

        Omega_tilde = cp.Variable(
            shape=(
                self.nx + self.nu + self.nz,
                self.nx + self.ny + self.nw,
            )
        )

        A_lin = self.A_lin.detach().numpy()
        B_lin = self.B_lin.detach().numpy()
        C_lin = self.C_lin.detach().numpy()
        D_lin = self.D_lin.detach().numpy()
        
        B_lin_2 = self.B_lin_2.detach().numpy()
        D_lin_2 = self.D_lin_2.detach().numpy()

        P_21_1 = cp.bmat(
            [
                [
                    A_lin @ Y,
                    A_lin,
                    B_lin,
                    np.zeros(shape=(self.nx, self.nw)),
                ],
                [
                    np.zeros(shape=(self.nx_rnn, self.nx)),
                    X @ A_lin,
                    X @ B_lin,
                    np.zeros(shape=(self.nx_rnn, self.nw)),          
                ],
                [
                    C_lin @ Y,
                    C_lin,
                    D_lin,
                    np.zeros(shape=(self.ne, self.nw)),
                ],
                [
                    np.zeros(shape=(self.nz, self.nx)),
                    np.zeros(shape=(self.nz, self.nx_rnn)),
                    np.zeros(shape=(self.nz, self.nd)),
                    np.zeros(shape=(self.nz, self.nw)),
                ]
            ]
        )
        
        P_21_2 = cp.bmat(
            [
                [
                    np.zeros(shape=(self.nx, self.nx)),
                    B_lin_2,
                    np.zeros(shape=(self.nx, self.nz))
                ],
                [
                    np.eye(self.nx_rnn),
                    np.zeros(shape=(self.nx_rnn, self.nu+self.nz)),
                ],
                [
                    np.zeros(shape=(self.ne, self.nx)),
                    D_lin_2,
                    np.zeros(shape=(self.ne, self.nz))
                ],
                [
                    np.zeros(shape=(self.nz,self.nx+self.nu)),
                    np.eye(self.nz)
                ]
            ]
        )
        
        P_21_4 = cp.bmat(
            [
                [
                    np.zeros(shape=(self.nx_rnn, self.nx)),
                    np.eye(self.nx_rnn),
                    np.zeros(shape=(self.nx, self.nd+self.nw))
                ],
                [
                    np.vstack((np.eye(self.nx), np.zeros((self.nd,self.nx)))),
                    np.zeros(shape=(self.ny, self.nx_rnn)),
                    np.vstack((np.zeros((self.nx,self.nd)), np.eye(self.nd))),
                    np.zeros(shape=(self.ny, self.nw))
                ],
                [
                    np.zeros(shape=(self.nw, self.nx+self.nx_rnn+self.nd)),
                    np.eye(self.nw),
                ]
            ]
        )

        gen_plant = P_21_1 + P_21_2 @ Omega_tilde @ P_21_4

        nxi = self.nx+self.nx
        
        A_bf = gen_plant[: nxi, : nxi]
        B1_bf = gen_plant[:nxi, nxi:nxi+self.nd]
        B2_bf = gen_plant[:nxi, nxi+self.nd:]

        C1_bf = gen_plant[nxi:nxi+self.ne, : nxi]
        D11_bf = gen_plant[nxi:nxi+self.ne, nxi:nxi+self.nd]
        D12_bf = gen_plant[nxi:nxi+self.ne, nxi+self.nd:]

        C2_bf_tilde = gen_plant[nxi+self.ne:, : nxi]
        D21_bf_tilde = gen_plant[nxi+self.ne:, nxi:nxi+self.nd]
        
        X_bf = cp.bmat([
                [Y, np.eye(self.nx)],
                [np.eye(self.nx), X],
        ])

        P_11 = cp.bmat(
            [
                [-X_bf, torch.zeros(size=(nxi, self.nd)), self.beta*C2_bf_tilde.T],
                [torch.zeros(size=(self.nd,nxi)), -self.gamma**2*torch.eye(self.nd), self.beta*D21_bf_tilde.T],
                [self.beta*C2_bf_tilde, self.beta*D21_bf_tilde, -(Lambda.T+Lambda)]
            ]
        )

        P_21 = cp.bmat(
            [
                [A_bf, B1_bf, B2_bf],
                [C1_bf, D11_bf, D12_bf],
            ]
        )

        P_22 = cp.bmat(
            [
                [-X_bf, torch.zeros(size=(nxi, self.ne))],
                [torch.zeros(size=(self.ne, nxi)), -torch.eye(self.ne)]
            ]
        )
        P = cp.bmat(
            [
                [P_11, P_21.T], 
                [P_21, P_22],
            ]
        )       

        nP = P.shape[0]

        device = self.Omega_tilde.device

        Omega_tilde_0 = self.Omega_tilde.cpu().detach().numpy()

        feasibility_constraint = [
            P << -1e-3 * np.eye(nP),
            cp.bmat([[Y, np.eye(self.nx)], [np.eye(self.nx), X]])
            >> 1e-3 * np.eye(self.nx * 2),
            *multiplier_constraints,
        ]

        d = cp.Variable(shape=(1,))

        problem = cp.Problem(
            cp.Minimize(d),
            feasibility_constraint + utils.get_distance_constraints(Omega_tilde_0, Omega_tilde, d),
        )
        problem.solve(solver=self.optimizer)
        d_fixed = np.float64(d.value + 1)

        logger.info(
            f'1. run: projection. '
            f'problem status {problem.status},'
            f'||Omega - Omega_0|| = {d.value}'
        )

        alpha = cp.Variable(shape=(1,))
        problem = cp.Problem(
            cp.Minimize(expr=alpha),
            feasibility_constraint
            + utils.get_distance_constraints(Omega_tilde_0, Omega_tilde, d_fixed)
            + utils.get_bounding_inequalities(X, Y, Omega_tilde, alpha),
        )
        problem.solve(solver=self.optimizer)
        logger.info(
            f'2. run: parameter bounds. '
            f'problem status {problem.status},'
            f'alpha_star = {alpha.value}'
        )

        alpha_fixed = np.float64(alpha.value + 1)

        beta = cp.Variable(shape=(1,))
        problem = cp.Problem(
            cp.Maximize(expr=beta),
            feasibility_constraint
            + utils.get_conditioning_constraints(Y, X, beta)
            + utils.get_distance_constraints(Omega_tilde_0, Omega_tilde, d_fixed)
            + utils.get_bounding_inequalities(X, Y, Omega_tilde, alpha_fixed),
        )
        problem.solve(solver=self.optimizer)
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
                utils.extract_vector_from_lower_triangular_matrix(
                    np.linalg.cholesky(np.array(X.value))
                )
            )
            .double()
            .to(device)
        )
        self.L_y_flat.data = (
            torch.tensor(
                utils.extract_vector_from_lower_triangular_matrix(
                    np.linalg.cholesky(np.array(Y.value))
                )
            )
            .double()
            .to(device)
        )

        if self.multiplier_type == 'diagonal':
            self.lam.data = (
                torch.tensor(np.diag(np.array(Lambda.value))).double().to(device)
            )
        elif self.multiplier_type == 'static_zf':
            self.lam.data = torch.tensor(Lambda.value).double().to(device)
        self.Omega_tilde.data = torch.tensor(Omega_tilde.value).double().to(device)

        return np.float64(d.value)
    
    def write_parameters(self, params: List[torch.Tensor]) -> None:
        for old_par, new_par in zip(params, self.parameters()):
            new_par.data = old_par.clone()


    def check_constraints(self) -> bool:
        with torch.no_grad():
            P = self.get_constraints()
            _, info = torch.linalg.cholesky_ex(-P)
        return True if info == 0 else False
