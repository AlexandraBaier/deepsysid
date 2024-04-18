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
from scipy.linalg import block_diag

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


class BasicLSTMDoubleLinearOutput(HiddenStateForwardModule):
    def __init__(
        self,
        input_dim: int,
        recurrent_dim: int,
        num_recurrent_layers: int,
        output_dim: int,
        dropout: float,
        C: torch.Tensor,
        bias: bool = True,
    ):
        super().__init__()

        self.num_recurrent_layers = num_recurrent_layers
        self.recurrent_dim = recurrent_dim
        self.C = C
        self.ne = self.C.shape[0]
        self.nx = output_dim

        with warnings.catch_warnings():
            self.predictor_lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=recurrent_dim,
                num_layers=num_recurrent_layers,
                dropout=dropout,
                batch_first=True,
                bias=bias,
            )

        self.out = nn.Linear(
            in_features=recurrent_dim, out_features=output_dim, bias=bias
        )
        nn.init.xavier_normal_(self.out.weight)

        for name, param in self.predictor_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)
            

    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, N, _ = x_pred.shape
        x, (h0, c0) = self.predictor_lstm(x_pred, hx)
        x = self.out(x).reshape(n_batch,N,self.nx,1)
        y = self.C @ x

        return y.reshape(n_batch,N,self.ne), (x.reshape(n_batch,N,self.nx), h0)

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

    def _detach_matrices(self) -> bool:
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
        nx: int,
        nd: int,
        ne: int,
        alpha: float,
        beta: float,
        nw: int,
        nonlinearity: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device = torch.device('cpu'),
        optimizer: str = cp.SCS,
        multiplier_type: Optional[str] = 'diag',
        init_omega: Optional[str]='zero',
        coupling_flat: Optional[bool] = True,
        increase_constraints: Optional[np.float64] = 1.2
    ) -> None:
        super().__init__()
        self.nx = nx  # state size
        self.nx_rnn = self.nx # controller has same state size
        self.nd = nd  # input size of performance channel
        self.ny = self.nx + self.nd  # output size of linearization
        self.ne = ne  # output size of performance channel
        self.nu = self.nx + self.nx + self.ne # output size of controller
        self.nw = nw
        self.nz = self.nw
        
        self.optimizer = optimizer
        self.multiplier_type = multiplier_type
        self.init_omega = init_omega
        self.coupling_flat = coupling_flat
        self.increase_constraints = increase_constraints

        self.alpha = alpha
        self.beta = beta

        self.device = device

        self.nl = nonlinearity


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

        if self.coupling_flat:
            L_flat_size = utils.extract_vector_from_lower_triangular_matrix(
                np.zeros((self.nx, self.nx))
            ).shape[0]
            self.L_x_flat = torch.nn.Parameter(
                    torch.normal(0, 1 / self.nx, size=(L_flat_size,)).double().to(device)
                )
            self.L_y_flat = torch.nn.Parameter(
                torch.normal(0, 1 / self.nx, size=(L_flat_size,)).double().to(device)
            )
        else:
            self.X = torch.nn.Parameter(
                torch.normal(0, 1 / self.nx, size=(self.nx,self.nx)).double().to(device)
            )
            self.Y = torch.nn.Parameter(
                torch.normal(0, 1 / self.nx, size=(self.nx,self.nx)).double().to(device)
            )


    def set_lft_transformation_matrices(
        self,
        A_lin: NDArray[np.float64],
        B_lin: NDArray[np.float64],
        C_lin: NDArray[np.float64],
        D_lin: NDArray[np.float64],
        B_lin_2: NDArray[np.float64],
        D_lin_2: NDArray[np.float64],
        gamma: np.float64
    ) -> None:
        if gamma < 1:
            self.gamma = 1.0
        else:
            self.gamma = gamma * self.increase_constraints
        
        self.A_lin = torch.tensor(A_lin, dtype=torch.float64).to(self.device)
        self.B_lin = torch.tensor(B_lin, dtype=torch.float64).to(self.device)
        self.C_lin = torch.tensor(C_lin, dtype=torch.float64).to(self.device)
        self.D_lin = torch.tensor(D_lin, dtype=torch.float64).to(self.device)
        self.B_lin_2 = torch.tensor(B_lin_2, dtype=torch.float64).to(self.device)
        self.D_lin_2 = torch.tensor(D_lin_2, dtype=torch.float64).to(self.device)
        
        self.S_s = torch.from_numpy(
            utils.bmat([
                [A_lin, np.zeros((self.nx, self.nx_rnn)), B_lin, np.zeros((self.nx, self.nw))],
                [np.zeros((self.nx_rnn, self.nx + self.nx_rnn + self.nd + self.nw))],
                [C_lin, np.zeros((self.ne, self.nx_rnn)), D_lin, np.zeros((self.ne, self.nw))],
                [np.zeros((self.nz, self.nx + self.nx_rnn + self.nd + self.nw))]
            ])
        ).to(self.device)

        self.S_l = torch.from_numpy(
            utils.bmat([
                [np.zeros((self.nx, self.nx_rnn)), B_lin_2, np.zeros((self.nx,self.nz))],
                [np.eye(self.nx_rnn), np.zeros((self.nx_rnn, self.nu + self.nz))],
                [np.zeros((self.ne, self.nx_rnn)), D_lin_2, np.zeros((self.ne, self.nz))],
                [np.zeros((self.nz,self.nx_rnn + self.nu)), np.eye(self.nz)]
            ])
        ).double().to(self.device)

        self.S_r = torch.from_numpy(
            utils.bmat([
                [np.eye(self.nx_rnn), np.zeros((self.nx,self.nx_rnn+self.nd+self.nw))],
                [
                    np.zeros((self.ny,self.nx)),
                    np.vstack((np.eye(self.nx_rnn),np.zeros((self.nd,self.nx_rnn)))), 
                    np.vstack((np.zeros((self.nx_rnn,self.nd)),np.eye(self.nd))), 
                    np.zeros((self.ny,self.nw))
                ],
                [np.zeros((self.nw,self.nx_rnn+self.nx+self.nd)),np.eye(self.nw)]
            ])
        ).double().to(self.device)
        

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

        X,Y,U,V = self.get_coupling_matrices()

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


    def get_coupling_matrices(
            self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
            if self.coupling_flat:
                L_x = utils.construct_lower_triangular_matrix(
                    L_flat=self.L_x_flat, diag_length=self.nx
                )
                L_y = utils.construct_lower_triangular_matrix(
                    L_flat=self.L_y_flat, diag_length=self.nx
                )

                X = L_x @ L_x.T
                Y = L_y @ L_y.T

            else:
                X = self.X
                Y = self.Y

            # 2. Determine non-singular U,V with V U^T = I - Y X
            U = torch.linalg.inv(Y) - X
            V = Y

            return (X, Y, U, V)

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
        T_r = utils.torch_bmat([
            [V.T, torch.zeros((self.nx_rnn,self.ny)), torch.zeros((self.nx_rnn,self.nw))],
            [torch.vstack((V.T, torch.zeros((self.nd, self.nx_rnn)))), torch.eye(self.ny), torch.zeros((self.ny, self.nw))],
            [torch.zeros((self.nw, self.nx_rnn)), torch.zeros((self.nw,self.ny)), torch.eye(self.nw)]
        ]).double().to(self.device)
        T_s = utils.torch_bmat([
            [X @ self.A_lin @ Y, torch.zeros((self.nx_rnn, self.ny+self.nw))],
            [torch.zeros((self.nu, self.nx_rnn+self.ny+self.nw))],
            [torch.zeros((self.nz, self.nx_rnn+self.ny+self.nw))]
        ]).double().to(self.device)
        # T_r = torch.concat(
        #     [
        #         torch.concat(
        #             [
        #                 torch.zeros(size=(self.nx_rnn, self.nx_rnn)).to(self.device),
        #                 torch.hstack((V.T,torch.zeros((self.nx_rnn,self.nd)).to(self.device))),
        #                 torch.zeros(size=(self.nx_rnn, self.nw)).to(self.device)
        #             ],
        #             dim=1,
        #         ),
        #         torch.concat(
        #             [
        #                 torch.vstack((torch.eye(self.nx_rnn), torch.zeros((self.nd, self.nx_rnn)))).to(self.device),
        #                 torch.concat([
        #                     torch.concat([Y, torch.zeros((self.nx_rnn,self.nd)).to(self.device)],dim=1),
        #                     torch.concat([torch.zeros((self.nd, self.nx_rnn)), torch.eye(self.nd)],dim=1),
        #                 ],dim=0),
        #                 torch.zeros(size=(self.ny, self.nw)).to(self.device)
        #             ],
        #             dim=1,
        #         ).to(self.device),
        #         torch.concat(
        #             [
        #                 torch.zeros(size=(self.nw, self.nx)),
        #                 torch.zeros(size=(self.nw, self.ny)),
        #                 torch.eye(self.nw),
        #             ],
        #             dim=1,
        #         ).to(self.device),
        #     ],
        #     dim=0,
        # ).double()
        # T_s = torch.concat(
        #     [
        #         torch.concat(
        #             [
        #                 torch.zeros(size=(self.nx_rnn, self.nx_rnn)).to(self.device),
        #                 torch.hstack((X @ self.A_lin @ Y,torch.zeros((self.nx_rnn,self.nd)))),
        #                 torch.zeros(size=(self.nx_rnn, self.nw)).to(self.device),
        #             ],
        #             dim=1,
        #         ),
        #         torch.zeros(size=(self.nu, self.nx_rnn + self.ny + self.nw)),
        #         torch.zeros(size=(self.nz, self.nx_rnn + self.ny + self.nw))
        #     ],
        #     dim=0,
        # ).double().to(self.device)

        return (T_l, T_r, T_s)


    def initialize_parameters(self) -> None:
        Omega_tilde, X,Y,Lambda = self.get_initial_parameters()
        U = np.linalg.inv(Y) - X
        V = Y

        assert(np.linalg.norm(Y @ X + V @ U.T - np.eye(self.nx))< 1e-10)

        if self.coupling_flat:
            self.L_x_flat.data = (
                torch.tensor(
                    utils.extract_vector_from_lower_triangular_matrix(
                        np.linalg.cholesky(np.array(X))
                    )
                )
                .double()
                .to(self.device)
            )
            self.L_y_flat.data = (
                torch.tensor(
                    utils.extract_vector_from_lower_triangular_matrix(
                        np.linalg.cholesky(np.array(Y))
                    )
                )
                .double()
                .to(self.device)
            )

        else:
            self.X.data = torch.tensor(X).double().to(self.device)
            self.Y.data = torch.tensor(Y).double().to(self.device)


        if self.multiplier_type == 'diagonal':
            self.lam.data = (
                torch.tensor(np.diag(np.array(Lambda))).double().to(self.device)
            )
        elif self.multiplier_type == 'static_zf':
            self.lam.data = torch.tensor(Lambda).double().to(self.device)
        self.Omega_tilde.data = torch.tensor(Omega_tilde).double().to(self.device)


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
        nx = self.nx
        nu = self.nu
        nz = self.nz
        ny = self.ny
        nw = self.nw
        ne = self.ne
        nd = self.nd
        nxi = nx+ nx
        theta = np.zeros((nx+nu+nz, nx+ny+nw))
        gen_plant = self.S_s.detach().numpy() + self.S_l.detach().numpy() @ theta @ self.S_r.detach().numpy()

        A_cal = gen_plant[: nxi, : nxi]
        B_cal = gen_plant[:nxi, nxi:nxi+nd]
        B2_cal = gen_plant[:nxi, nxi+nd:]

        C_cal = gen_plant[nxi:nxi+ne, : nxi]
        D_cal = gen_plant[nxi:nxi+ne, nxi:nxi+nd]
        D12_cal = gen_plant[nxi:nxi+ne, nxi+nd:]

        C2_cal = gen_plant[nxi+ne:, : nxi]
        D21_cal = gen_plant[nxi+ne:, nxi:nxi+nd]
        D22_cal = gen_plant[nxi+ne:, nxi+nd:]

        L1 = utils.bmat([
            [np.eye(nxi), np.zeros((nxi,nd+nw))],
            [A_cal, B_cal, B2_cal],
        ])
        L2 = utils.bmat([
            [np.zeros((nd,nxi)), np.eye(nd), np.zeros((nd,nw))],
            [C_cal, D_cal, D12_cal],
        ])
        L3 = utils.bmat([
            [np.zeros((nw,nxi+nd)), np.eye(nw)],
            [C2_cal, D21_cal, D22_cal]
        ])

        X = cp.Variable((nx,nx))
        U = cp.Variable((nx,nx))
        X_hat = cp.Variable((nx,nx))
        X_cal = cp.bmat([
            [X, U],
            [U.T, X_hat]
        ])

        multiplier_constraints = []
        if self.multiplier_type == 'diagonal':
            # diagonal multiplier, elements need to be positive
            lam = cp.Variable(shape=(nz, 1))
            for lam_el in lam:
                multiplier_constraints.append(lam_el >= 0)
            Lambda = cp.diag(lam)

        elif self.multiplier_type == 'static_zf':
            # static zames falb multiplier, Lambda must be double hyperdominant
            Lambda = cp.Variable(shape=(nz, nw))
            multiplier_constraints.extend(
                [
                    np.ones(shape=(nw, 1)).T @ Lambda >= 0,
                    Lambda @ np.ones(shape=(nw, 1)) >= 0,
                ]
            )
            for col_idx in range(nw):
                for row_idx in range(nw):
                    if not (col_idx == row_idx):
                        multiplier_constraints.append(Lambda[col_idx, row_idx] <= 0)
        ga = self.gamma * 1.0

        M_theta = L1.T @ cp.bmat([[-X_cal, np.zeros((nxi,nxi))], [np.zeros((nxi,nxi)), X_cal]]) @ L1  \
        + L2.T @ cp.bmat([[-ga**2 * np.eye(nd), np.zeros((nd,ne))], [np.zeros((ne,nd)), np.eye(ne)]])@L2 \
        + L3.T @ cp.bmat([[-(Lambda + Lambda.T), self.beta*Lambda], [self.beta*Lambda.T, np.zeros((nw,nz))]]) @ L3

        eps = 1e-3
        nM = M_theta.shape[0]
        constraints = [
            M_theta << -eps*np.eye(nM), 
            X_cal >> eps*np.eye(nxi),
            *multiplier_constraints
        ]
        problem = cp.Problem(
            cp.Minimize([None]),
            constraints
        )
        problem.solve(solver=self.optimizer, verbose = False)
        if not problem.status == 'optimal':
            raise ValueError(f'Optimizer did not find a solution: {problem.status}')

        logger.info(
            f'Optimizing for |theta| = 0, status: {problem.status} \n'
            f'Max real eig (M_theta): {max(np.real(np.linalg.eig(M_theta.value)[0]))}'
        )
        
        # extract coupling matrices from optimization result and transform to Omega_tilde parameters
        X = X.value
        # U = U.value
        Lambda = Lambda.value
        Y = np.linalg.inv(X_cal.value)[:nx, :nx]
        U = np.linalg.inv(Y) - X
        V = Y

        T_l, T_r, T_s = [T.detach().numpy() for T in self.get_T(
            torch.tensor(X),
            torch.tensor(Y),
            torch.tensor(U),
            torch.tensor(V),
            torch.tensor(Lambda)
        )]

        Omega = T_l @ theta @ T_r + T_s
        Omega_tilde = block_diag(np.eye(nx), np.eye(nu), Lambda) @ Omega

        logger.info(
            f'Constraints satisfied? {self.check_constraints()}'
        )

        return (Omega, (Omega_tilde, X, Y, Lambda))

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
        if self.coupling_flat:
            L_x = utils.construct_lower_triangular_matrix(
                L_flat=self.L_x_flat, diag_length=self.nx
            ).to(self.device)
            L_y = utils.construct_lower_triangular_matrix(
                L_flat=self.L_y_flat, diag_length=self.nx
            ).to(self.device)

            X = L_x @ L_x.T
            Y = L_y @ L_y.T
        else:
            X = self.X
            Y = self.Y

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
        if self.coupling_flat:
            L_x = utils.construct_lower_triangular_matrix(
                L_flat=self.L_x_flat, diag_length=self.nx
            ).to(self.device)
            L_y = utils.construct_lower_triangular_matrix(
                L_flat=self.L_y_flat, diag_length=self.nx
            ).to(self.device)
            X = L_x @ L_x.T
            Y = L_y @ L_y.T
        else:
            X = self.X
            Y = self.Y
        U = torch.linalg.inv(Y) - X
        V = Y
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
        # Y_cal = torch.concat([
        #     torch.concat([Y, torch.eye(self.nx)], dim=1),
        #     torch.concat([V.T, torch.zeros((self.nx, self.nx))], dim=1)
        # ], dim=0)
        # X_cal = torch.concat([
        #     torch.concat([X, U], dim=1),
        #     torch.concat([U.T, - U.T @ Y @ torch.linalg.inv(V).T], dim=1)
        # ], dim=0)
        # X_bf = Y_cal.T @ X_cal @ X_cal

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
            for lam_el in lam:
                multiplier_constraints.append(lam_el >= 0)
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
                [np.eye(self.nx_rnn), np.zeros((self.nx,self.nx_rnn+self.nd+self.nw))],
                [
                    np.zeros((self.ny,self.nx)),
                    np.vstack((np.eye(self.nx_rnn),np.zeros((self.nd,self.nx_rnn)))), 
                    np.vstack((np.zeros((self.nx_rnn,self.nd)),np.eye(self.nd))), 
                    np.zeros((self.ny,self.nw))
                ],
                [np.zeros((self.nw,self.nx_rnn+self.nx+self.nd)),np.eye(self.nw)]
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
        # X_0, Y_0, U_0, V_0 = utils.get_coupling_matrices(self.L_x_flat,self.L_y_flat, self.nx)
        # Omega_tilde_0[:self.nx, self.nx : self.nx+self.nx] = X_0.detach().numpy()@self.A_lin.detach().numpy()@Y_0.detach().numpy()
        eps = 1e-3

        feasibility_constraint = [
            P << -eps * np.eye(nP),
            cp.bmat([[Y, np.eye(self.nx)], [np.eye(self.nx), X]])
            >> eps * np.eye(self.nx * 2),
            *multiplier_constraints,
        ]

        d = cp.Variable(shape=(1,))
        lam_0 = self.lam.detach().numpy()
        if self.multiplier_type == 'diagonal':
            Lambda_0 = np.diag(lam_0)
        elif self.multiplier_type=='static_zf':
            Lambda_0 = lam_0
        Omega_tilde_0 = self.Omega_tilde.detach().numpy()

        # distance_constraint = [cp.norm(X-X_0) <= d]
        # distance_constraint.append(cp.norm(Y-Y_0) <= d)
        # distance_constraint.append(cp.norm(Lambda-Lambda_0) <= d)
        # distance_constraint.append(cp.norm(Omega_tilde-Omega_tilde_0) <= d)

        # problem = cp.Problem(
        #     cp.Minimize(d),
        #     feasibility_constraint + distance_constraint
        # )

        problem = cp.Problem(
            cp.Minimize(d),
            feasibility_constraint + utils.get_distance_constraints(Omega_tilde_0, Omega_tilde, d)
        )
        # # problem = cp.Problem(
        # #     cp.Minimize(None),
        # #     feasibility_constraint
        # # )
        problem.solve(solver=self.optimizer, verbose=False, accept_unknown=True)

        logger.info(
            f'1. run: projection. '
            f'problem status {problem.status},'
            f'||Omega - Omega_0|| = {d.value}'
        )
        d_fixed = np.float64(d.value * self.increase_constraints)
        # d_fixed = np.float64(500)

        alpha = cp.Variable(shape=(1,))
        problem = cp.Problem(
            cp.Minimize(expr=alpha),
            feasibility_constraint
            + utils.get_distance_constraints(Omega_tilde_0, Omega_tilde, d_fixed)
            + utils.get_bounding_inequalities(X, Y, Omega_tilde, alpha),
        )
        problem.solve(solver=self.optimizer, verbose = False, accept_unknown=True)
        logger.info(
            f'2. run: parameter bounds. '
            f'problem status {problem.status},'
            f'alpha_star = {alpha.value}'
            f'||Omega - Omega_0|| = {np.linalg.norm(Omega_tilde.value- Omega_tilde_0)}'
        )

        alpha_fixed = np.float64(alpha.value * self.increase_constraints)
        # logger.info(
        #     'Size of coupling matrices: '
        #     f'|X| = {np.linalg.norm(X.value)}'
        #     f'|Y| = {np.linalg.norm(Y.value)}'
        # )

        beta = cp.Variable(shape=(1,))
        problem = cp.Problem(
            cp.Maximize(expr=beta),
            feasibility_constraint
            + utils.get_conditioning_constraints(Y, X, beta)
            + utils.get_distance_constraints(Omega_tilde_0, Omega_tilde, d_fixed)
            + utils.get_bounding_inequalities(X, Y, Omega_tilde, alpha_fixed),
        )
        problem.solve(solver=self.optimizer, accept_unknown=True)
        logger.info(
            f'3. run: coupling conditions. '
            f'problem status {problem.status},'
            f'beta_star = {beta.value}'
        )

        if not write_parameter:
            logger.info('Return distance.')
            return np.float64(d)

        logger.info('Write back projected parameters.')
        if self.coupling_flat:
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
        else:
            self.X.data = torch.tensor(X.value)
            self.Y.data = torch.tensor(Y.value)

        if self.multiplier_type == 'diagonal':
            self.lam.data = (
                torch.tensor(np.diag(np.array(Lambda.value))).double().to(device)
            )
        elif self.multiplier_type == 'static_zf':
            self.lam.data = torch.tensor(Lambda.value).double().to(device)
        self.Omega_tilde.data = torch.tensor(Omega_tilde.value).double().to(device)

        return np.float64(np.linalg.norm(Omega_tilde.value))
    
    def write_parameters(self, params: List[torch.Tensor]) -> None:
        for old_par, new_par in zip(params, self.parameters()):
            new_par.data = old_par.clone()


    def check_constraints(self) -> bool:
        with torch.no_grad():
            P = self.get_constraints()
            _, info = torch.linalg.cholesky_ex(-P)
        return True if info == 0 else False
    
class InputLinearizationRnnNoConstraint(ConstrainedForwardModule):
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
        nonlinearity: Callable[[torch.Tensor], torch.Tensor],
        device: torch.device = torch.device('cpu'),
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
            
        if self.init_omega == 'zero':
            self.theta = torch.nn.Parameter(
                torch.zeros(
                    size=(
                        self.nx + self.nu + self.nz,
                        self.nx + self.ny + self.nw,
                    )
                )
            ).to(device)
        elif self.init_omega == 'rand':
            self.theta = torch.nn.Parameter(
                torch.normal(0,1/self.nx, size=(
                    self.nx + self.nu + self.nz,
                    self.nx + self.ny + self.nw,
                )).double().to(device)
            )
        else:
            raise ValueError(f'Initialization method {self.init_omega} is not supported.')


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

    def set_lure_system(self) -> NDArray[np.float64]:
        device = self.device

        theta = self.theta

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

        return generalized_plant.cpu().detach().numpy()


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

    def get_constraints(self) -> torch.Tensor:
        pass

    def check_constraints(self) -> bool:
        pass
    
