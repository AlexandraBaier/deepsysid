import abc
import logging
import warnings
from typing import Callable, List, Optional, Tuple, Union, Literal, Set

import cvxpy as cp
import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from torch import nn
from scipy.linalg import block_diag
import importlib

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


class ConstrainedLSTM(ConstrainedForwardModule):
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

        # self.project_parameters(write_parameter=True)

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

    def get_initial_parameters(self):
        return super().get_initial_parameters()

    def get_constraints(self, constraints=torch.tensor(0.0)) -> torch.Tensor:
        l = self.num_recurrent_layers
        for l_i in range(l):
            (W_fs, W_is, W_cs, W_os) = utils.get_iss_parameter_layer_lstm(
                getattr(self.predictor_lstm, f"weight_ih_l{l_i}"),
                getattr(self.predictor_lstm, f"weight_hh_l{l_i}"),
                getattr(self.predictor_lstm, f"bias_ih_l{l_i}"),
                getattr(self.predictor_lstm, f"bias_hh_l{l_i}"),
                self.recurrent_dim
            )
            constraint, _ = utils.check_iss_lstm(W_fs, W_is, W_cs, W_os) 
            constraints += constraint + 1e-3 

        return constraints

    def check_constraints(self):
        l = self.num_recurrent_layers
        satisfieds = []
        for l_i in range(l):
            (W_fs, W_is, W_cs, W_os) = utils.get_iss_parameter_layer_lstm(
                getattr(self.predictor_lstm, f"weight_ih_l{l_i}"),
                getattr(self.predictor_lstm, f"weight_hh_l{l_i}"),
                getattr(self.predictor_lstm, f"bias_ih_l{l_i}"),
                getattr(self.predictor_lstm, f"bias_hh_l{l_i}"),
                self.recurrent_dim
            )
            _, satisfied = utils.check_iss_lstm(W_fs, W_is, W_cs, W_os)
            satisfieds.append(satisfied)

        return all(satisfieds)

    def project_parameters(self, write_parameter: bool = True) -> float:
        """
        Project LSTM parameters to satisfy ISS condition for each layer.
        
        The ISS condition for LSTM is:
        s_f + s_z * s_i * |U_c|_1 < 1
        
        Where:
        - s_f = sigmoid(||[W_f, U_f]||_∞)
        - s_z = sigmoid(||[W_o, U_o]||_∞) 
        - s_i = sigmoid(||[W_i, U_i]||_∞)
        - |U_c|_1 is the 1-norm of U_c
        
        Uses binary search to find optimal scaling factor in 60 iterations.
        All LSTM parameters (input-to-hidden weights, hidden-to-hidden weights,
        and biases for all gates: input, forget, cell, output) are scaled uniformly.
        
        Args:
            write_parameter: If True, update the model parameters in-place
            
        Returns:
            Distance measure of the parameter change
        """
        import logging
        logger = logging.getLogger(__name__)
        
        if self.check_constraints():
            logger.info('No projection necessary, ISS constraints are satisfied.')
            return 0.0
            
        total_distance = 0.0
        max_iterations = 60
        min_scaling = 0.001
        max_scaling = 1.0
        
        with torch.no_grad():
            for l_i in range(self.num_recurrent_layers):
                # Get current parameters
                weight_ih = getattr(self.predictor_lstm, f"weight_ih_l{l_i}")
                weight_hh = getattr(self.predictor_lstm, f"weight_hh_l{l_i}")
                bias_ih = getattr(self.predictor_lstm, f"bias_ih_l{l_i}")
                bias_hh = getattr(self.predictor_lstm, f"bias_hh_l{l_i}")
                
                # Store original parameters for distance calculation
                orig_weight_ih = weight_ih.clone()
                orig_weight_hh = weight_hh.clone()
                orig_bias_ih = bias_ih.clone()
                orig_bias_hh = bias_hh.clone()
                
                # Get ISS parameters
                (W_fs, W_is, W_cs, W_os) = utils.get_iss_parameter_layer_lstm(
                    weight_ih, weight_hh, bias_ih, bias_hh, self.recurrent_dim
                )
                
                # Check if this layer violates ISS condition
                iss_cond, satisfied = utils.check_iss_lstm(W_fs, W_is, W_cs, W_os)
                
                if not satisfied:
                    logger.info(f'Layer {l_i} violates ISS condition: {iss_cond.item():.6f}')
                    
                    # Binary search for optimal scaling factor
                    best_scaling = max_scaling
                    low_scaling = min_scaling
                    high_scaling = max_scaling
                    
                    # Store original weights for scaling
                    h = self.recurrent_dim
                    W_ii_orig, W_if_orig, W_ig_orig, W_io_orig = torch.split(weight_ih, h, dim=0)
                    W_hi_orig, W_hf_orig, W_hg_orig, W_ho_orig = torch.split(weight_hh, h, dim=0)
                    b_ii_orig, b_if_orig, b_ig_orig, b_io_orig = torch.split(bias_ih, h, dim=0)
                    b_hi_orig, b_hf_orig, b_hg_orig, b_ho_orig = torch.split(bias_hh, h, dim=0)
                    
                    for iteration in range(max_iterations):
                        # Current scaling factor (binary search)
                        current_scaling = (low_scaling + high_scaling) / 2.0
                        
                        # Apply scaling to all LSTM gate weights and biases
                        # Scale input-to-hidden weights
                        W_ii_scaled = W_ii_orig * current_scaling
                        W_if_scaled = W_if_orig * current_scaling
                        W_ig_scaled = W_ig_orig * current_scaling
                        W_io_scaled = W_io_orig * current_scaling
                        
                        # Scale hidden-to-hidden (recurrent) weights
                        W_hi_scaled = W_hi_orig * current_scaling
                        W_hf_scaled = W_hf_orig * current_scaling
                        W_hg_scaled = W_hg_orig * current_scaling
                        W_ho_scaled = W_ho_orig * current_scaling
                        
                        # Scale biases
                        b_ii_scaled = b_ii_orig * current_scaling
                        b_if_scaled = b_if_orig * current_scaling
                        b_ig_scaled = b_ig_orig * current_scaling
                        b_io_scaled = b_io_orig * current_scaling
                        b_hi_scaled = b_hi_orig * current_scaling
                        b_hf_scaled = b_hf_orig * current_scaling
                        b_hg_scaled = b_hg_orig * current_scaling
                        b_ho_scaled = b_ho_orig * current_scaling
                        
                        # Create temporary weight matrices with scaled weights
                        weight_ih_temp = torch.cat([W_ii_scaled, W_if_scaled, W_ig_scaled, W_io_scaled], dim=0)
                        weight_hh_temp = torch.cat([W_hi_scaled, W_hf_scaled, W_hg_scaled, W_ho_scaled], dim=0)
                        bias_ih_temp = torch.cat([b_ii_scaled, b_if_scaled, b_ig_scaled, b_io_scaled], dim=0)
                        bias_hh_temp = torch.cat([b_hi_scaled, b_hf_scaled, b_hg_scaled, b_ho_scaled], dim=0)
                        
                        # Test ISS condition with scaled weights
                        (W_fs_test, W_is_test, W_cs_test, W_os_test) = utils.get_iss_parameter_layer_lstm(
                            weight_ih_temp, weight_hh_temp, bias_ih_temp, bias_hh_temp, self.recurrent_dim
                        )
                        
                        iss_cond_test, satisfied_test = utils.check_iss_lstm(W_fs_test, W_is_test, W_cs_test, W_os_test)
                        
                        if satisfied_test:
                            # Constraint satisfied, try larger scaling (move up)
                            best_scaling = current_scaling
                            low_scaling = current_scaling
                            logger.debug(f'Layer {l_i}, iter {iteration+1}: scaling {current_scaling:.6f} satisfies constraint (iss_cond: {iss_cond_test.item():.6f})')
                        else:
                            # Constraint violated, try smaller scaling (move down)
                            high_scaling = current_scaling
                            logger.debug(f'Layer {l_i}, iter {iteration+1}: scaling {current_scaling:.6f} violates constraint (iss_cond: {iss_cond_test.item():.6f})')
                        
                        # Check convergence
                        if abs(high_scaling - low_scaling) < 1e-6:
                            logger.debug(f'Layer {l_i}: Converged after {iteration+1} iterations')
                            break
                    
                    # Apply the best scaling factor found
                    if write_parameter:
                        # Apply scaling to all LSTM parameters
                        W_ii_final = W_ii_orig * best_scaling
                        W_if_final = W_if_orig * best_scaling
                        W_ig_final = W_ig_orig * best_scaling
                        W_io_final = W_io_orig * best_scaling
                        W_hi_final = W_hi_orig * best_scaling
                        W_hf_final = W_hf_orig * best_scaling
                        W_hg_final = W_hg_orig * best_scaling
                        W_ho_final = W_ho_orig * best_scaling
                        b_ii_final = b_ii_orig * best_scaling
                        b_if_final = b_if_orig * best_scaling
                        b_ig_final = b_ig_orig * best_scaling
                        b_io_final = b_io_orig * best_scaling
                        b_hi_final = b_hi_orig * best_scaling
                        b_hf_final = b_hf_orig * best_scaling
                        b_hg_final = b_hg_orig * best_scaling
                        b_ho_final = b_ho_orig * best_scaling
                        
                        # Update the actual parameters
                        weight_ih_final = torch.cat([W_ii_final, W_if_final, W_ig_final, W_io_final], dim=0)
                        weight_hh_final = torch.cat([W_hi_final, W_hf_final, W_hg_final, W_ho_final], dim=0)
                        bias_ih_final = torch.cat([b_ii_final, b_if_final, b_ig_final, b_io_final], dim=0)
                        bias_hh_final = torch.cat([b_hi_final, b_hf_final, b_hg_final, b_ho_final], dim=0)
                        
                        weight_ih.data = weight_ih_final
                        weight_hh.data = weight_hh_final
                        bias_ih.data = bias_ih_final
                        bias_hh.data = bias_hh_final
                        
                        logger.info(f'Layer {l_i}: Applied scaling factor {best_scaling:.6f} to all LSTM parameters')
                        
                        # Verify final constraint satisfaction
                        (W_fs_final, W_is_final, W_cs_final, W_os_final) = utils.get_iss_parameter_layer_lstm(
                            weight_ih, weight_hh, bias_ih, bias_hh, self.recurrent_dim
                        )
                        iss_cond_final, satisfied_final = utils.check_iss_lstm(W_fs_final, W_is_final, W_cs_final, W_os_final)
                        logger.info(f'Layer {l_i}: Final ISS condition: {iss_cond_final.item():.6f}, satisfied: {satisfied_final}')
                    
                    # Calculate parameter change distance
                    new_weight_ih = getattr(self.predictor_lstm, f"weight_ih_l{l_i}")
                    new_weight_hh = getattr(self.predictor_lstm, f"weight_hh_l{l_i}")
                    new_bias_ih = getattr(self.predictor_lstm, f"bias_ih_l{l_i}")
                    new_bias_hh = getattr(self.predictor_lstm, f"bias_hh_l{l_i}")
                    
                    layer_distance = (
                        torch.norm(new_weight_ih - orig_weight_ih).item() +
                        torch.norm(new_weight_hh - orig_weight_hh).item() +
                        torch.norm(new_bias_ih - orig_bias_ih).item() +
                        torch.norm(new_bias_hh - orig_bias_hh).item()
                    )
                    total_distance += layer_distance
        
        if write_parameter:
            # Final verification
            if self.check_constraints():
                logger.info(f'All layers now satisfy ISS constraints. Total parameter distance: {total_distance:.6f}')
            else:
                logger.warning('Some layers still violate ISS constraints after projection.')
        
        return total_distance


class BasicMamba(HiddenStateForwardModule):
    def __init__(
        self,
        d_model: int,
        recurrent_dim: int,
        d_conv:int,
        expand:int,
    ):
        super().__init__()
        Mamba = getattr(importlib.import_module('mamba_ssm'), 'Mamba')
        self.predictor_mamba = Mamba(
            d_model=d_model, # Model dimension d_model
            d_state=recurrent_dim,  # SSM state expansion factor
            d_conv=d_conv,    # Local convolution width
            expand=2,    # Block expansion factor
        )

    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        y = self.predictor_mamba(x_pred)
        return y, (torch.zeros_like(y), torch.zeros_like(y))

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
        bias_terms: Tuple[torch.Tensor] = (),
        D22: Union[torch.Tensor, None] = None,
    ) -> None:
        super().__init__(A=A, B=B1, C=C1, D=D11)
        self._nw = B2.shape[1]
        self._nz = C2.shape[0]
        assert self._nw == self._nz
        self.B2 = B2
        self.C2 = C2
        self.D12 = D12
        self.D21 = D21
        self.D22 = D22
        self.Delta = Delta  # static nonlinearity
        self.device = device
        if len(bias_terms) == 0:
            self.bx = torch.zeros((self._nx,1))
            self.by = torch.zeros((self._ny,1))
            self.bz = torch.zeros((self._nz,1))
        else:
            self.bx, self.by, self.bz = bias_terms

        assert self._check_D22()

    def _check_D22(self)-> bool:
        # check if D22 has only entrys below the main diagonal
        return True
    
    def forward(
        self, x0: torch.Tensor, us: torch.Tensor, return_states: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        n_batch, N, _, _ = us.shape
        # x = torch.zeros(size=(n_batch, N + 1, self._nx, 1)).to(self.device)
        y = torch.zeros(size=(n_batch, N, self._ny, 1)).to(self.device)
        # w = torch.zeros(size=(n_batch, self._nw, 1)).to(self.device)
        x = x0.reshape(n_batch, self._nx, 1)

        for k in range(N):
            if self.D22 is None:
                w = self.Delta(self.C2 @ x + self.D21 @ us[:, k, :, :] + self.bz)
            else:
                ws = []
                for n_zi in range(self._nz):
                    if n_zi == 0:
                        ws.append(self.Delta(self.C2[n_zi, :].reshape((1,-1)) @ x + self.D21[n_zi,:].reshape((1,-1)) @ us[:,k,:,:]))
                    else:
                        ws.append(
                            self.Delta(
                                self.C2[n_zi, :].reshape((1,-1)) @ x 
                                + self.D21[n_zi,:].reshape((1,-1)) @ us[:,k,:,:] 
                                + self.D22[n_zi,:n_zi].reshape((1,n_zi)) @ torch.concat(ws,dim=1)
                            )
                        )
                w = torch.concat(ws,dim=1)
            x = super().state_dynamics(x=x, u=us[:, k, :, :]) + self.B2 @ w + self.bx
            y[:, k, :, :] = (
                super().output_dynamics(x=x, u=us[:, k, :, :]) + self.D12 @ w + self.by
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
        increase_constraints: Optional[np.float64] = 1.0,
        nu: Optional[int] = 0,
        bias: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.nx = nx  # state size
        self.nx_rnn = self.nx # controller has same state size
        self.nd = nd  # input size of performance channel
        self.ny = self.nx + self.nd  # output size of linearization
        self.ne = ne  # output size of performance channel
        if nu == 0:
            self.nu = self.nx + self.nx + self.ne # output size of controller
        else:
            self.nu = nu

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

        if bias:
            self.bx = torch.nn.Parameter(torch.zeros((self.nx+self.nx_rnn,1)).double().to(device))
            self.by = torch.nn.Parameter(torch.zeros((self.ne,1)).double().to(device))
            # self.bx = torch.zeros((self.nx+self.nx_rnn,1))
            # self.by = torch.zeros((self.ne,1))
            # self.bz = torch.nn.Parameter(torch.zeros((self.nz,1)).double().to(device))
            self.bz = torch.zeros((self.nz,1))

        else:
            self.bx = torch.zeros((self.nx+self.nx_rnn,1))
            self.by = torch.zeros((self.ne,1))
            self.bz = torch.zeros((self.nz,1))

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

        self.nu = B_lin_2.shape[1]

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

        # print(f'nx {self.nx}, nx_rnn {self.nx_rnn}, nz {self.nz}, nu {self.nu}')
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
                [np.zeros((self.nx,self.nx)), np.eye(self.nx), np.zeros((self.nx,self.nd+self.nw))],
                [
                    np.vstack((np.eye(self.nx_rnn),np.zeros((self.nd,self.nx_rnn)))),
                    np.zeros((self.ny,self.nx)),
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
            bias_terms=(self.bx,self.by,self.bz)
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
            [torch.vstack((Y, torch.zeros((self.nd, self.nx_rnn)))), torch.eye(self.ny), torch.zeros((self.ny, self.nw))],
            [torch.zeros((self.nw, self.nx_rnn)), torch.zeros((self.nw,self.ny)), torch.eye(self.nw)]
        ]).double().to(self.device)
        T_s = utils.torch_bmat([
            [X @ self.A_lin @ Y, torch.zeros((self.nx_rnn, self.ny+self.nw))],
            [torch.zeros((self.nu, self.nx_rnn+self.ny+self.nw))],
            [torch.zeros((self.nz, self.nx_rnn+self.ny+self.nw))]
        ]).double().to(self.device)

        return (T_l, T_r, T_s)


    def initialize_parameters(self) -> None:
        (Omega, (Omega_tilde, X, Y, U, V, L)) = self.get_initial_parameters()
    
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
                torch.tensor(np.diag(np.array(L))).double().to(self.device)
            )
        elif self.multiplier_type == 'static_zf':
            self.lam.data = torch.tensor(L).double().to(self.device)
        self.Omega_tilde.data = torch.tensor(Omega_tilde).double().to(self.device)

        assert self.check_constraints(), "Constraints are not satisfied."

        return


    def get_initial_parameters(
        self,
    ) -> Union[
        NDArray[np.float64],
        Tuple[
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
            NDArray[np.float64],
        ],
    ]:  
        nx, nu, nz, ny, nw, ne, nd = self.nx, self.nu, self.nz, self.ny, self.nw, self.ne, self.nd
        nxi = nx+ nx
        theta = np.zeros((nx+nu+nz, nx+ny+nw))
        gen_plant = self.S_s.detach().numpy() + self.S_l.detach().numpy() @ theta @ self.S_r.detach().numpy()

        (A_cal,B_cal,B2_cal, C_cal,D_cal,D12_cal,C2_cal,D21_cal,D22_cal) = utils.get_cal_matrices(gen_plant,nxi,nd,ne,nz)

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

        X = cp.Variable((nx,nx), symmetric=True)
        U = cp.Variable((nx,nx))
        X_hat = cp.Variable((nx,nx))
        X_cal = cp.bmat([
            [X, U],
            [U.T, -U]
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
        ga = cp.Variable((1,1))
        # ga = self.gamma**2

        M_theta = L1.T @ cp.bmat([[-X_cal, np.zeros((nxi,nxi))], [np.zeros((nxi,nxi)), X_cal]]) @ L1  \
        + L2.T @ cp.bmat([[-ga * np.eye(nd), np.zeros((nd,ne))], [np.zeros((ne,nd)), np.eye(ne)]])@L2 \
        + L3.T @ cp.bmat([[-(Lambda + Lambda.T), self.beta*Lambda], [self.beta*Lambda.T, np.zeros((nw,nz))]]) @ L3

        eps = 1e-3
        nM = M_theta.shape[0]
        constraints = [
            M_theta << -eps*np.eye(nM), 
            X_cal >> eps*np.eye(nxi),
            *multiplier_constraints
        ]
        problem = cp.Problem(
            cp.Minimize(ga),
            constraints
        )
        problem.solve(solver=self.optimizer, verbose = False)
        if not problem.status == 'optimal':
            raise ValueError(f'Optimizer did not find a solution: {problem.status}')

        assert np.sqrt(ga.value) <= self.gamma, f'||H_lin||_inf: {self.gamma}, ||S_theta||_inf: {np.sqrt(ga.value)}'

        logger.info(
            f'Optimizing for |theta| = 0, status: {problem.status} \n'
            f'Max real eig (M_theta): {max(np.real(np.linalg.eig(M_theta.value)[0]))}'
        )
        
        # extract coupling matrices from optimization result and transform to Omega_tilde parameters
        X, U, L = X.value, U.value, Lambda.value
        X_cal_inv = np.linalg.inv(utils.bmat([[X, U],[U.T,-U]]))
        # X_cal_inv = np.linalg.inv(X_cal.value)
        Y = X_cal_inv[:nx,:nx]
        V = X_cal_inv[:nx,nx:nx+nx]
        
        T_l, T_r, T_s = [T.detach().numpy() for T in self.get_T(
            torch.tensor(X),
            torch.tensor(Y),
            torch.tensor(U),
            torch.tensor(V),
            torch.tensor(L)
        )]

        Omega = T_l @ theta @ T_r + T_s
        Omega_tilde = block_diag(np.eye(nx), np.eye(nu), L) @ Omega

        return (Omega, (Omega_tilde, X, Y, U, V, L))

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
        P_21_4 = torch.from_numpy(
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
        )
        # P_21_4 = self.S_r
        # print(f'omega tilde {self.Omega_tilde.shape} P21 1 {P_21_1.shape}, p21 4 {P_21_4.shape}')
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
        
    def project_zero_theta(self) -> np.float64:
        nx, nx_rnn, nu, nz, ny, nw, nd, ne = self.nx, self.nx_rnn, self.nu, self.nz, self.y, self.nw, self.nx, self.ne
        nxi = nx+nx_rnn

        theta = np.zeros((nx+nu+nz,nx+ny+nw))
        gen_plant = self.S_s + self.S_l @ theta @ self.S_r

        (
            A_cal,
            B_cal,
            B2_cal,
            C_cal,
            D_cal,
            D12_cal,
            C2_cal,
            D21_cal,
            D22_cal,
        ) = utils.get_cal_matrices(
            gen_plant,
            nx+self.nx_rnn,
            nd,
            ne,
            nz
        )
        L1 = utils.bmat([
            [np.np.eye(nxi), np.np.zeros((nxi,nd+nw))],
            [A_cal, B_cal, B2_cal]
        ])
        L2 = utils.bmat([
            [np.zeros((nd,nxi)), np.eye(nd), np.zeros((nd,nw))],
            [C_cal, D_cal, D12_cal]
        ])
        L3 = utils.bmat([
            [np.zeros((nw,nxi+nd)), np.eye(nw)],
            [C2_cal, D21_cal, D22_cal]
        ])

        X = cp.Variable((nx,nx), symmetric=True)
        U = cp.Variable((nx,nx))
        X_cal = cp.bmat([
            [X, U],
            [U.T,-U]
        ])

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

        ga = cp.Variable((1,1))

        M = L1.T @ cp.bmat([[-X_cal, np.zeros((nxi,nxi))], [np.zeros((nxi,nxi)), X]]) @ L1 + \
            L2.T @ cp.bmat([[-ga * np.eye(nd), np.zeros((nd,ne))], [np.zeros((ne,nd)), np.eye(ne)]]) @ L2 + \
            L3.T @ cp.bmat([[-(Lambda+Lambda.T), self.beta*Lambda],[self.beta*Lambda.T, np.zeros((nz,nz))]])@L3

        constr = []
        constr.append(M<<1e-3 * np.eye(M.shape[0]))

        prob = cp.Problem(cp.Minimize(ga),constr)
        prob.solve(solver=cp.MOSEK,verbose=False)

        X, U, L = X.value, U.value, Lambda.value
        X_cal_inv = np.linalg.inv(X_cal.value)
        Y = X_cal_inv[:nx,:nx]
        V = X_cal_inv[:nx,nx:nx+nx]

        T_l = utils.bmat([
            [U, X@self.B_lin_2, np.zeros((nx_rnn, nz)).to(self.device)],
            [np.zeros((nu, nx_rnn)),np.eye(nu),np.zeros((nu, nz))],
            [np.zeros((nz, nx_rnn+nu)),np.eye(nz)]
        ])
        T_r = utils.bmat([
            [V.T, np.zeros((nx_rnn,ny)), np.zeros((nx_rnn,nw))],
            [np.vstack((Y, np.zeros((nd, nx_rnn)))), np.eye(ny), np.zeros((ny, nw))],
            [np.zeros((nw, nx_rnn)), np.zeros((nw,ny)), np.eye(nw)]
        ])
        T_s = utils.bmat([
            [X @ self.A_lin @ Y, np.zeros((nx_rnn, ny+nw))],
            [np.zeros((nu, nx_rnn+ny+nw))],
            [np.zeros((nz, nx_rnn+ny+nw))]
        ])

        Omega = T_l @ theta @ T_r + T_s
        

        return np.sqrt(ga.value)

    def project_parameters(self, write_parameter: bool = True) -> np.float64:
        if self.check_constraints():
            logger.info('No projection necessary, constraints are satisfied.')
            return np.float64(0.0)
        X = cp.Variable(shape=(self.nx, self.nx), symmetric=True)
        # X = cp.Variable(shape=(self.nx, self.nx))
        Y = cp.Variable(shape=(self.nx_rnn, self.nx_rnn), symmetric=True)       
        # Y = cp.Variable(shape=(self.nx_rnn, self.nx_rnn))       

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

        # Omega_tilde = cp.Variable(
        #     shape=(
        #         self.nx + self.nu + self.nz,
        #         self.nx + self.ny + self.nw,
        #     )
        # )
        K,L,L2 = cp.Variable((self.nx,self.nx)), cp.Variable((self.nx,self.ny)), cp.Variable((self.nx,self.nw))
        M,N,N12 = cp.Variable((self.nu,self.nx)), cp.Variable((self.nu,self.ny)), cp.Variable((self.nu,self.nw))
        M2,N21,N22 = cp.Variable((self.nz,self.nx)), cp.Variable((self.nz,self.ny)), np.zeros((self.nz,self.nw))
        Omega_tilde = cp.bmat([
            [K,L,L2],
            [M,N,N12],
            [M2,N21,N22]
        ])

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
        # gen_plant = P_21_2 @ Omega_tilde @ P_21_4

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

        model_match_dist = cp.Variable((1,1))
        P_11 = cp.bmat(
            [
                [-X_bf, torch.zeros(size=(nxi, self.nd)), self.beta*C2_bf_tilde.T],
                [torch.zeros(size=(self.nd,nxi)), -model_match_dist*torch.eye(self.nd), self.beta*D21_bf_tilde.T],
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

        # Omega_tilde_0 = self.Omega_tilde.cpu().detach().numpy()
        # X_0, Y_0, U_0, V_0 = utils.get_coupling_matrices(self.L_x_flat,self.L_y_flat, self.nx)
        # Omega_tilde_0[:self.nx, self.nx : self.nx+self.nx] = X_0.detach().numpy()@self.A_lin.detach().numpy()@Y_0.detach().numpy()
        eps = 1e-3

        feasibility_constraint = [
            P << -eps * np.eye(nP),
            # cp.bmat([[Y, np.eye(self.nx)], [np.eye(self.nx), X]])
            # >> eps * np.eye(self.nx * 2),
            *multiplier_constraints,
        ]

        # problem = cp.Problem(
        #     cp.Minimize(cp.norm(Omega_tilde)),
        #     feasibility_constraint
        # )
        problem = cp.Problem(
            cp.Minimize(model_match_dist),
            feasibility_constraint
        )
        problem.solve(solver=self.optimizer, verbose=False, accept_unknown=True)

        logger.info(
            f'1. run: projection. '
            f'problem status {problem.status},'
            # f'||Omega - Omega_0|| = {d.value}'
        )

        # d = cp.Variable(shape=(1,))
        # lam_0 = self.lam.detach().numpy()
        # if self.multiplier_type == 'diagonal':
        #     Lambda_0 = np.diag(lam_0)
        # elif self.multiplier_type=='static_zf':
        #     Lambda_0 = lam_0
        # Omega_tilde_0 = self.Omega_tilde.detach().numpy()

        # distance_constraint = [cp.norm(X-X_0) <= d]
        # distance_constraint.append(cp.norm(Y-Y_0) <= d)
        # distance_constraint.append(cp.norm(Lambda-Lambda_0) <= d)
        # distance_constraint.append(cp.norm(Omega_tilde-Omega_tilde_0) <= d)

        # problem = cp.Problem(
        #     cp.Minimize(d),
        #     feasibility_constraint + distance_constraint
        # )

        # problem = cp.Problem(
        #     cp.Minimize(d),
        #     feasibility_constraint + utils.get_distance_constraints(Omega_tilde_0, Omega_tilde, d)
        # )

        # # problem = cp.Problem(
        # #     cp.Minimize(None),
        # #     feasibility_constraint
        # # )

        # d_fixed = np.float64(d.value * self.increase_constraints)
        # # d_fixed = np.float64(500)

        # alpha = cp.Variable(shape=(1,))
        # problem = cp.Problem(
        #     cp.Minimize(expr=alpha),
        #     feasibility_constraint
        #     + utils.get_distance_constraints(Omega_tilde_0, Omega_tilde, d_fixed)
        #     + utils.get_bounding_inequalities(X, Y, Omega_tilde, alpha),
        # )
        # problem.solve(solver=self.optimizer, verbose = False, accept_unknown=True)
        # logger.info(
        #     f'2. run: parameter bounds. '
        #     f'problem status {problem.status},'
        #     f'alpha_star = {alpha.value}'
        #     f'||Omega - Omega_0|| = {np.linalg.norm(Omega_tilde.value- Omega_tilde_0)}'
        # )

        # alpha_fixed = np.float64(alpha.value * self.increase_constraints)
        # logger.info(
        #     'Size of coupling matrices: '
        #     f'|X| = {np.linalg.norm(X.value)}'
        #     f'|Y| = {np.linalg.norm(Y.value)}'
        # )

        # beta = cp.Variable(shape=(1,))
        # problem = cp.Problem(
        #     cp.Maximize(expr=beta),
        #     feasibility_constraint
        #     + utils.get_conditioning_constraints(Y, X, beta)
        #     + utils.get_distance_constraints(Omega_tilde_0, Omega_tilde, d_fixed)
        #     + utils.get_bounding_inequalities(X, Y, Omega_tilde, alpha_fixed),
        # )
        # problem.solve(solver=self.optimizer, accept_unknown=True)
        # logger.info(
        #     f'3. run: coupling conditions. '
        #     f'problem status {problem.status},'
        #     f'beta_star = {beta.value}'
        # )

        if not write_parameter:
            logger.info('Return distance.')
            return np.float64(0.0)  # Fixed: return 0.0 when d is not calculated

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
    

class InputLinearizationRnn3(ConstrainedForwardModule):
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
        increase_constraints: Optional[np.float64] = 1.0,
        nu: Optional[int] = 0,
        bias: Optional[bool] = False
    ) -> None:
        super().__init__()
        self.nx = nx  # state size
        self.nx_rnn = self.nx # controller has same state size
        self.nd = nd  # input size of performance channel
        self.ny = self.nx + self.nd  # output size of linearization
        self.ne = ne  # output size of performance channel
        if nu == 0:
            self.nu = self.nx + self.nx + self.ne # output size of controller
        else:
            self.nu = nu

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

        # \State $\Lambda, \Xc \gets I$
        if self.multiplier_type == 'diagonal':
            self.lam = torch.ones(size=(self.nz,)).double().to(device)
        elif self.multiplier_type == 'static_zf':
            self.lam = torch.eye(self.nz).double().to(device)
        else:
            raise ValueError(f'Multiplier type {self.multiplier_type} not supported.')
        # self.Xcal = torch.eye(self.nx+self.nx_rnn).double().to(device)

        lb = -1/np.sqrt(self.nx)
        ub = -lb

        rnd_X = (ub - lb) * torch.rand(self.nx,self.nx) + lb
        X = torch.eye(self.nx) + rnd_X.T @ rnd_X
        self.X = 1/2*(X.T @ X)

        rnd_Y = (ub - lb) * torch.rand(self.nx,self.nx) + lb
        Y = torch.eye(self.nx) + rnd_Y.T @ rnd_Y
        self.Y = 1/2*(Y.T @ Y)

        self.X_cal = self.get_Xcal(*self.get_coupling_matrices())

        # \State $\theta \gets 0$
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

        if bias:
            self.bx = torch.nn.Parameter(torch.zeros((self.nx+self.nx_rnn,1)).double().to(device))
            self.by = torch.nn.Parameter(torch.zeros((self.ne,1)).double().to(device))
            self.bz = torch.zeros((self.nz,1))

        else:
            self.bx = torch.zeros((self.nx+self.nx_rnn,1))
            self.by = torch.zeros((self.ne,1))
            self.bz = torch.zeros((self.nz,1))

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

        self.u = B_lin_2.shape[1]

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
                [np.zeros((self.nx,self.nx)), np.eye(self.nx), np.zeros((self.nx,self.nd+self.nw))],
                [
                    np.vstack((np.eye(self.nx_rnn),np.zeros((self.nd,self.nx_rnn)))),
                    np.zeros((self.ny,self.nx)),
                    np.vstack((np.zeros((self.nx_rnn,self.nd)),np.eye(self.nd))), 
                    np.zeros((self.ny,self.nw))
                ],
                [np.zeros((self.nw,self.nx_rnn+self.nx+self.nd)),np.eye(self.nw)]
            ])
        ).double().to(self.device)
        
    def find_Xcal_lambda(self) -> bool:
        logger.info(f'---Find X_cal and Lambda ---')
        device = self.device
        nx, nu, nz, ny, nw, ne, nd = self.nx, self.nu, self.nz, self.ny, self.nw, self.ne, self.nd
        nxi = nx+ nx

        theta = self.theta.detach().numpy()
        generalized_plant = self.S_s.detach().numpy() + self.S_l.detach().numpy() @ theta @ self.S_r.detach().numpy()
        (
            A_cal,
            B_cal,
            B2_cal,
            C_cal,
            D_cal,
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

        L1 = utils.bmat([
            [np.eye(nxi), np.zeros((nxi,nd+nw))],
            [A_cal, B_cal, B2_cal]
        ])
        L2 = utils.bmat([
            [np.zeros((nd,nxi)), np.eye(nd), np.zeros((nd,nw))],
            [C_cal, D_cal, D12_cal]
        ])
        L3 = utils.bmat([
            [np.zeros((nw,nxi+nd)), np.eye(nw)],
            [C2_cal, D21_cal, D22_cal]
        ])

        X_cal = cp.Variable((nxi,nxi))

        multiplier_constraints = []
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

        t = cp.Variable((1,1))

        M = L1.T @ cp.bmat([[-X_cal, np.zeros((nxi,nxi))], [np.zeros((nxi,nxi)), X_cal]]) @ L1 + \
            L2.T @ cp.bmat([[-self.gamma**2 * np.eye(nd), np.zeros((nd,ne))], [np.zeros((ne,nd)), np.eye(ne)]]) @ L2 + \
            L3.T @ cp.bmat([[-(Lambda+Lambda.T), self.beta*Lambda],[self.beta*Lambda.T, np.zeros((nz,nz))]])@L3

        constr = []
        constr.append(M<<t * np.eye(M.shape[0]))

        prob = cp.Problem(cp.Minimize(t),constr)
        try:
            prob.solve(solver=cp.MOSEK,verbose=False)
        except:
            logger.info('Could not solve SDP')
            return False

        if t.value > 0.0:
            logger.info('Did not find feasible X_cal and Lambda')
            return False
    
        logger.info(
            f'1. run.'
            f'problem status: {prob.status}'
            f't: {t.value}'
        )
        
        logger.info('Write back projected parameters.')
        if self.multiplier_type == 'diagonal':
            self.lam = (
                torch.tensor(np.diag(np.array(Lambda.value))).double().to(device)
            )
        elif self.multiplier_type == 'static_zf':
            self.lam = torch.tensor(Lambda.value).double().to(device)
        self.X_cal = torch.tensor(X_cal.value).double().to(device)
        return True

    def bijective_transformation(self) -> NDArray[np.float64]:
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
                        Lambda
                    ], dim=1
                )
            ],dim=0
        )
        
        X,Y,U,V = self.get_coupling_matrices()

        T_l,T_r,T_s = self.get_T(X,Y,U,V,Lambda)

        Omega = T_l @ self.theta @ T_r + T_s

        Omega_tilde = L @ Omega

        return Omega_tilde.cpu().detach().numpy()


    def set_lure_system(self) -> Tuple[SimAbcdParameter, NDArray[np.float64]]:
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
            bias_terms=(self.bx,self.by,self.bz)
        ).to(device)

        if self.multiplier_type == 'diagonal':
            Lambda = np.diag(self.lam.detach().numpy())

        elif self.multiplier_type == 'static_zf':
            # static zames falb multiplier, Lambda must be double hyperdominant
            Lambda = self.lam.detach().numpy()

        pars = SimAbcdParameter(
            theta.cpu().detach().numpy(),
            self.get_Xcal(*self.get_coupling_matrices()).cpu().detach().numpy(),
            Lambda
        )

        return (pars, generalized_plant.cpu().detach().numpy())

    def get_Xcal(self, X:torch.Tensor,Y:torch.Tensor,U:torch.Tensor,V:torch.Tensor) -> torch.Tensor:
        return torch.linalg.inv(utils.torch_bmat([
            [Y,V],
            [torch.eye(self.nx), torch.zeros((self.nx,self.nx))]
        ])) @ utils.torch_bmat([
            [torch.eye(self.nx), torch.zeros((self.nx,self.nx))],
            [X, U]
        ])

    def get_coupling_matrices(
            self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
            [torch.vstack((Y, torch.zeros((self.nd, self.nx_rnn)))), torch.eye(self.ny), torch.zeros((self.ny, self.nw))],
            [torch.zeros((self.nw, self.nx_rnn)), torch.zeros((self.nw,self.ny)), torch.eye(self.nw)]
        ]).double().to(self.device)
        T_s = utils.torch_bmat([
            [X @ self.A_lin @ Y, torch.zeros((self.nx_rnn, self.ny+self.nw))],
            [torch.zeros((self.nu, self.nx_rnn+self.ny+self.nw))],
            [torch.zeros((self.nz, self.nx_rnn+self.ny+self.nw))]
        ]).double().to(self.device)

        return (T_l, T_r, T_s)


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


    def get_constraints(self) -> torch.Tensor:
        nx, nu, nz, ny, nw, ne, nd = self.nx, self.nu, self.nz, self.ny, self.nw, self.ne, self.nd
        nxi = nx+ nx

        theta = self.theta.detach().numpy()
        generalized_plant = self.S_s.detach().numpy() + self.S_l.detach().numpy() @ theta @ self.S_r.detach().numpy()
        (
            A_cal,
            B_cal,
            B2_cal,
            C_cal,
            D_cal,
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
        L1 = utils.bmat([
            [np.eye(nxi), np.zeros((nxi,nd+nw))],
            [A_cal, B_cal, B2_cal]
        ])
        L2 = utils.bmat([
            [np.zeros((nd,nxi)), np.eye(nd), np.zeros((nd,nw))],
            [C_cal, D_cal, D12_cal]
        ])
        L3 = utils.bmat([
            [np.zeros((nw,nxi+nd)), np.eye(nw)],
            [C2_cal, D21_cal, D22_cal]
        ])

        X_cal = self.get_Xcal(*self.get_coupling_matrices())

        if self.multiplier_type == 'diagonal':
            Lambda = np.diag(self.lam.detach().numpy())

        elif self.multiplier_type == 'static_zf':
            # static zames falb multiplier, Lambda must be double hyperdominant
            Lambda = self.lam.detach().numpy()

        M = L1.T @ utils.bmat([[-X_cal, np.zeros((nxi,nxi))], [np.zeros((nxi,nxi)), X_cal]]) @ L1 + \
            L2.T @ utils.bmat([[-self.gamma**2 * np.eye(nd), np.zeros((nd,ne))], [np.zeros((ne,nd)), np.eye(ne)]]) @ L2 + \
            L3.T @ utils.bmat([[-(Lambda+Lambda.T), self.beta*Lambda],[self.beta*Lambda.T, np.zeros((nz,nz))]])@L3
        return M
        
    def project_theta_parameters(self, theta_0:NDArray[np.float64]) -> None:
        logger.info(f'--- Project theta parameter ---')
        device = self.device
        nx, nu, nz, ny, nw, ne, nd = self.nx, self.nu, self.nz, self.ny, self.nw, self.ne, self.nd
        nxi = nx+ nx

        theta = cp.Variable((
            self.nx + self.nu + self.nz,
            self.nx + self.ny + self.nw,
        ))
        generalized_plant = self.S_s.detach().numpy() + self.S_l.detach().numpy() @ theta @ self.S_r.detach().numpy()
        (
            A_cal,
            B_cal,
            B2_cal,
            C_cal,
            D_cal,
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

        X, Y, U, V = self.get_coupling_matrices()

        X_cal = self.get_Xcal(X,Y,U,V)

        if self.multiplier_type == 'diagonal':
            Lambda = np.diag(self.lam.detach().numpy())

        elif self.multiplier_type == 'static_zf':
            # static zames falb multiplier, Lambda must be double hyperdominant
            Lambda = self.lam.detach().numpy()

        d = cp.Variable((1,))

        P_11 = cp.bmat(
            [
                [-X_cal, torch.zeros(size=(nxi, self.nd)), (Lambda @ C2_cal).T],
                [torch.zeros(size=(self.nd,nxi)), -self.gamma**2 * torch.eye(self.nd), (Lambda @ D21_cal).T],
                [Lambda @ C2_cal, Lambda @ D21_cal, -(Lambda.T+Lambda)]
            ]
        )

        P_21 = cp.bmat(
            [
                [A_cal, B_cal, B2_cal],
                [C_cal, D_cal, D12_cal],
            ]
        )

        P_22 = cp.bmat(
            [
                [-X_cal, torch.zeros(size=(nxi, self.ne))],
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

        eps = 0

        constr = []
        constr.append(P<<-eps * np.eye(nP))
        constr.append(cp.norm(theta_0-theta)<=d)

        prob = cp.Problem(cp.Minimize(d),constr)
        prob.solve(solver=cp.MOSEK,verbose=False)
    
        logger.info(
            f'1. run: projection. problem status: {prob.status}'
            f'||theta-theta_0||: {d.value}'
        )
        
        logger.info('Write back projected parameters.')
        self.theta.data = torch.as_strided(torch.tensor(theta.value), theta.value.shape, self.theta.grad.stride())



    def project_omega_parameters(self, Omega_tilde_0:NDArray[np.float64]) -> np.float64:
        logger.info('---Project Omega tilde parameters---')
        X = cp.Variable(shape=(self.nx, self.nx), symmetric=True)
        Y = cp.Variable(shape=(self.nx_rnn, self.nx_rnn), symmetric=True)        

        multiplier_constraints = []
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
        # gen_plant = P_21_2 @ Omega_tilde @ P_21_4

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
                [torch.zeros(size=(self.nd,nxi)), -self.gamma**2 * torch.eye(self.nd), self.beta*D21_bf_tilde.T],
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

        eps = 0

        feasibility_constraint = [
            P << -eps * np.eye(nP),
            cp.bmat([[Y, np.eye(self.nx)], [np.eye(self.nx), X]])
            >> eps * np.eye(self.nx * 2),
            *multiplier_constraints,
        ]
        d = cp.Variable(shape=(1,))

        problem = cp.Problem(
            cp.Minimize(d),
            feasibility_constraint
            + utils.get_distance_constraints(Omega_tilde_0, Omega_tilde,d)
        )
        problem.solve(solver=self.optimizer, verbose=False, accept_unknown=True)

        logger.info(
            f'1. run: projection. '
            f'problem status {problem.status},'
            f'||Omega - Omega_0|| = {d.value}'
        )

        logger.info(
            'Size of coupling matrices: '
            f'|X| = {np.linalg.norm(X.value)}'
            f'|Y| = {np.linalg.norm(Y.value)}'
        )

        d_fixed = d.value + 100
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
        logger.info(
            'Size of coupling matrices: '
            f'|X| = {np.linalg.norm(X.value)}'
            f'|Y| = {np.linalg.norm(Y.value)}'
        )

        alpha_fixed = np.float64(alpha.value + 10)

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
            f'||Omega - Omega_0|| = {np.linalg.norm(Omega_tilde.value- Omega_tilde_0)}'
        )
        logger.info(
            'Size of coupling matrices: '
            f'|X| = {np.linalg.norm(X.value)}'
            f'|Y| = {np.linalg.norm(Y.value)}'
        )

        self.X = torch.tensor(X.value)
        self.Y = torch.tensor(Y.value)

        self.X_cal = self.get_Xcal(*self.get_coupling_matrices())

        if self.multiplier_type == 'diagonal':
            self.lam.data = (
                torch.tensor(np.diag(np.array(Lambda.value))).double().to(self.device)
            )
        elif self.multiplier_type == 'static_zf':
            self.lam.data = torch.tensor(Lambda.value).double().to(self.device)
        # self.Omega_tilde.data = torch.tensor(Omega_tilde.value).double().to(device)

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
    
class InputLinearizationRnnNonConvex(ConstrainedForwardModule):
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
        init_omega: Optional[str]='zero',
        optimizer: str = cp.SCS,
        multiplier_type: Optional[str] = 'diag',
        nu: Optional[int] = 0,
        increase_constraints: Optional[np.float64] = 1.0
    ) -> None:
        super().__init__()
        self.nx = nx  # state size
        self.nx_rnn = self.nx # controller has same state size
        self.nd = nd  # input size of performance channel
        self.ny = self.nx + self.nd  # output size of linearization
        self.ne = ne  # output size of performance channel
        if nu == 0:
            self.nu = self.nx + self.nx + self.ne # output size of controller
        else:
            self.nu = nu
        self.nw = nw
        self.nz = self.nw
        
        self.init_omega = init_omega

        self.alpha = alpha
        self.beta = beta

        self.device = device
        self.nl = nonlinearity
        self.multiplier_type = multiplier_type
        self.increase_constraints = increase_constraints
            
        if self.multiplier_type == 'diagonal':
            self.lam = torch.nn.Parameter(
                torch.ones(size=(self.nz,)).double().to(device)
            )
        elif self.multiplier_type == 'static_zf':
            self.lam = torch.nn.Parameter(torch.eye(self.nz).double().to(device))
        else:
            raise ValueError(f'Multiplier type {self.multiplier_type} not supported.')

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
        
        self.X_cal = torch.nn.Parameter(
            torch.zeros((self.nx+self.nx_rnn, self.nx+self.nx_rnn))
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

        self.u = B_lin_2.shape[1]

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
                [np.zeros((self.nx,self.nx)), np.eye(self.nx), np.zeros((self.nx,self.nd+self.nw))],
                [
                    np.vstack((np.eye(self.nx_rnn),np.zeros((self.nd,self.nx_rnn)))),
                    np.zeros((self.ny,self.nx)),
                    np.vstack((np.zeros((self.nx_rnn,self.nd)),np.eye(self.nd))), 
                    np.zeros((self.ny,self.nw))
                ],
                [np.zeros((self.nw,self.nx_rnn+self.nx+self.nd)),np.eye(self.nw)]
            ])
        ).double().to(self.device)
    

    def initialize_parameters(self) -> None:
        device = self.device
        nx, nu, nz, ny, nw, ne, nd = self.nx, self.nu, self.nz, self.ny, self.nw, self.ne, self.nd
        nxi = nx+ nx
        # theta = cp.Variable((
        #     self.nx + self.nu + self.nz,
        #     self.nx + self.ny + self.nw,)
        # )
        theta = self.theta.detach().numpy()
        generalized_plant = self.S_s.detach().numpy() + self.S_l.detach().numpy() @ theta @ self.S_r.detach().numpy()
        (
            A_cal,
            B_cal,
            B2_cal,
            C_cal,
            D_cal,
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

        L1 = utils.bmat([
            [np.eye(nxi), np.zeros((nxi,nd+nw))],
            [A_cal, B_cal, B2_cal]
        ])
        L2 = utils.bmat([
            [np.zeros((nd,nxi)), np.eye(nd), np.zeros((nd,nw))],
            [C_cal, D_cal, D12_cal]
        ])
        L3 = utils.bmat([
            [np.zeros((nw,nxi+nd)), np.eye(nw)],
            [C2_cal, D21_cal, D22_cal]
        ])

        X_cal = cp.Variable((nxi,nxi))

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

        ga = cp.Variable((1,1))

        if self.nd ==1:
            M = L1.T @ cp.bmat([[-X_cal, np.zeros((nxi,nxi))], [np.zeros((nxi,nxi)), X_cal]]) @ L1 + \
                L2.T @ cp.bmat([[-ga, np.zeros((nd,ne))], [np.zeros((ne,nd)), np.eye(ne)]]) @ L2 + \
                L3.T @ cp.bmat([[-(Lambda+Lambda.T), self.beta*Lambda],[self.beta*Lambda.T, np.zeros((nz,nz))]])@L3
        else:
            M = L1.T @ cp.bmat([[-X_cal, np.zeros((nxi,nxi))], [np.zeros((nxi,nxi)), X_cal]]) @ L1 + \
                L2.T @ cp.bmat([[-ga * np.eye(nd), np.zeros((nd,ne))], [np.zeros((ne,nd)), np.eye(ne)]]) @ L2 + \
                L3.T @ cp.bmat([[-(Lambda+Lambda.T), self.beta*Lambda],[self.beta*Lambda.T, np.zeros((nz,nz))]])@L3

        constr = []
        constr.append(M<<-1e-3 * np.eye(M.shape[0]))

        prob = cp.Problem(cp.Minimize(ga),constr)
        prob.solve(solver=cp.MOSEK,verbose=False)


        if not prob.status == 'optimal':
            raise ValueError(f'Optimizer did not find a solution: {prob.status}')

        logger.info(
            f'SDP status: {prob.status} optimal gamma: {np.sqrt(ga.value)}\n'
            f'Max real eig (M_theta): {max(np.real(np.linalg.eig(M.value)[0]))}'
        )
        
        logger.info('Write back projected parameters.')
        if self.multiplier_type == 'diagonal':
            self.lam.data = (
                torch.tensor(np.diag(np.array(Lambda.value))).double().to(device)
            )
        elif self.multiplier_type == 'static_zf':
            self.lam.data = torch.tensor(Lambda.value).double().to(device)
        self.X_cal.data = torch.tensor(X_cal.value).double().to(device)


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

        if self.multiplier_type == 'diagonal':
            Lambda = torch.diag(self.lam).to(self.device)
        elif self.multiplier_type == 'static_zf':
            Lambda = self.lam.to(self.device)

        sim_parameter = SimAbcdParameter(
            theta.cpu().detach().numpy(),
            self.X_cal.cpu().detach().numpy(),
            Lambda.cpu().detach().numpy()
        )

        return (sim_parameter, generalized_plant.cpu().detach().numpy())


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

    def get_barriers(self, t: torch.Tensor) -> torch.Tensor:
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
            self.X_cal
        ]

        barrier = torch.tensor(0.0).to(self.device)
        for constraint in constraints:
            barrier += -t * utils.get_logdet(constraint).to(self.device)

        return barrier

    def get_constraints(self) -> torch.Tensor:
        nx, nu, nz, ny, nw, ne, nd = self.nx, self.nu, self.nz, self.ny, self.nw, self.ne, self.nd
        nxi = nx+ nx
        generalized_plant = self.S_s + self.S_l @ self.theta @ self.S_r
        (
            A_cal,
            B_cal,
            B2_cal,
            C_cal,
            D_cal,
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

        L1 = utils.torch_bmat([
            [torch.eye(nxi), torch.zeros((nxi,nd+nw))],
            [A_cal, B_cal, B2_cal]
        ])
        L2 = utils.torch_bmat([
            [torch.zeros((nd,nxi)), torch.eye(nd), torch.zeros((nd,nw))],
            [C_cal, D_cal, D12_cal]
        ])
        L3 = utils.torch_bmat([
            [torch.zeros((nw,nxi+nd)), torch.eye(nw)],
            [C2_cal, D21_cal, D22_cal]
        ])

        if self.multiplier_type == 'diagonal':
            Lambda = torch.diag(self.lam).to(self.device)
        elif self.multiplier_type == 'static_zf':
            Lambda = self.lam.to(self.device)

        ga = (self.gamma*self.increase_constraints) **2
        M = L1.T @ utils.torch_bmat([[-self.X_cal, torch.zeros((nxi,nxi))], [torch.zeros((nxi,nxi)), self.X_cal]]) @ L1 + \
            L2.T @ utils.torch_bmat([[-ga * torch.eye(nd), torch.zeros((nd,ne))], [torch.zeros((ne,nd)), torch.eye(ne)]]) @ L2 + \
            L3.T @ utils.torch_bmat([[-(Lambda+Lambda.T), self.beta*Lambda],[self.beta*Lambda.T, torch.zeros((nz,nz))]])@L3


        if self.multiplier_type == 'diagonal':
            # https://yalmip.github.io/faq/semidefiniteelementwise/
            # symmetrize variable
            return 0.5 * (M + M.T)
        else:
            return M
         
    def write_parameters(self, params: List[torch.Tensor]) -> None:
        for old_par, new_par in zip(params, self.parameters()):
            new_par.data = old_par.clone()


    def check_constraints(self) -> bool:
        with torch.no_grad():
            P = self.get_constraints()
            _, info = torch.linalg.cholesky_ex(-P)
        return True if info == 0 else False
   
    