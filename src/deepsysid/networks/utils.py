from typing import List, Union, Tuple
from numpy.typing import NDArray

import cvxpy as cp
import numpy as np
import torch


def get_feasibility_constraint(
    P: cp.Variable, t: Union[cp.Variable, np.float64]
) -> List:
    nP = P.shape[0]
    return [P << t * np.eye(nP)]


def get_bounding_inequalities(
    X: cp.Variable,
    Y: cp.Variable,
    KLMN: cp.Variable,
    alpha: Union[cp.Variable, np.float64],
) -> List:
    nx = X.shape[0]
    cols, rows = KLMN.shape
    constraints = [(X << alpha * np.eye(nx))]
    constraints.append(Y << alpha * np.eye(nx))
    constraints.append(
        cp.bmat(
            [
                [alpha * np.eye(cols), KLMN],
                [
                    KLMN.T,
                    alpha * np.eye(rows),
                ],
            ]
        )
        >> 0
    )
    return constraints


def get_conditioning_constraints(
    Y: cp.Variable, X: cp.Variable, beta: Union[cp.Variable, np.float64]
) -> List:
    nx = X.shape[0]
    constraints = [cp.bmat([[Y, beta * np.eye(nx)], [beta * np.eye(nx), X]]) >> 0]
    return constraints


def get_distance_constraints(
    klmn_0: NDArray[np.float64], klmn: cp.Variable, d: Union[cp.Variable, np.float64]
) -> List:
    return [cp.norm(klmn_0 - klmn) <= d]

def construct_lower_triangular_matrix(
    L_flat: torch.Tensor, diag_length: int
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
    L: NDArray[np.float64]
) -> NDArray[np.float64]:
    diag_length = L.shape[0]
    vector_list = []
    for diag_idx in range(0, -diag_length, -1):
        vector_list.append(np.diag(L, k=diag_idx))

    return np.hstack(vector_list)




def get_cal_matrices(
    gen_plant: torch.Tensor,
    nxi: int, #state
    nd:int, # input
    ne:int, # output
    nz:int, # uncertainty
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
        A_cal = gen_plant[: nxi, : nxi]
        B1_cal = gen_plant[:nxi, nxi:nxi+nd]
        B2_cal = gen_plant[:nxi, nxi+nd:]

        C1_cal = gen_plant[nxi:nxi+ne, : nxi]
        D11_cal = gen_plant[nxi:nxi+ne, nxi:nxi+nd]
        D12_cal = gen_plant[nxi:nxi+ne, nxi+nd:]

        C2_cal = gen_plant[nxi+ne:, : nxi]
        D21_cal = gen_plant[nxi+ne:, nxi:nxi+nd]
        D22_cal = gen_plant[nxi+ne:, nxi+nd:]

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


def get_logdet(mat: torch.Tensor) -> torch.Tensor:
    # return logdet of matrix mat, if it is not positive semi-definite, return inf
    if len(mat.shape) < 2:
        return torch.log(mat)

    _, info = torch.linalg.cholesky_ex(mat.cpu())

    if info > 0:
        logdet = torch.tensor(float('inf'))
    else:
        logdet = (mat.logdet())

    return logdet

def bmat(mat: List[List[NDArray[np.float64]]]) -> NDArray[np.float64]:
    mat_list = []
    for col in mat:
        mat_list.append(np.hstack(col))
    return np.vstack(mat_list)


def torch_bmat(mat: List[List[torch.Tensor]]) -> torch.Tensor:
    mat_list = []
    for col in mat:
        mat_list.append(torch.hstack(col))
    return torch.vstack(mat_list)



def get_layer_parameters_lstm(weight_ih, weight_hh, bias_ih, bias_hh, h):
    W_ii, W_if, W_ig, W_io = torch.split(weight_ih, h, dim=0)
    W_hi, W_hf, W_hg, W_ho = torch.split(weight_hh, h, dim=0)
    b_ii, b_if, b_ig, b_io = torch.split(bias_ih, h, dim=0)
    b_hi, b_hf, b_hg, b_ho = torch.split(bias_hh, h, dim=0)
    return (W_ii, W_if, W_ig, W_io, W_hi, W_hf, W_hg, W_ho, b_ii, b_if, b_ig, b_io, b_hi, b_hf, b_hg, b_ho)

def get_iss_parameter_layer_lstm(weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0, h):
    (W_ii, W_if, W_ig, W_io, W_hi, W_hf, W_hg, W_ho, b_ii, b_if, b_ig, b_io, b_hi, b_hf, b_hg, b_ho) = get_layer_parameters_lstm(
            weight_ih_l0,
            weight_hh_l0,
            bias_ih_l0,
            bias_hh_l0,
            h
        )
    W_f, U_f, b_f = W_if, W_hf, (b_if + b_hf).reshape(-1,1)
    W_i, U_i, b_i = W_ii, W_hi, (b_ii + b_hi).reshape(-1,1)
    W_c, U_c, b_c = W_ig, W_hg, (b_ig + b_hg).reshape(-1,1)
    W_o, U_o, b_o = W_io, W_ho, (b_io + b_ho).reshape(-1,1)
    return (W_f, U_f, b_f), (W_i, U_i, b_i), (W_c, U_c, b_c), (W_o, U_o, b_o)

def check_iss_lstm(W_fs, W_is, W_cs, W_os):
    # print(f'W_o = {W_os[0].shape}, U_o = {W_os[1].shape}, b_o = {W_os[2].shape}')
    # print(f'|U_c|_1 = {torch.linalg.norm(W_cs[1],ord=1)}')

    s_f = torch.sigmoid(torch.linalg.norm(torch.hstack(W_fs),ord=np.inf))
    s_z = torch.sigmoid(torch.linalg.norm(torch.hstack(W_os),ord=np.inf))
    s_i = torch.sigmoid(torch.linalg.norm(torch.hstack(W_is),ord=np.inf))

    iss_cond = s_f + s_z * s_i * torch.linalg.norm(W_cs[1],ord=1) -1

    # iss_cond_1 = (1+torch.sigmoid(torch.linalg.norm(torch.hstack(W_os),ord=np.inf)))*(torch.sigmoid(torch.linalg.norm(torch.hstack(W_fs),ord=np.inf)))
    # iss_cond_2 = (1+torch.sigmoid(torch.linalg.norm(torch.hstack(W_os),ord=np.inf)))*(torch.sigmoid(torch.linalg.norm(torch.hstack(W_is),ord=np.inf)))*torch.linalg.norm(W_cs[1],ord=1)
    # print(iss_cond_1, iss_cond_2)
    # return (iss_cond_1, iss_cond_2),(iss_cond_1 < 1) & (iss_cond_2 <1)
    return iss_cond, iss_cond < 0



