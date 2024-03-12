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

def get_coupling_matrices(
        X:torch.Tensor,
        Y:torch.Tensor
        # nx:int
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # L_x = construct_lower_triangular_matrix(
        #     L_flat=L_x_flat, diag_length=nx
        # )
        # L_y = construct_lower_triangular_matrix(
        #     L_flat=L_y_flat, diag_length=nx
        # )

        # X = L_x @ L_x.T
        # Y = L_y @ L_y.T

        # 2. Determine non-singular U,V with V U^T = I - Y X
        U = torch.linalg.inv(Y) - X
        V = Y

        return (X, Y, U, V)


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



