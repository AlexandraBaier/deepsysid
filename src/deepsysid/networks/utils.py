from typing import List, Union
from numpy.typing import NDArray

import cvxpy as cp
import numpy as np


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

def get_distance_constraints(klmn_0: NDArray[np.float64], klmn: cp.Variable, d: Union[cp.Variable, np.float64]) -> List:
    return [cp.norm(klmn_0 - klmn) <= d]