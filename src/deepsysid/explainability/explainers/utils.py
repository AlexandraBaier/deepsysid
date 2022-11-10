from typing import Tuple

import numpy as np
from numpy.typing import NDArray


def denormalize_state_weights(
    state_mean: NDArray[np.float64],
    state_std: NDArray[np.float64],
    state_matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:

    state_dim = state_mean.shape[0]

    state_matrix_den = np.zeros(state_matrix.shape)
    intercept = np.zeros((state_dim,))
    for out_idx in range(state_dim):
        for in_idx in range(state_dim):
            weight = (
                state_std[out_idx] / state_std[in_idx] * state_matrix[out_idx, in_idx]
            )
            state_matrix_den[out_idx, in_idx] = weight
            intercept[out_idx] = intercept[out_idx] - weight * state_mean[in_idx]

    return state_matrix_den, intercept


def denormalize_control_weights(
    state_std: NDArray[np.float64],
    control_mean: NDArray[np.float64],
    control_std: NDArray[np.float64],
    control_matrix: NDArray[np.float64],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    state_dim = control_matrix.shape[0]
    control_dim = control_matrix.shape[1]

    control_matrix_den = np.zeros(control_matrix.shape)
    intercept = np.zeros((state_dim,))
    for out_idx in range(state_dim):
        for in_idx in range(control_dim):
            weight = (
                state_std[out_idx]
                / control_std[in_idx]
                * control_matrix[out_idx, in_idx]
            )
            control_matrix_den[out_idx, in_idx] = weight
            intercept[out_idx] = intercept[out_idx] - weight * control_mean[in_idx]

    return control_matrix_den, intercept
