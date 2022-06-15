from typing import Callable, List, Tuple

import numpy as np


def mean_stddev(array_seq: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(np.vstack(array_seq), axis=0)
    stddev = np.std(np.vstack(array_seq), axis=0)
    return mean, stddev


def denormalize(x, mean, stddev):
    return x * stddev + mean


def normalize(x, mean, stddev):
    return (x - mean) / stddev


def sliding_window(arr: np.ndarray, width: int) -> np.ndarray:
    # non-overlapping windows
    return np.hstack([arr[i : 1 + i - width or None : width] for i in range(width)])


def transform_to_single_step_training_data(
    control: np.ndarray, state: np.ndarray, window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    full_dim = control.shape[1] + state.shape[1]
    control_dim = control.shape[1]

    rows = np.hstack((control, state))
    windows = sliding_window(rows, window_size + 1)
    # x = [u(t-W) x(t-W) ... u(t-1) x(t-1) u(t)]
    x = windows[:, : full_dim * window_size + control_dim]
    # y = x(t)
    y = windows[:, full_dim * window_size + control_dim :]

    return x, y


def euler_method(ic, dx, dt):
    return ic + dx * dt


def two_step_adam_bashford(ic, dx0, dx1, dt):
    return ic + 1.5 * dt * dx1 - 0.5 * dt * dx0


def coord2angle(cos_alpha, sin_alpha):
    return np.arctan(sin_alpha / cos_alpha)


def compute_trajectory_4dof(
    state: np.ndarray,
    state_names: List[str],
    sample_time: float,
    x0=0.0,
    y0=0.0,
    psi0=0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    name2idx = dict((name, idx) for idx, name in enumerate(state_names))

    u = state[:, name2idx['u']]
    v = state[:, name2idx['v']]
    r = state[:, name2idx['r']]
    phi = state[:, name2idx['phi']]

    shape = u.shape

    psi = np.zeros(shape)
    psi[0] = psi0
    x = np.zeros(shape)
    x[0] = x0
    y = np.zeros(shape)
    y[0] = y0
    old_xd = None
    old_yd = None

    for i in range(1, shape[0]):
        if i == 1:
            psi[i] = euler_method(psi[i - 1], r[i], sample_time)
        else:
            psi[i] = two_step_adam_bashford(psi[i - 1], r[i - 1], r[i], sample_time)

        xd = np.cos(psi[i]) * u[i] - (np.sin(psi[i]) * np.cos(phi[i]) * v[i])
        yd = np.sin(psi[i]) * u[i] + (np.cos(psi[i]) * np.cos(phi[i]) * v[i])

        if i == 1:
            x[i] = euler_method(x[i - 1], xd, sample_time)
            y[i] = euler_method(y[i - 1], yd, sample_time)
        else:
            x[i] = two_step_adam_bashford(x[i - 1], old_xd, xd, sample_time)
            y[i] = two_step_adam_bashford(y[i - 1], old_yd, yd, sample_time)

        old_xd = xd
        old_yd = yd

    return x, y, phi, psi


def velocity2speed(state: np.ndarray, state_names: List[str]) -> np.ndarray:
    u = state[:, state_names.index('u')]
    v = state[:, state_names.index('v')]
    return np.sqrt(np.multiply(u, u) + np.multiply(v, v)).reshape(-1, 1)


def index_of_agreement(true: np.ndarray, pred: np.ndarray, j=1) -> np.ndarray:
    error_sum = np.sum(np.power(np.abs(true - pred), j), axis=0)
    partial_diff_true = np.abs(true - np.mean(true, axis=0))
    partial_diff_pred = np.abs(pred - np.mean(true, axis=0))
    partial_diff_sum = np.sum(
        np.power(partial_diff_true + partial_diff_pred, j), axis=0
    )
    return 1 - (error_sum / partial_diff_sum)


def score_on_sequence(
    true_seq: List[np.ndarray],
    pred_seq: List[np.ndarray],
    score_fnc: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    score = np.zeros((len(pred_seq), pred_seq[0].shape[1]))
    for i, (true, pred) in enumerate(zip(true_seq, pred_seq)):
        score[i, :] = score_fnc(true, pred)
    return score
