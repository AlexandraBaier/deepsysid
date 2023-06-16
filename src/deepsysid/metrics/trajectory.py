from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .base import BaseMetric, BaseMetricConfig, validate_measure_arguments


class Trajectory4DOFRootMeanSquaredErrorMetric(BaseMetric):
    """
    Computes trajectory in the horizontal plane given velocities in BODY coordinates
    and roll angle in NED coordinates.
    Expects state variables with the names u, v, r, phi to exist.
        u = velocity along x-axis
        v = velocity along y-axis
        r = rate around z-axis
        phi = angle around x-axis
    """

    def __init__(self, config: BaseMetricConfig):
        super().__init__(config)
        self.state_names = config.state_names
        self.sample_time = config.sample_time

    @validate_measure_arguments
    def measure(
        self, y_true: List[NDArray[np.float64]], y_pred: List[NDArray[np.float64]]
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        traj_rmse_per_step_seq: List[NDArray[np.float64]] = []
        for pred_state, true_state in zip(y_pred, y_true):
            pred_x, pred_y, _, _ = self._compute_trajectory(pred_state)
            true_x, true_y, _, _ = self._compute_trajectory(true_state)
            traj_rmse_per_step = np.sqrt(
                (pred_x - true_x) ** 2 + (pred_y - true_y) ** 2
            )
            traj_rmse_per_step_seq.append(traj_rmse_per_step)

        seq = np.array(traj_rmse_per_step_seq, dtype=np.float64)
        traj_rmse = np.array([float(np.mean(np.concatenate(seq)))], dtype=np.float64)
        traj_stddev = np.array([float(np.std(np.concatenate(seq)))], dtype=np.float64)
        n_samples = np.array([np.concatenate(seq).size], dtype=np.float64)

        return traj_rmse, dict(std=traj_stddev, n_samples=n_samples, error_per_step=seq)

    def _compute_trajectory(
        self,
        state: NDArray[np.float64],
        x0: float = 0.0,
        y0: float = 0.0,
        psi0: float = 0.0,
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        name2idx = dict((name, idx) for idx, name in enumerate(self.state_names))

        u = state[:, name2idx['u']]
        v = state[:, name2idx['v']]
        r = state[:, name2idx['r']]
        phi = state[:, name2idx['phi']]

        n_steps = u.shape[0]

        psi = np.zeros(n_steps, dtype=np.float64)
        psi[0] = psi0
        x = np.zeros(n_steps, dtype=np.float64)
        x[0] = x0
        y = np.zeros(n_steps, dtype=np.float64)
        y[0] = y0

        for i in range(1, n_steps):
            psi[i] = euler_method(psi[i - 1], r[i], self.sample_time)

            xd = np.cos(psi[i]) * u[i] - (np.sin(psi[i]) * np.cos(phi[i]) * v[i])
            yd = np.sin(psi[i]) * u[i] + (np.cos(psi[i]) * np.cos(phi[i]) * v[i])

            x[i] = euler_method(x[i - 1], xd, self.sample_time)
            y[i] = euler_method(y[i - 1], yd, self.sample_time)

        return x, y, phi, psi


class TrajectoryNED6DOFRootMeanSquaredErrorMetric(BaseMetric):
    def __init__(self, config: BaseMetricConfig):
        super().__init__(config)
        self.state_names = config.state_names
        self.sample_time = config.sample_time

    @validate_measure_arguments
    def measure(
        self, y_true: List[NDArray[np.float64]], y_pred: List[NDArray[np.float64]]
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        traj_rmse_per_step_seq: List[NDArray[np.float64]] = []
        for pred_state, true_state in zip(y_pred, y_true):
            pred_x, pred_y, pred_z = self._compute_trajectory(pred_state)
            true_x, true_y, true_z = self._compute_trajectory(true_state)
            traj_rmse_per_step = np.sqrt(
                (pred_x - true_x) ** 2 + (pred_y - true_y) ** 2 + (pred_z - true_z) ** 2
            )
            traj_rmse_per_step_seq.append(traj_rmse_per_step)

        seq = np.array(traj_rmse_per_step_seq, dtype=np.float64)
        traj_rmse = np.array([float(np.mean(np.concatenate(seq)))], dtype=np.float64)
        traj_stddev = np.array([float(np.std(np.concatenate(seq)))], dtype=np.float64)
        n_samples = np.array([np.concatenate(seq).size], dtype=np.float64)

        return traj_rmse, dict(std=traj_stddev, n_samples=n_samples, error_per_step=seq)

    def _compute_trajectory(
        self, state: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        name2idx = dict((name, idx) for idx, name in enumerate(self.state_names))

        xdot = state[:, name2idx['dx']]
        ydot = state[:, name2idx['dy']]
        zdot = state[:, name2idx['dz']]

        shape = xdot.shape

        x = np.zeros(shape, dtype=np.float64)
        y = np.zeros(shape, dtype=np.float64)
        z = np.zeros(shape, dtype=np.float64)
        x[0] = 0.0
        y[0] = 0.0
        z[0] = 0.0

        for i in range(1, shape[0]):
            x[i] = euler_method(x[i - 1], xdot[i], self.sample_time)
            y[i] = euler_method(y[i - 1], ydot[i], self.sample_time)
            z[i] = euler_method(z[i - 1], zdot[i], self.sample_time)

        return x, y, z


def euler_method(
    ic: NDArray[np.float64], dx: NDArray[np.float64], dt: float
) -> NDArray[np.float64]:
    return ic + dx * dt
