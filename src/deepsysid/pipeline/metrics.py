import abc
from typing import Dict, List, Tuple, Type

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.metrics import mean_absolute_error, mean_squared_error


class BaseMetricConfig(BaseModel):
    state_names: List[str]
    sample_time: float


class BaseMetric(metaclass=abc.ABCMeta):
    CONFIG: Type[BaseMetricConfig] = BaseMetricConfig

    def __init__(self, config: BaseMetricConfig):
        pass

    @abc.abstractmethod
    def measure(
        self, y_true: List[NDArray[np.float64]], y_pred: List[NDArray[np.float64]]
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        """
        :param y_true: List of NDArrays of shape (time, state).
        :param y_pred: List of NDArrays of shape (time, state).
        :return: Tuple of the (primary) metric with shape (state,)
            or (1,) and supporting data. For example, mean error
            per state over all time steps as first element while
            the dictionary contains standard deviation  of the
            error per state variable and the mean error per time step.
        """
        pass


class AverageOverSequenceMetric(BaseMetric, metaclass=abc.ABCMeta):
    def measure(
        self,
        y_true: List[NDArray[np.float64]],
        y_pred: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        if len(y_true) != len(y_pred):
            raise ValueError(
                'Lists y_true and y_pred and steps need to have the same length.'
            )
        if not all(t.shape == p.shape for t, p in zip(y_true, y_pred)):
            raise ValueError(
                'Shapes of pairwise elements in y_true and y_pred have to match.'
            )
        if not all(
            len(t.shape) == 2 and len(p.shape) == 2 for t, p in zip(y_true, y_pred)
        ):
            raise ValueError('y_true and y_pred have to be 2-dimensional arrays.')

        steps = [p.shape[0] for p in y_pred]
        scores = self.score_over_sequences(y_true, y_pred)
        return np.average(scores, weights=steps, axis=0), dict(per_step=scores)

    def score_over_sequences(
        self, true_seq: List[NDArray[np.float64]], pred_seq: List[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        score = np.zeros((len(pred_seq), pred_seq[0].shape[1]), dtype=np.float64)
        for i, (true, pred) in enumerate(zip(true_seq, pred_seq)):
            score[i, :] = self.score_per_sequence(true, pred)
        return score

    @abc.abstractmethod
    def score_per_sequence(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        pass


class IndexOfAgreementMetricConfig(BaseMetricConfig):
    j: int


class IndexOfAgreementMetric(AverageOverSequenceMetric):
    CONFIG: Type[BaseMetricConfig] = IndexOfAgreementMetricConfig

    def __init__(self, config: IndexOfAgreementMetricConfig):
        super().__init__(config)

        if config.j < 1:
            raise ValueError('Exponent j needs to be larger than 0.')

        self.j = config.j

    def score_per_sequence(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        error_sum = np.sum(np.power(np.abs(y_true - y_pred), self.j), axis=0)
        partial_diff_true = np.abs(y_true - np.mean(y_true, axis=0))
        partial_diff_pred = np.abs(y_pred - np.mean(y_true, axis=0))
        partial_diff_sum = np.sum(
            np.power(partial_diff_true + partial_diff_pred, self.j), axis=0
        )
        return 1 - (error_sum / partial_diff_sum)


class MeanSquaredErrorMetric(AverageOverSequenceMetric):
    def score_per_sequence(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return mean_squared_error(y_true, y_pred, multioutput='raw_values')


class RootMeanSquaredErrorMetric(AverageOverSequenceMetric):
    def score_per_sequence(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return np.sqrt(mean_squared_error(y_true, y_pred, multioutput='raw_values'))


class MeanAbsoluteErrorMetric(AverageOverSequenceMetric):
    def score_per_sequence(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        return mean_absolute_error(y_true, y_pred, multioutput='raw_values')


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

        xdot = state[:, name2idx['vx']]
        ydot = state[:, name2idx['vy']]
        zdot = state[:, name2idx['vz']]

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


def retrieve_metric_class(metric_class_string: str) -> Type[BaseMetric]:
    # https://stackoverflow.com/a/452981
    parts = metric_class_string.split('.')
    module_string = '.'.join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, BaseMetric):
        raise ValueError(f'{cls} is not a subclass of BaseMetric.')
    return cls  # type: ignore


def euler_method(
    ic: NDArray[np.float64], dx: NDArray[np.float64], dt: float
) -> NDArray[np.float64]:
    return ic + dx * dt
