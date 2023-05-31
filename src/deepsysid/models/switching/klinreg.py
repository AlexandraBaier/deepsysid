"""
KLinearRegressionARXModel is implemented based on
    F. Lauer, Estimating the probability of success
    of a simple algorithm for switched linear regression,
    Nonlinear Analysis: Hybrid Systems, 8:31-47, 2013
"""

import dataclasses
import logging
import pickle
from typing import Optional, Tuple, Callable

import h5py
import numpy as np
from numpy.typing import NDArray
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics import euclidean_distances
from sklearn.multioutput import MultiOutputRegressor
from sklearn.utils.validation import check_array, check_X_y

from ...tracker.base import EventData
from deepsysid.models.base import DynamicIdentificationModelConfig, FixedWindowModel

logger = logging.getLogger(__name__)


class KLinearRegressionARXModelConfig(DynamicIdentificationModelConfig):
    lag: int
    n_modes: int
    probability_failure: float = 0.001
    initialization_bound: float = 100.0
    zero_probability_restarts: int = 100
    use_max_restarts: bool = False


class KLinearRegressionARXModel(FixedWindowModel[MultiOutputRegressor]):
    CONFIG = KLinearRegressionARXModelConfig

    def __init__(self, config: KLinearRegressionARXModelConfig):
        super().__init__(
            window_size=config.lag,
            regressor=MultiOutputRegressor(
                KLinearRegression(
                    n_modes=config.n_modes,
                    probability_failure=config.probability_failure,
                    initialization_bound=config.initialization_bound,
                    zero_probability_restarts=config.zero_probability_restarts,
                    use_max_restarts=config.use_max_restarts,
                )
            ),
        )

        self.n_modes = config.n_modes
        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)
        self.lag = config.lag

    def save(
        self,
        file_path: Tuple[str, ...],
        tracker: Callable[[EventData], None] = lambda _: None,
    ) -> None:
        if (
            self.state_mean is None
            or self.state_stddev is None
            or self.control_mean is None
            or self.control_stddev is None
        ):
            raise ValueError('Model needs to be trained prior to save.')

        for regressor in self.regressor.estimators_:
            if (
                regressor.weights_ is None
                or regressor.mode_estimates_ is None
                or regressor.best_cost_ is None
                or regressor.n_restarts_ is None
                or regressor.cost_per_restart_ is None
                or regressor.weights_per_restart_ is None
                or regressor.initial_weights_per_restart_ is None
                or regressor.mode_centroids_ is None
            ):
                raise ValueError('Model needs to be trained prior to save.')

        with open(file_path[0], mode='wb') as f:
            pickle.dump(self.regressor, f)

        with h5py.File(file_path[1], 'w') as f:
            f.create_dataset('control_mean', data=self.control_mean)
            f.create_dataset('control_stddev', data=self.control_stddev)
            f.create_dataset('state_mean', data=self.state_mean)
            f.create_dataset('state_stddev', data=self.state_stddev)

    def load(self, file_path: Tuple[str, ...]) -> None:
        with open(file_path[0], mode='rb') as f:
            self.regressor = pickle.load(f)

        with h5py.File(file_path[1], 'r') as f:
            self.control_mean = f['control_mean'][:]
            self.control_stddev = f['control_stddev'][:]
            self.state_mean = f['state_mean'][:]
            self.state_stddev = f['state_stddev'][:]

    def get_file_extension(self) -> Tuple[str, ...]:
        return ('pickle', 'hdf5')

    def get_parameter_count(self) -> int:
        return self.n_modes * (
            self.state_dim
            * (self.control_dim + self.lag * (self.state_dim + self.control_dim))
        )


@dataclasses.dataclass
class KLinRegOutput:
    weights: NDArray[np.float64]
    mode_estimates: NDArray[np.int64]
    best_error: float
    n_restarts: int
    error_per_restart: NDArray[np.float64]
    weights_per_restart: NDArray[np.float64]
    initial_weights_per_attempt: NDArray[np.float64]


class KLinearRegression(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        n_modes: int = 2,
        probability_failure: float = 0.001,
        initialization_bound: float = 100.0,
        zero_probability_restarts: int = 100,
        use_max_restarts: bool = False,
    ) -> None:
        self.n_modes = n_modes
        self.probability_failure = probability_failure
        self.initialization_bound = initialization_bound
        self.zero_probability_restarts = zero_probability_restarts
        self.use_max_restarts = use_max_restarts

        self.weights_: Optional[NDArray[np.float64]] = None
        self.mode_estimates_: Optional[NDArray[np.int64]] = None
        self.best_cost_: Optional[float] = None
        self.n_restarts_: Optional[int] = None
        self.cost_per_restart_: Optional[NDArray[np.float64]] = None
        self.weights_per_restart_: Optional[NDArray[np.float64]] = None
        self.initial_weights_per_restart_: Optional[NDArray[np.float64]] = None
        self.mode_centroids_: Optional[NDArray[np.float64]] = None

    def fit(
        self, X: NDArray[np.float64], y: NDArray[np.float64]
    ) -> 'KLinearRegression':
        X, y = check_X_y(X, y, y_numeric=True, force_all_finite=True)
        output = self._fit(
            X=X,
            y=y,
        )
        self.weights_ = output.weights
        self.mode_estimates_ = output.mode_estimates
        self.best_cost_ = output.best_error
        self.n_restarts_ = output.n_restarts
        self.cost_per_restart_ = output.error_per_restart
        self.weights_per_restart_ = output.weights_per_restart
        self.initial_weights_per_restart_ = output.initial_weights_per_attempt

        self.mode_centroids_ = np.zeros((self.n_modes, X.shape[1]))
        for mode in range(self.n_modes):
            self.mode_centroids_[mode, :] = np.mean(
                X[self.mode_estimates_ == mode], axis=0
            )

        return self

    def predict(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.weights_ is None or self.mode_centroids_ is None:
            raise ValueError('Model has to be trained prior to predict.')

        check_array(X)

        distances = euclidean_distances(X, self.mode_centroids_)
        mode_idx = np.argmin(distances, axis=1)
        y = X @ self.weights_
        y = y[np.arange(X.shape[0]), mode_idx]
        return y

    def _fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> KLinRegOutput:
        n_samples, input_dim = X.shape
        if not self.use_max_restarts:
            n_restarts = self.get_restarts(
                n_samples,
                input_dim,
            )
        else:
            n_restarts = self.zero_probability_restarts

        logger.info(f'Running {n_restarts} restarts to identify kLinReg.')

        y = y[:, np.newaxis]

        weights_per_restart = np.zeros((n_restarts * input_dim, self.n_modes))
        initial_weights_per_restart = np.zeros((n_restarts * input_dim, self.n_modes))
        error_per_restart = np.zeros((n_restarts,))

        for restart_idx in range(n_restarts):
            weights = (
                2
                * self.initialization_bound
                * np.random.random((input_dim, self.n_modes))
                - self.initialization_bound
            )
            initial_weights_per_restart[
                restart_idx * input_dim : (restart_idx + 1) * input_dim, :
            ] = weights

            has_good_clustering = True
            estimated_modes_old = np.zeros((n_samples,))
            estimated_modes = -np.ones((n_samples,))

            while (
                has_good_clustering
                and np.sum(estimated_modes == estimated_modes_old) < n_samples
            ):
                error_residual = y - X @ weights
                estimated_modes_old = estimated_modes
                estimated_modes = np.argmin(error_residual * error_residual, axis=1)

                for mode_idx in range(self.n_modes):
                    if np.sum(estimated_modes == mode_idx) > 2:
                        solution, _, _, _ = np.linalg.lstsq(
                            X[estimated_modes == mode_idx, :],
                            y[estimated_modes == mode_idx],
                            rcond=None,
                        )
                        weights[:, mode_idx] = solution.squeeze()
                    else:
                        has_good_clustering = False
                        break

            if has_good_clustering:
                weights_per_restart[
                    restart_idx * input_dim : (restart_idx + 1) * input_dim, :
                ] = weights
                error_residual = y - X @ weights
                minimal_error = np.min(error_residual * error_residual, axis=1)
                error_per_restart[restart_idx] = np.sum(minimal_error)
            else:
                error_per_restart[restart_idx] = np.inf

        best_error = np.min(error_per_restart)
        restart_idx = int(np.argmin(error_per_restart))
        weights = weights_per_restart[
            restart_idx * input_dim : (restart_idx + 1) * input_dim, :
        ]
        error_residual = y - X @ weights
        estimated_modes = np.argmin(error_residual * error_residual, axis=1)

        return KLinRegOutput(
            weights=weights,
            mode_estimates=estimated_modes.astype(np.int64),
            best_error=float(best_error),
            n_restarts=n_restarts,
            error_per_restart=error_per_restart,
            weights_per_restart=weights_per_restart,
            initial_weights_per_attempt=initial_weights_per_restart,
        )

    def get_restarts(
        self,
        n_samples: int,
        input_dim: int,
    ) -> int:
        probability_success = self.get_success_probability(n_samples, input_dim)
        if probability_success > 0:
            n_restarts = int(
                np.ceil(
                    np.log(self.probability_failure) / np.log(1 - probability_success)
                )
            )
        else:
            n_restarts = self.zero_probability_restarts

        if n_restarts < 1:
            n_restarts = 1

        return n_restarts

    def get_success_probability(
        self,
        n_samples: int,
        input_dim: int,
    ) -> float:
        k = 1.02 - 0.023 * self.n_modes
        t = 52 * np.sqrt(2**self.n_modes * input_dim) - 220
        tau = 1.93 * 2**self.n_modes * input_dim - 37

        probability_success = max(0, k * (1 - np.exp(-(n_samples - tau) / t)))
        return float(probability_success)
