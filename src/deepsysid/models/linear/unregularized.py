from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LinearRegression

from ...tracker.base import BaseEventTracker
from .. import utils
from ..base import DynamicIdentificationModel, DynamicIdentificationModelConfig
from .arx import (
    KernelRegression,
    construct_fit_input_arguments,
    construct_predict_input_arguments_for_single_sample,
)
from .kernel import KernelHyperparameter, ZeroKernel


class LinearModel(DynamicIdentificationModel):
    def __init__(self, config: DynamicIdentificationModelConfig):
        super().__init__(config)

        self.regressor = LinearRegression(fit_intercept=True)

        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)

        self.control_mean: Optional[NDArray[np.float64]] = None
        self.control_stddev: Optional[NDArray[np.float64]] = None
        self.state_mean: Optional[NDArray[np.float64]] = None
        self.state_stddev: Optional[NDArray[np.float64]] = None

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]] = None,
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> None:
        assert len(control_seqs) == len(state_seqs)
        assert control_seqs[0].shape[0] == state_seqs[0].shape[0]
        assert control_seqs[0].shape[1] == self.control_dim
        assert state_seqs[0].shape[1] == self.state_dim

        self.control_mean, self.control_stddev = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_stddev = utils.mean_stddev(state_seqs)

        train_x_list, train_y_list = [], []
        for i in range(len(control_seqs)):
            control = control_seqs[i]
            state = state_seqs[i]

            control = utils.normalize(control, self.control_mean, self.control_stddev)
            state = utils.normalize(state, self.state_mean, self.state_stddev)

            x = np.hstack((control[1:], state[:-1]))
            y = state[1:]

            train_x_list.append(x)
            train_y_list.append(y)

        train_x = np.vstack(train_x_list)
        train_y = np.vstack(train_y_list)

        self.regressor.fit(train_x, train_y)

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        x0: Optional[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        assert initial_control.shape[1] == self.control_dim
        assert initial_state.shape[1] == self.state_dim
        assert control.shape[1] == self.control_dim

        if (
            self.state_mean is None
            or self.state_stddev is None
            or self.control_mean is None
            or self.control_stddev is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        control = utils.normalize(control, self.control_mean, self.control_stddev)
        state = utils.normalize(initial_state, self.state_mean, self.state_stddev)[-1]

        pred_states = np.array(
            [
                np.squeeze(
                    self.regressor.predict(
                        np.concatenate((control[i], state)).reshape(1, -1)
                    )
                )
                for i in range(control.shape[0])
            ],
            dtype=np.float64,
        )

        pred_states = utils.denormalize(pred_states, self.state_mean, self.state_stddev)

        return pred_states

    def save(
        self,
        file_path: Tuple[str, ...],
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> None:
        with h5py.File(file_path[0], 'w') as f:
            f.create_dataset('coef_', data=self.regressor.coef_)
            f.create_dataset('intercept_', data=self.regressor.intercept_)

            f.create_dataset('control_mean', data=self.control_mean)
            f.create_dataset('control_stddev', data=self.control_stddev)
            f.create_dataset('state_mean', data=self.state_mean)
            f.create_dataset('state_stddev', data=self.state_stddev)

    def load(self, file_path: Tuple[str, ...]) -> None:
        with h5py.File(file_path[0], 'r') as f:
            self.regressor.coef_ = f['coef_'][:].astype(np.float64)
            self.regressor.intercept_ = f['intercept_'][:].astype(np.float64)

            self.control_mean = f['control_mean'][:].astype(np.float64)
            self.control_stddev = f['control_stddev'][:].astype(np.float64)
            self.state_mean = f['state_mean'][:].astype(np.float64)
            self.state_stddev = f['state_stddev'][:].astype(np.float64)

    def get_file_extension(self) -> Tuple[str, ...]:
        return ('hdf5',)

    def get_parameter_count(self) -> int:
        if (
            self.state_mean is None
            or self.state_stddev is None
            or self.control_mean is None
            or self.control_stddev is None
        ):
            return (self.control_dim + self.state_dim) * self.state_dim + self.state_dim

        count = 0
        if len(self.regressor.coef_.shape) == 2:
            count += np.product(self.regressor.coef_.shape)
        else:
            count += self.regressor.coef_.shape[0]
        count += self.regressor.intercept_.shape[0]
        return count


class LinearLagConfig(DynamicIdentificationModelConfig):
    lag: int


class LinearLag(DynamicIdentificationModel):
    CONFIG = LinearLagConfig

    def __init__(self, config: LinearLagConfig) -> None:
        super().__init__(config)

        self.input_size = len(config.control_names)
        self.output_size = len(config.state_names)
        self.window_size = config.lag

        self.regressor = KernelRegression(
            input_dimension=self.get_actual_input_dimension(),
            output_dimension=self.output_size,
            input_window_size=self.window_size,
            output_window_size=self.window_size,
            input_kernels=[
                ZeroKernel(KernelHyperparameter())
                for _ in range(self.get_actual_input_dimension())
            ],
            output_kernels=[
                ZeroKernel(KernelHyperparameter()) for _ in range(self.output_size)
            ],
            bias=True,
            ignore_kernel=True,
        )

        self.control_mean: Optional[NDArray[np.float64]] = None
        self.control_std: Optional[NDArray[np.float64]] = None
        self.state_mean: Optional[NDArray[np.float64]] = None
        self.state_std: Optional[NDArray[np.float64]] = None

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]] = None,
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> Optional[Dict[str, NDArray[np.float64]]]:
        self.control_mean, self.control_std = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_std = utils.mean_stddev(state_seqs)

        control_seqs = [
            self.map_input(
                utils.normalize(control, self.control_mean, self.control_std)
            )
            for control in control_seqs
        ]
        state_seqs = [
            utils.normalize(state, self.state_mean, self.state_std)
            for state in state_seqs
        ]

        x, y = construct_fit_input_arguments(
            control_seqs=control_seqs,
            state_seqs=state_seqs,
            input_window_size=self.window_size,
            output_window_size=self.window_size,
        )
        self.regressor.fit(x, y)

        return dict()

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        x0: Optional[NDArray[np.float64]],
    ) -> Union[
        NDArray[np.float64], Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]
    ]:
        if (
            self.control_mean is None
            or self.control_std is None
            or self.state_mean is None
            or self.state_std is None
        ):
            raise ValueError('Model needs to be trained with .train before simulating.')

        initial_control = utils.normalize(
            initial_control, self.control_mean, self.control_std
        )
        initial_state = utils.normalize(initial_state, self.state_mean, self.state_std)
        control = utils.normalize(control, self.control_mean, self.control_std)

        state_window = initial_state
        for time, _ in enumerate(control):
            control_window = self.map_input(
                np.vstack((initial_control[1:, :], control[: time + 1, :]))
            )
            x = construct_predict_input_arguments_for_single_sample(
                control_window,
                state_window,
                input_window_size=self.window_size,
                output_window_size=self.window_size,
            )
            yhat = self.regressor.predict(x)
            state_window = np.vstack((state_window, yhat))

        predicted_states = utils.denormalize(
            state_window[initial_state.shape[0] :, :], self.state_mean, self.state_std
        )

        return predicted_states

    def save(
        self, file_path: Tuple[str, ...], tracker: BaseEventTracker = BaseEventTracker()
    ) -> None:
        if (
            self.control_mean is None
            or self.control_std is None
            or self.state_mean is None
            or self.state_std is None
        ):
            raise ValueError(
                'Model needs to be trained with .train before calling save.'
            )

        with h5py.File(file_path[0], mode='w') as f:
            f.create_dataset('weights_', data=self.regressor.weights_)
            f.create_dataset('control_mean', data=self.control_mean)
            f.create_dataset('control_std', data=self.control_std)
            f.create_dataset('state_mean', data=self.state_mean)
            f.create_dataset('state_std', data=self.state_std)

    def load(self, file_path: Tuple[str, ...]) -> None:
        with h5py.File(file_path[0], mode='r') as f:
            self.regressor.weights_ = f['weights_'][:]
            self.control_mean = f['control_mean'][:]
            self.control_std = f['control_std'][:]
            self.state_mean = f['state_mean'][:]
            self.state_std = f['state_std'][:]

    def get_file_extension(self) -> Tuple[str, ...]:
        return ('hdf5',)

    def get_parameter_count(self) -> int:
        return self.regressor.kernel_matrix_.shape[0] * self.output_size

    def get_actual_input_dimension(self) -> int:
        return self.input_size

    def map_input(self, control: NDArray[np.float64]) -> NDArray[np.float64]:
        return control


class QuadraticControlLag(LinearLag):
    def get_actual_input_dimension(self) -> int:
        return 2 * self.input_size

    def map_input(self, control: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.hstack((control, control * control))
