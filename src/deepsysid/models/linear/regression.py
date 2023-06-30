import abc
from typing import Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.typing import NDArray
from sklearn.base import BaseEstimator

from ...tracker.base import BaseEventTracker
from ..base import DynamicIdentificationModel
from .kernel import Kernel
from .normalization import StandardNormalizer


class KernelRegression(BaseEstimator):
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        input_window_size: int,
        output_window_size: int,
        input_kernels: List[Kernel],
        output_kernels: List[Kernel],
        ignore_kernel: bool = False,
    ) -> None:
        if len(input_kernels) != input_dimension:
            raise ValueError(
                'Expected is a kernel per input variable, '
                f'but only received {len(input_kernels)} kernels '
                f'for {input_dimension} inputs.'
            )
        if len(output_kernels) != output_dimension:
            raise ValueError(
                'Expected is a kernel per output variable, '
                f'but only received {len(output_kernels)} kernels '
                f'for {output_dimension} outputs.'
            )

        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.input_window_size = input_window_size
        self.output_window_size = output_window_size
        self.input_kernels = input_kernels
        self.output_kernels = output_kernels
        self.ignore_kernel = ignore_kernel

        self.kernel_matrix = self._construct_kernel_matrix()
        self.weights_: Optional[NDArray[np.float64]] = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> 'KernelRegression':
        """
        # Input structure:
        x.shape = (N, nu * k + ny * o)
        y.shape = (N, o)

        for i = 1...N
        x_i = [
            u_1(t) u_1(t-1) ... u_1(t-nu+1)
            ...
            u_k(t) u_k(t-1) ... u_k(t-nu+1)
            y_1(t-1) y_1(t-2) ... y_1(t-ny)
            ...
            y_o(t-1) y_o(t-2) ... y_o(t-ny)
        ]
        y_i = [
            y_1(t) y_2(t) ... y_o(t)
        ]
        with
            N = x.shape[0] = y.shape[0],
            nu = input_window_size,
            ny = output_window_size,
            k = input_size,
            o = output_size

        # Optimization
        theta_j = argmin_theta || y_j - x theta ||^2 + theta^T Pi^-1 theta (1)
            = (R + Pi^-1)^-1  x^T y_j (2)
        with
            R = x^T x
            Pi = self.kernel_matrix_ (constructed from input_kernels and output_kernels)
            x in R^(N times (input_size*input_window + output_size*output_window)
            y_j in R^N for j=1...output_size
            N = number of training samples

        We solve this optimization problem with numpy.lstsq
        by writing the equation (2) as
            (R + Pi^-1) theta_j =  x^T y_j
        which matches the form Ax = b expected for least squares.
        """
        self._check_fit_arguments(x, y)

        # For each output compute the parameter vector using least-squares optimization.
        self.weights_ = np.zeros(
            (self.output_dimension, self._count_parameter_per_output_channel()),
            dtype=np.float64,
        )
        for j in range(self.output_dimension):
            # Do the dimensions work together?
            # N = of number training samples
            # d = input_dimension * input_window_size
            #   + output_dimension * output_window_size
            # x = (N, d), x.T = (d, N), y[:, j] = (N,)
            # x.T @ x = (d, d), self.kernel_matrix = (d, d)
            # => a = (d, d)
            # x.T @ y[:, j] = (d,)
            # => b = (d, )
            # ls formulation: a * x = b
            # => x = (d, ) => self.weights[j, :] requires shape (d,)
            # Everything is fine.
            if self.ignore_kernel:
                a = x.T @ x
            else:
                a = x.T @ x + np.linalg.inv(self.kernel_matrix)

            b = x.T @ y[:, j]
            weight, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
            self.weights_[j, :] = weight

        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Check fit for input structure of x.
        """
        if self.weights_ is None:
            raise ValueError(
                'Model has not been trained yet, call fit before calling predict.'
            )

        self._check_x(x)

        y = np.zeros((x.shape[0], self.output_dimension), dtype=np.float64)
        for j in range(self.output_dimension):
            yj = x @ self.weights_[j, :]
            y[:, j] = yj

        return y

    def _construct_kernel_matrix(self) -> NDArray[np.float64]:
        n_params = self._count_parameter_per_output_channel()
        full_kernel_matrix = np.zeros((n_params, n_params), dtype=np.float64)

        for input_idx, kernel in enumerate(self.input_kernels):
            partial_kernel_matrix = kernel.construct(self.input_window_size)
            partial_kernel_slice = slice(
                input_idx * self.input_window_size,
                (input_idx + 1) * self.input_window_size,
            )
            full_kernel_matrix[
                partial_kernel_slice, partial_kernel_slice
            ] = partial_kernel_matrix

        offset = self.input_dimension * self.input_window_size
        for output_idx, kernel in enumerate(self.output_kernels):
            partial_kernel_matrix = kernel.construct(self.output_window_size)
            partial_kernel_slice = slice(
                offset + output_idx * self.output_window_size,
                offset + (output_idx + 1) * self.output_window_size,
            )
            full_kernel_matrix[
                partial_kernel_slice, partial_kernel_slice
            ] = partial_kernel_matrix

        # We leave the bias regularization at 0, which is found
        # in the last column and row of full_kernel_matrix.

        return full_kernel_matrix

    def _count_parameter_per_output_channel(self) -> int:
        n_parameters_per_output = (
            self.input_dimension * self.input_window_size
            + self.output_dimension * self.output_window_size
        )
        return n_parameters_per_output

    def _check_fit_arguments(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> None:
        self._check_x(x)

        if len(y.shape) != 2:
            raise ValueError(
                f'y needs to be a two-dimensional array '
                f'but is {len(y.shape)}-dimensional instead.'
            )

        if y.shape[1] != self.output_dimension:
            raise ValueError(
                f'Expected second dimension of y to be {self.output_dimension} '
                f'but found {y.shape[1]} instead.'
            )

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f'x and y are mismatched on the first dimension '
                f'but they should match: '
                f'x={x.shape} and y={y.shape}.'
            )

    def _check_x(self, x: NDArray[np.float64]) -> None:
        if len(x.shape) != 2:
            raise ValueError(
                f'x needs to be a two-dimensional array '
                f'but is {len(x.shape)}-dimensional instead.'
            )

        expected_size = self._count_parameter_per_output_channel()
        if x.shape[1] != expected_size:
            raise ValueError(
                f'Second dimension of x has expected size {expected_size}, '
                f'but found {x.shape[1]} instead.'
            )


def construct_fit_input_arguments(
    control_seqs: List[NDArray[np.float64]],
    state_seqs: List[NDArray[np.float64]],
    input_window_size: int,
    output_window_size: int,
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    input_size = control_seqs[0].shape[1]
    output_size = state_seqs[0].shape[1]
    window_size = max(input_window_size, output_window_size)

    x_ls = []
    y_ls = []
    for control, state in zip(control_seqs, state_seqs):
        inputs = (
            # Create sliding window over time for the larger window size.
            # shape = (samples, input_size, window_size)
            sliding_window_view(control[1:, :], window_size, axis=0)
            # Reverse the time ordering, so that each window descends in time.
            [:, :, ::-1]
            # Cut off the window at the required length,
            # Either nothing changes or the window is truncated.
            # shape = (samples, input_size, input_window_size)
            [:, :, :input_window_size]
            # Flatten each time window. The first dimension is the number of samples.
            .reshape((-1, input_window_size * input_size))
        )
        # Same logic as above for outputs.
        outputs = sliding_window_view(state[:-1, :], window_size, axis=0)[:, :, ::-1][
            :, :, :output_window_size
        ].reshape((-1, output_window_size * output_size))
        # Concatenate them for complete input x_i.
        x_i = np.hstack((inputs, outputs))

        # y_i is simply the outputs shifted
        # by the window_size with the last element not truncated.
        # For x_i, the last output is truncated.
        y_i = state[window_size:]

        x_ls.append(x_i)
        y_ls.append(y_i)

    x = np.vstack(x_ls)
    y = np.vstack(y_ls)
    return x, y


def construct_predict_input_arguments_for_single_sample(
    control: NDArray[np.float64],
    state: NDArray[np.float64],
    input_window_size: int,
    output_window_size: int,
) -> NDArray[np.float64]:
    """
    control.shape = (t, k)
    state.shape = (t, o)
    input_window_size <= t
    output_window_size <= t
    control = [
        [u_1(1) ... u_k(1)],
        ...
        [u_1(t) ... u_k(t)]
    ]
    state = [
        [y_1(0), ... y_o(1)],
        ...
        [y_1(t-1), ..., y_o(t-1)]
    ]
    """
    if control.shape[0] != state.shape[0]:
        raise ValueError(
            'First dimension (for time) of control and state needs to match, '
            f'but control.shape={control.shape} and state.shape={state.shape}.'
        )
    if input_window_size > control.shape[0]:
        raise ValueError(
            'input_window_size needs to be smaller '
            'than first dimension of control and state, '
            f'but {input_window_size=} is greater than {control.shape[0]}.'
        )
    if output_window_size > control.shape[0]:
        raise ValueError(
            'input_window_size needs to be smaller '
            'than first dimension of control and state, '
            f'but {output_window_size=} is greater than {control.shape[0]}.'
        )

    input_size = control.shape[1]
    output_size = state.shape[1]

    # Example result of the following transformation with window = 3:
    # [['u_0(9)', 'u_1(9)'],
    #  ['u_0(8)', 'u_1(8)'],
    #  ['u_0(7)', 'u_1(7)'],
    #  ['u_0(6)', 'u_1(6)']]
    # -> [['u_0(9)', 'u_0(8)', 'u_0(7)', 'u_1(9)', 'u_1(8)', 'u_1(7)']]
    # transpose is required to get the above result,
    # omitting the transpose will give the following result
    # - >[['u_0(9)', 'u_1(9)', 'u_0(8)', 'u_1(8)', 'u_0(7)', 'u_1(7)']]
    # fit/predict expect the former variant achieved via transpose.
    inputs = control[::-1, :][:input_window_size, :].T.reshape(
        1, input_window_size * input_size
    )
    outputs = state[::-1, :][:output_window_size, :].T.reshape(
        1, output_window_size * output_size
    )

    x = np.hstack((inputs, outputs))
    return x


class BaseKernelRegressionModel(DynamicIdentificationModel, metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        self.normalizer: Optional[StandardNormalizer] = None

    @property
    @abc.abstractmethod
    def regressor(self) -> KernelRegression:
        pass

    @abc.abstractmethod
    def map_input(self, control: NDArray[np.float64]) -> NDArray[np.float64]:
        pass

    @abc.abstractmethod
    def train_kernel_regressor(
        self,
        normalized_control_seqs: List[NDArray[np.float64]],
        normalized_state_seqs: List[NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:
        pass

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]] = None,
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> Optional[Dict[str, NDArray[np.float64]]]:
        self.normalizer = StandardNormalizer.from_training_data(
            control_seqs=control_seqs, state_seqs=state_seqs
        )
        control_seqs = [
            self.normalizer.normalize_control(control) for control in control_seqs
        ]
        state_seqs = [self.normalizer.normalize_state(state) for state in state_seqs]

        metadata = self.train_kernel_regressor(
            normalized_control_seqs=control_seqs, normalized_state_seqs=state_seqs
        )
        metadata['kernel_matrix_'] = self.regressor.kernel_matrix
        return metadata

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        x0: Optional[NDArray[np.float64]],
        initial_x0: Optional[NDArray[np.float64]],
    ) -> Union[
        NDArray[np.float64], Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]
    ]:
        normalizer = self.normalizer
        if normalizer is None:
            raise ValueError('Model needs to be trained with .train before simulating.')

        initial_control = normalizer.normalize_control(initial_control)
        initial_state = normalizer.normalize_state(initial_state)
        control = normalizer.normalize_control(control)

        state_window = initial_state
        for time, _ in enumerate(control):
            control_window = self.map_input(
                np.vstack((initial_control[1:, :], control[: time + 1, :]))
            )
            x = construct_predict_input_arguments_for_single_sample(
                control_window,
                state_window,
                input_window_size=self.regressor.input_window_size,
                output_window_size=self.regressor.output_window_size,
            )
            yhat = self.regressor.predict(x)
            state_window = np.vstack((state_window, yhat))

        predicted_states = state_window[initial_state.shape[0] :, :]
        predicted_states = normalizer.denormalize_state(predicted_states)
        return predicted_states

    def save(
        self, file_path: Tuple[str, ...], tracker: BaseEventTracker = BaseEventTracker()
    ) -> None:
        if self.normalizer is None:
            raise ValueError(
                'Model needs to be trained with .train before calling save.'
            )

        with h5py.File(file_path[0], mode='w') as f:
            f.create_dataset('weights_', data=self.regressor.weights_)
            f.create_dataset('control_mean', data=self.normalizer.control_mean)
            f.create_dataset('control_std', data=self.normalizer.control_std)
            f.create_dataset('state_mean', data=self.normalizer.state_mean)
            f.create_dataset('state_std', data=self.normalizer.state_std)

    def load(self, file_path: Tuple[str, ...]) -> None:
        with h5py.File(file_path[0], mode='r') as f:
            self.regressor.weights_ = f['weights_'][:]
            self.normalizer = StandardNormalizer(
                control_mean=f['control_mean'][:],
                control_std=f['control_std'][:],
                state_mean=f['state_mean'][:],
                state_std=f['state_std'][:],
            )

    def get_file_extension(self) -> Tuple[str, ...]:
        return ('hdf5',)

    def get_parameter_count(self) -> int:
        return self.regressor.kernel_matrix.shape[0] * self.regressor.output_dimension
