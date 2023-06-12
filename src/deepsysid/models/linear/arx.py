from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

from .kernel import Kernel


class KernelRegression:
    def __init__(
        self,
        input_dimension: int,
        output_dimension: int,
        input_window_size: int,
        output_window_size: int,
        input_kernels: List[Kernel],
        output_kernels: List[Kernel],
        bias: bool = False,
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
        self.bias = bias

        self.kernel_matrix_ = self._construct_kernel_matrix()
        self.weights_: Optional[NDArray[np.float64]] = None

    def fit(self, x: NDArray[np.float64], y: NDArray[np.float64]) -> 'KernelRegression':
        """
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

        # Extend input with bias term.
        if self.bias:
            x = np.hstack((x, np.ones((x.shape[0]))))

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
            a = x.T @ x + np.linalg.inv(self.kernel_matrix_)
            b = x.T @ y[:, j]
            weight, _, _, _ = np.linalg.lstsq(a, b)
            self.weights_[j, :] = weight

        return self

    def predict(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.weights_ is None:
            raise ValueError(
                'Model has not been trained yet, call fit before calling predict.'
            )

        self._check_x(x)

        if self.bias:
            x = np.hstack((x, np.ones((x.shape[0]))))

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
            partial_kernel_matrix = kernel.construct(self.output_dimension)
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
        if self.bias:
            n_parameters_per_output += 1

        return n_parameters_per_output

    def _check_fit_arguments(
        self, x: NDArray[np.float64], y: NDArray[np.float64]
    ) -> None:
        self._check_x(x)

        if len(y.shape) != 2:
            raise ValueError()

        if y.shape[1] != self.output_dimension:
            raise ValueError()

        if x.shape[0] != y.shape[0]:
            raise ValueError()

    def _check_x(self, x: NDArray[np.float64]) -> None:
        if len(x.shape) != 2:
            raise ValueError()
        n_params_per_channel = self._count_parameter_per_output_channel()
        if (not self.bias and x.shape[1] != n_params_per_channel) or (
            self.bias and x.shape[1] != (n_params_per_channel - 1)
        ):
            raise ValueError()
