from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from ..base import DynamicIdentificationModelConfig
from .kernel import KernelHyperparameter, ZeroKernel
from .regression import (
    BaseKernelRegressionModel,
    KernelRegression,
    construct_fit_input_arguments,
)


class LinearModel(BaseKernelRegressionModel):
    def __init__(self, config: DynamicIdentificationModelConfig):
        super().__init__()

        input_size = len(config.control_names)
        output_size = len(config.state_names)
        self._regressor = KernelRegression(
            input_dimension=input_size,
            output_dimension=output_size,
            input_window_size=1,
            output_window_size=1,
            input_kernels=[
                ZeroKernel(KernelHyperparameter()) for _ in range(input_size)
            ],
            output_kernels=[
                ZeroKernel(KernelHyperparameter()) for _ in range(output_size)
            ],
            ignore_kernel=True,
        )

    @property
    def regressor(self) -> KernelRegression:
        return self._regressor

    def map_input(self, control: NDArray[np.float64]) -> NDArray[np.float64]:
        return control

    def train_kernel_regressor(
        self,
        normalized_control_seqs: List[NDArray[np.float64]],
        normalized_state_seqs: List[NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:
        normalized_control_seqs = [
            self.map_input(control) for control in normalized_control_seqs
        ]
        x, y = construct_fit_input_arguments(
            control_seqs=normalized_control_seqs,
            state_seqs=normalized_state_seqs,
            input_window_size=self.regressor.input_window_size,
            output_window_size=self.regressor.output_window_size,
        )
        self.regressor.fit(x, y)
        return dict()


class LinearLagConfig(DynamicIdentificationModelConfig):
    lag: int


class LinearLag(BaseKernelRegressionModel):
    CONFIG = LinearLagConfig

    def __init__(self, config: LinearLagConfig) -> None:
        super().__init__()

        self.input_size = len(config.control_names)
        self.output_size = len(config.state_names)

        self._regressor = KernelRegression(
            input_dimension=self.get_actual_input_dimension(),
            output_dimension=self.output_size,
            input_window_size=config.lag,
            output_window_size=config.lag,
            input_kernels=[
                ZeroKernel(KernelHyperparameter())
                for _ in range(self.get_actual_input_dimension())
            ],
            output_kernels=[
                ZeroKernel(KernelHyperparameter()) for _ in range(self.output_size)
            ],
            ignore_kernel=True,
        )

    @property
    def regressor(self) -> KernelRegression:
        return self._regressor

    def train_kernel_regressor(
        self,
        normalized_control_seqs: List[NDArray[np.float64]],
        normalized_state_seqs: List[NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:
        normalized_control_seqs = [
            self.map_input(control) for control in normalized_control_seqs
        ]
        x, y = construct_fit_input_arguments(
            control_seqs=normalized_control_seqs,
            state_seqs=normalized_state_seqs,
            input_window_size=self.regressor.input_window_size,
            output_window_size=self.regressor.output_window_size,
        )
        self.regressor.fit(x, y)
        return dict()

    def get_actual_input_dimension(self) -> int:
        return self.input_size

    def map_input(self, control: NDArray[np.float64]) -> NDArray[np.float64]:
        return control


class QuadraticControlLag(LinearLag):
    def get_actual_input_dimension(self) -> int:
        return 2 * self.input_size

    def map_input(self, control: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.hstack((control, control * control))
