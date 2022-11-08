import abc
from typing import Literal, Type, Union

import numpy as np
from numpy.typing import NDArray

from ....models.base import NormalizedHiddenStateInitializerPredictorModel
from ..base import BaseTest, BaseTestConfig, TestSequenceResult


class StabilityTestConfig(BaseTestConfig):
    optimization_steps: int
    optimization_lr: float
    initial_mean_delta: float
    initial_std_delta: float
    clip_gradient_norm: float
    regularization_scale: float
    evaluation_sequence: Union[Literal['all'], int]


class BaseStabilityTest(BaseTest):
    CONFIG: Type[StabilityTestConfig] = StabilityTestConfig

    @abc.abstractmethod
    def evaluate_stability_of_sequence(
        self,
        model: NormalizedHiddenStateInitializerPredictorModel,
        device_name: str,
        control_dim: int,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        true_control: NDArray[np.float64],
    ) -> TestSequenceResult:
        pass
