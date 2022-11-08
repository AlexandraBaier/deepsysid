import abc
from typing import List, Type, Union, Literal

import numpy as np
from numpy.typing import NDArray

from ....models.base import DynamicIdentificationModel
from ..base import (
    BaseTest,
    BaseTestConfig,
    TestSimulation,
    TestResult,
    TestSequenceResult,
)


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

    def __init__(self, config: StabilityTestConfig):
        pass

    @abc.abstractmethod
    def test(
        self, model: DynamicIdentificationModel, simulations: List[TestSimulation]
    ) -> TestResult:
        pass

    @abc.abstractmethod
    def evaluate_stability_of_sequence(
        self,
        model: DynamicIdentificationModel,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        true_control: NDArray[np.float64],
    ) -> TestSequenceResult:
        pass
