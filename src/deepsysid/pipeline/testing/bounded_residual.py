from typing import List, Optional, Union, Tuple, Dict, Any

import numpy as np
from numpy.typing import NDArray

from deepsysid.models.base import DynamicIdentificationModel
from deepsysid.models.hybrid.bounded_residual import HybridResidualLSTMModel
from deepsysid.pipeline.configuration import ExperimentConfiguration
from deepsysid.pipeline.testing import TestSimulation, ModelTestResult, split_simulations, BaseTest, BaseTestConfig, \
    TestResult


class BoundedResidualInferenceTestConfig(BaseTestConfig):
    thresholds: List[float]


class BoundedResidualInferenceTest(BaseTest):
    CONFIG = BoundedResidualInferenceTestConfig

    def __init__(self, config: BoundedResidualInferenceTestConfig) -> None:
        super().__init__(config)

        self.window_size = config.window_size
        self.horizon_size = config.horizon_size
        self.thresholds = config.thresholds

    def test(self, model: DynamicIdentificationModel, simulations: List[TestSimulation]) -> TestResult:
        if not isinstance(model, HybridResidualLSTMModel):
            return dict()

        results = dict()
        for threshold in self.thresholds:
            control = []
            pred_states = []
            true_states = []
            file_names = []
            metadata = []
            for sample in split_simulations(self.window_size, self.horizon_size, simulations):
                simulation_result = model.simulate(
                    sample.initial_control,
                    sample.initial_state,
                    sample.true_control,
                    threshold=threshold
                )
                if isinstance(simulation_result, np.ndarray):
                    pred_target = simulation_result
                else:
                    pred_target = simulation_result[0]
                    metadata.append(simulation_result[1])

                control.append(sample.true_control)
                pred_states.append(pred_target)
                true_states.append(sample.true_state)
                file_names.append(sample.file_name)

            result = dict(
                control=dict((str(k), v) for k, v in enumerate(control)),
                true_state=dict((str(k), v) for k, v in enumerate(true_states)),
                pred_state=dict((str(k), v) for k, v in enumerate(pred_states)),
                file_names=file_names,
                metadata=dict((str(k), v) for k, v in enumerate(metadata))
            )
            results[str(threshold)] = result

        return results
