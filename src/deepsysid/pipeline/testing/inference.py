import logging
from typing import List

import numpy as np

from deepsysid.models.base import DynamicIdentificationModel
from deepsysid.pipeline.testing import BaseTest, BaseTestConfig, TestSimulation, split_simulations, TestResult

logger = logging.getLogger(__name__)


class InferenceTest(BaseTest):
    def __init__(self, config: BaseTestConfig) -> None:
        super().__init__(config)

        self.window_size = config.window_size
        self.horizon_size = config.horizon_size

    def test(
            self,
            model: DynamicIdentificationModel,
            simulations: List[TestSimulation]
    ) -> TestResult:
        control = []
        pred_states = []
        true_states = []
        file_names = []
        metadata = []
        for sample in split_simulations(self.window_size, self.horizon_size, simulations):
            simulation_result = model.simulate(
                sample.initial_control, sample.initial_state, sample.true_control
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

        return dict(
            control=dict((str(k), v) for k, v in enumerate(control)),
            true_state=dict((str(k), v) for k, v in enumerate(true_states)),
            pred_state=dict((str(k), v) for k, v in enumerate(pred_states)),
            file_names=file_names,
            metadata=dict((str(k), v) for k, v in enumerate(metadata))
        )
