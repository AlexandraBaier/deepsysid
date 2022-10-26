import logging
from typing import List

import numpy as np

from ...models.base import DynamicIdentificationModel
from .base import (
    BaseTest,
    BaseTestConfig,
    TestResult,
    TestSequenceResult,
    TestSimulation,
)
from .io import split_simulations

logger = logging.getLogger(__name__)


class InferenceTest(BaseTest):
    def __init__(self, config: BaseTestConfig) -> None:
        super().__init__(config)

        self.window_size = config.window_size
        self.horizon_size = config.horizon_size

    def test(
        self, model: DynamicIdentificationModel, simulations: List[TestSimulation]
    ) -> TestResult:
        control = []
        pred_states = []
        true_states = []
        file_names = []
        metadata = []
        for sample in split_simulations(
            self.window_size, self.horizon_size, simulations
        ):
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

        sequences: List[TestSequenceResult] = []
        if len(metadata) == 0:
            for c, ps, ts in zip(control, pred_states, true_states):
                sequences.append(
                    TestSequenceResult(
                        inputs=dict(control=c),
                        outputs=dict(true_state=ts, pred_state=ps),
                        metadata=dict(),
                    )
                )
        else:
            for c, ps, ts, md in zip(control, pred_states, true_states, metadata):
                sequences.append(
                    TestSequenceResult(
                        inputs=dict(control=c),
                        outputs=dict(true_state=ts, pred_state=ps),
                        metadata=md,
                    )
                )
        return TestResult(sequences=sequences, metadata=dict(file_names=file_names))
