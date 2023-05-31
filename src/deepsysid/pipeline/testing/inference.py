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
    def __init__(self, config: BaseTestConfig, device_name: str) -> None:
        super().__init__(config, device_name)

        self.window_size = config.window_size
        self.horizon_size = config.horizon_size

    def test(
        self, model: DynamicIdentificationModel, simulations: List[TestSimulation]
    ) -> TestResult:
        control = []
        pred_states = []
        true_states = []
        initial_states = []
        file_names = []
        metadata = []
        for sample in split_simulations(
            self.window_size, self.horizon_size, simulations
        ):

            simulation_result = model.simulate(
                sample.initial_control,
                sample.initial_state,
                sample.true_control,
                sample.x0,
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
            initial_states.append(sample.x0)

        sequences: List[TestSequenceResult] = []
        if len(metadata) == 0:
            for c, ps, ts, xs in zip(control, pred_states, true_states, initial_states):
                sequences.append(
                    TestSequenceResult(
                        inputs=dict(control=c),
                        outputs=dict(true_state=ts, pred_state=ps),
                        initial_states=dict(initial_state=xs),
                        metadata=dict(),
                    )
                )
        else:
            for c, ps, ts, xs, md in zip(
                control, pred_states, true_states, initial_states, metadata
            ):
                sequences.append(
                    TestSequenceResult(
                        inputs=dict(control=c),
                        outputs=dict(true_state=ts, pred_state=ps),
                        initial_states=dict(initial_state=xs),
                        metadata=md,
                    )
                )
        return TestResult(sequences=sequences, metadata=dict(file_names=file_names))
