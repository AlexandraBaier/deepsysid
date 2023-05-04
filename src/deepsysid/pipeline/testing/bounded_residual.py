from typing import List

from ...models.base import DynamicIdentificationModel
from ...models.hybrid.bounded_residual import HybridResidualLSTMModel
from .base import (
    BaseTest,
    BaseTestConfig,
    TestResult,
    TestSequenceResult,
    TestSimulation,
)
from .io import split_simulations


class BoundedResidualInferenceTestConfig(BaseTestConfig):
    thresholds: List[float]


class BoundedResidualInferenceTest(BaseTest):
    CONFIG = BoundedResidualInferenceTestConfig

    def __init__(
        self, config: BoundedResidualInferenceTestConfig, device_name: str
    ) -> None:
        super().__init__(config, device_name=device_name)

        self.window_size = config.window_size
        self.horizon_size = config.horizon_size
        self.thresholds = config.thresholds

    def test(
        self, model: DynamicIdentificationModel, simulations: List[TestSimulation]
    ) -> TestResult:
        if not isinstance(model, HybridResidualLSTMModel):
            return TestResult(list(), dict())

        metadata_ls = []
        control_ls = []
        pred_state_ls = []
        true_state_ls = []
        initial_states = []
        file_names = []
        thresholds = []
        for threshold in self.thresholds:
            for idx, sample in enumerate(
                split_simulations(self.window_size, self.horizon_size, simulations)
            ):
                pred_target, metadata_pred = model.simulate(
                    sample.initial_control,
                    sample.initial_state,
                    sample.true_control,
                    sample.x0,
                    threshold=threshold,
                )
                metadata_ls.append(metadata_pred)
                control_ls.append(sample.true_control)
                pred_state_ls.append(pred_target)
                true_state_ls.append(sample.true_state)
                file_names.append(sample.file_name)
                thresholds.append(threshold)
                initial_states.append(sample.x0)

        sequences = []
        for md, c, ps, ts, xs in zip(
            metadata_ls, control_ls, pred_state_ls, true_state_ls, initial_states
        ):
            sequences.append(
                TestSequenceResult(
                    inputs=dict(control=c),
                    outputs=dict(true_state=ts, pred_state=ps),
                    initial_states=dict(initial_state=xs),
                    metadata=md,
                )
            )
        return TestResult(
            sequences=sequences,
            metadata=dict(file_names=file_names, thresholds=thresholds),
        )
