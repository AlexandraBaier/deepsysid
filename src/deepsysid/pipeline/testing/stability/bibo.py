import logging
import time
from typing import List, Optional, Union

import numpy as np
import torch
from numpy.typing import NDArray

from ....models import utils
from ....models.base import (
    DynamicIdentificationModel,
    NormalizedHiddenStateInitializerPredictorModel,
    NormalizedHiddenStatePredictorModel,
)
from ..base import TestResult, TestSequenceResult, TestSimulation
from ..io import split_simulations
from .base import BaseStabilityTest, StabilityTestConfig

from ....tracker.event_data import TrackMetrics
from ....tracker.base import BaseEventTracker

logger = logging.getLogger(__name__)


class BiboStabilityTest(BaseStabilityTest):
    CONFIG = StabilityTestConfig

    def __init__(self, config: StabilityTestConfig, device_name: str) -> None:
        super().__init__(config, device_name)

        self.control_dim = len(config.control_names)

        self.device_name = device_name
        self.window_size = config.window_size
        self.horizon_size = config.horizon_size
        self.evaluation_sequence = config.evaluation_sequence

        self.initial_mean_delta = config.initial_mean_delta
        self.initial_std_delta = config.initial_std_delta
        self.optimization_lr = config.optimization_lr
        self.optimization_steps = config.optimization_steps
        self.regularization_scale = config.regularization_scale
        self.clip_gradient_norm = config.clip_gradient_norm

    def test(
        self,
        model: DynamicIdentificationModel,
        simulations: List[TestSimulation],
        tracker: BaseEventTracker = BaseEventTracker(),
    ) -> TestResult:

        # if not isinstance(model, NormalizedHiddenStateInitializerPredictorModel):
        #     return TestResult(list(), dict())

        test_sequence_results: List[TestSequenceResult] = []

        time_start_test = time.time()

        if isinstance(self.evaluation_sequence, int):
            logger.info(
                f'Test bibo stability for sequence number {self.evaluation_sequence}'
            )
            dataset = list(
                split_simulations(self.window_size, self.horizon_size, simulations)
            )
            sim = dataset[self.evaluation_sequence]
            test_sequence_results.append(
                self.evaluate_stability_of_sequence(
                    model=model,
                    device_name=self.device_name,
                    control_dim=self.control_dim,
                    true_control=sim.true_control,
                    tracker=tracker,
                )
            )
            tracker(
                TrackMetrics(
                    f'incremental stability gain for sequence {self.evaluation_sequence}',
                    {
                        'incremental gamma': float(
                            test_sequence_results[-1].metadata.stability_gain
                        )
                    },
                    self.evaluation_sequence,
                )
            )

        elif self.evaluation_sequence == 'all':
            logger.info(f'Test bibo stability for {self.evaluation_sequence} sequences')
            for idx_data, sim in enumerate(
                split_simulations(self.window_size, self.horizon_size, simulations)
            ):
                logger.info(f'Sequence number: {idx_data}')

                test_sequence_results.append(
                    self.evaluate_stability_of_sequence(
                        model=model,
                        device_name=self.device_name,
                        control_dim=self.control_dim,
                        true_control=sim.true_control,
                        tracker=tracker,
                    )
                )
                tracker(
                    TrackMetrics(
                        f'incremental stability gain for sequence {idx_data}',
                        {
                            'incremental gamma': float(
                                test_sequence_results[-1].metadata.stability_gain
                            )
                        },
                        idx_data,
                    )
                )

        time_end_test = time.time()

        test_time = time_end_test - time_start_test

        return TestResult(
            sequences=test_sequence_results, metadata=dict(test_time=[test_time])
        )

    def evaluate_stability_of_sequence(
        self,
        model: Union[
            NormalizedHiddenStateInitializerPredictorModel,
            NormalizedHiddenStatePredictorModel,
        ],
        device_name: str,
        control_dim: int,
        true_control: NDArray[np.float64],
    ) -> TestSequenceResult:
        if (
            model.state_mean is None
            or model.state_std is None
            or model.control_mean is None
            or model.control_std is None
        ):
            raise ValueError(
                'Mean and standard deviation is not initialized in the model'
            )

        model.predictor.train()

        wp = torch.from_numpy(true_control).double().to(device_name)

        # disturb input
        delta = torch.normal(
            self.initial_mean_delta,
            self.initial_std_delta,
            size=(self.horizon_size, control_dim),
            requires_grad=True,
            device=device_name,
        )

        # optimizer
        opt = torch.optim.Adam(  # type: ignore
            [delta], lr=self.optimization_lr, maximize=True
        )

        gamma_2: Optional[np.float64] = None
        for step_idx in range(self.optimization_steps):
            wp_a = wp + delta

            # model prediction
            zp_hat_a, _ = model.predictor(wp_a.unsqueeze(0))
            zp_hat_a = zp_hat_a.squeeze()

            # use log to avoid zero in the denominator (goes to -inf)
            # since we maximize this results in a punishment
            regularization = self.regularization_scale * torch.log(
                utils.sequence_norm(wp_a)
            )
            gamma_2_torch = utils.sequence_norm(zp_hat_a) / utils.sequence_norm(wp_a)
            L = gamma_2_torch + regularization
            L.backward()
            torch.nn.utils.clip_grad_norm_(delta, self.clip_gradient_norm)
            opt.step()

            gamma_2 = gamma_2_torch.cpu().detach().numpy()

            if step_idx % 100 == 0:
                logger.info(
                    f'step: {step_idx} \t '
                    f'gamma_2: {gamma_2:.3f} \t '
                    f'gradient norm: {torch.norm(delta.grad):.3f} '
                    f'\t -log(norm(denominator)): {regularization:.3f}'
                )

        return TestSequenceResult(
            inputs=dict(a=wp_a),
            outputs=dict(a=zp_hat_a),
            initial_states=dict(),
            metadata=dict(stability_gain=np.array([gamma_2])),
        )
