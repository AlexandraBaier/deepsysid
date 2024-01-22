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
    NormalizedControlStateModel
)
from ....models.switching.switchrnn import SwitchingLSTMBaseModel
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
                    true_output = sim.true_state
                )
            )
            tracker(
                TrackMetrics(
                    f'stability gain for sequence {self.evaluation_sequence}',
                    {
                        'max gamma': float(
                            test_sequence_results[-1].metadata['stability_gain']
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
                    )
                )
                tracker(
                    TrackMetrics(
                        f'stability gain for sequence {idx_data}',
                        {
                            'max gamma': float(
                                test_sequence_results[-1].metadata['stability_gain']
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
            DynamicIdentificationModel
        ],
        device_name: str,
        control_dim: int,
        true_control: NDArray[np.float64],
        true_output: NDArray[np.float64],
    ) -> TestSequenceResult:
        # if (
        #     model.state_mean is None
        #     or model.state_std is None
        #     or model.control_mean is None
        #     or model.control_std is None
        # ):
        #     raise ValueError(
        #         'Mean and standard deviation is not initialized in the model'
        #     )

        model.predictor.train()
        N, nu = true_control.shape
        t = torch.linspace(0,N-1,N)
        # u_norm = torch.from_numpy(true_control).double().to(device_name)
        # u_norm = torch.zeros_like(torch.tensor(true_control)).double().to(device_name)
        # d = torch.tensor(np.sin(t/0.02*1/1000).reshape((N,nu)))
        # d = torch.from_numpy(true_control).double().to(device_name)
        d = torch.zeros_like(torch.tensor(true_control)).double().to(device_name)

        # disturb input
        delta = torch.normal(
            self.initial_mean_delta,
            self.initial_std_delta,
            size=(self.horizon_size, control_dim),
            requires_grad=True,
            device=device_name,
        )
        c1 = torch.tensor(0.0, requires_grad=True)
        c2 = torch.tensor(0.0, requires_grad=True)
        c3 = torch.tensor(1.0, requires_grad=True)

        # A = torch.tensor(1.0, requires_grad=True) # sine amplitude
        f = torch.tensor(1.0/100, requires_grad=True) # ordinary frequency
        phi = torch.tensor(0.0, requires_grad=True) # phase
        b = torch.tensor(0.0, requires_grad=True) # bias

        # optimizer
        opt = torch.optim.Adam(  # type: ignore
            [delta], lr=self.optimization_lr, maximize=True
        )
        # logger.info(
        #     f'f {f:.4f}, phi {phi:.4f}, b {b:.4f}, c1 {c1:.4f} c2 {c2:.4f}, c3 {c3:.4f} \t'
        # )

        
        gamma_2: Optional[np.float64] = None
        gamma_2s: List[np.float64] = []
        e_hat_as: List[NDArray[np.float64]] = []
        d_as: List[NDArray[np.float64]] = []

        for step_idx in range(self.optimization_steps):
            opt.zero_grad()
            # delta = ((t*c2+c3)*torch.sin(2*torch.pi*f*t+phi)+b).reshape(N,nu)
            d_a_norm = d + delta

            # normalize input
            # if isinstance(model, NormalizedControlStateModel):
            #     d_a_norm = utils.normalize(d_a, model.control_mean, model.control_std)
            # else:
            #     d_a_norm = d_a

            # model prediction
            if isinstance(model, SwitchingLSTMBaseModel):
                # d_a_norm = utils.normalize(d_a, model.control_mean, model.control_std)
                ne = model.output_dim
                e_hat_a_norm = model.predictor(d_a_norm.float().unsqueeze(0), previous_output=torch.zeros(1,ne)).outputs.squeeze()
            else:
                e_hat_a_norm = model.predictor(d_a_norm.unsqueeze(0))[0].squeeze()
            
            # e_hat_a = e_hat_a.squeeze()

            # use log to avoid zero in the denominator (goes to -inf)
            # since we maximize this results in a punishment
            regularization = self.regularization_scale * torch.log(
                utils.sequence_norm(d_a_norm)
            )
            gamma_2_torch = torch.sqrt(utils.sequence_norm(e_hat_a_norm) / utils.sequence_norm(d_a_norm))
            L = gamma_2_torch + regularization
            # L = gamma_2_torch
            L.backward(retain_graph=True)
            # torch.nn.utils.clip_grad_norm_(delta, self.clip_gradient_norm)
            # torch.nn.utils.clip_grad.clip_grad_value_(f,1)
            # torch.nn.utils.clip_grad.clip_grad_value_(c2,1)
            opt.step()

            gamma_2 = gamma_2_torch.cpu().detach().numpy()

            if step_idx % 1 == 0:
                logger.info(
                    f'step: {step_idx} \t '
                    f'gamma_2: {gamma_2:.3f} \t '
                    f'gradient norm: {torch.norm(delta.grad):.3f} \t'
                    # f'f {f:.4f}, phi {phi:.4f}, b {b:.4f}, c2 {c2:.4f}, c3 {c3:.4f} \t'
                    # f'grads: f {f.grad:.4f}, phi {phi.grad:.4f}, b {b.grad:.4f}, c2 {c2.grad:.4f}, c3 {c3.grad:.4f} \t'
                    f'\t -log(norm(denominator)): {regularization:.3f}'
                )
            gamma_2s.append(gamma_2)
            if isinstance(model, NormalizedControlStateModel):
                e_hat_a = utils.denormalize(e_hat_a_norm.detach().numpy(), model.state_mean, model.state_mean)
                d_a = utils.denormalize(d_a_norm.detach().numpy(), model.control_mean, model.control_std)
            elif isinstance(model, SwitchingLSTMBaseModel):
                e_hat_a = utils.denormalize(e_hat_a_norm.detach().numpy(), model.output_mean, model.output_std)
                d_a = utils.denormalize(d_a_norm.detach().numpy(), model.control_mean, model.control_std)
            else:
                e_hat_a = e_hat_a_norm.detach().numpy()
                d_a = d_a_norm.detach().numpy()
            e_hat_as.append(e_hat_a)
            d_as.append(d_a)

        return TestSequenceResult(
            inputs=dict(a=d_as[np.argmax(gamma_2s)]),
            outputs=dict(a=e_hat_as[np.argmax(gamma_2s)]),
            initial_states=dict(),
            metadata=dict(stability_gain=np.array([max(gamma_2s)]))
        )
