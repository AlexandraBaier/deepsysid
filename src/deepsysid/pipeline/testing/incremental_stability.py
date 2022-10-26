import abc
import logging
from typing import List, Literal, Optional, Protocol, Tuple, Union, runtime_checkable

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn
from deepsysid.models.switching.switchrnn import StableSwitchingLSTMModel
from deepsysid.models.recurrent import (
    RnnInit,
    LtiRnnInit,
    ConstrainedRnn,
    LSTMCombinedInitModel,
    LSTMInitModel,
)

from deepsysid.networks.switching import StableSwitchingLSTM

from ...models import utils
from ...models.base import DynamicIdentificationModel
from .base import (
    BaseTest,
    BaseTestConfig,
    TestResult,
    TestSequenceResult,
    TestSimulation,
    StabilityTestConfig,
)
from .io import split_simulations

logger = logging.getLogger(__name__)


@runtime_checkable
class SupportsRNNForward(Protocol):
    def forward(
        self,
        x_pred: torch.Tensor,
        hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        y_init: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        pass


class SupportsRNNForwardModule(SupportsRNNForward, nn.Module):
    pass


@runtime_checkable
class SupportsStabilityTest(Protocol):
    device_name: str
    control_dim: int
    state_mean: NDArray[np.float64]
    state_std: NDArray[np.float64]
    control_mean: NDArray[np.float64]
    control_std: NDArray[np.float64]
    initializer: SupportsRNNForward
    predictor: SupportsRNNForward


class SupportsStabilityTestDynamicIdentificationModel(
    SupportsStabilityTest, DynamicIdentificationModel, metaclass=abc.ABCMeta
):
    pass


class IncrementalStabilityTest(BaseTest):
    CONFIG = StabilityTestConfig

    def __init__(self, config: StabilityTestConfig) -> None:
        super().__init__(config)

        self.config = config
        self.window_size = config.window_size
        self.horizon_size = config.horizon_size
        self.evaluation_sequence = config.evaluation_sequence

    def test(
        self, model: DynamicIdentificationModel, simulations: List[TestSimulation]
    ) -> TestResult:
        # if not isinstance(model, SupportsStabilityTestDynamicIdentificationModel):
        #     return TestResult(list(), dict())
        if not (
            any(
                (
                    isinstance(model, RnnInit),
                    isinstance(model, LtiRnnInit),
                    isinstance(model, ConstrainedRnn),
                    isinstance(model, LSTMInitModel),
                )
            )
        ):
            return TestResult(list(), dict())

        controls = []
        pred_states = []
        true_states = []
        stability_gains = []

        if isinstance(self.evaluation_sequence, int):
            logger.info(
                f'Test incremental stability for sequence number {self.evaluation_sequence}'
            )
            dataset = list(
                split_simulations(self.window_size, self.horizon_size, simulations)
            )
            sim = dataset[self.evaluation_sequence]
            gamma_2, control, pred_state = optimize_input_disturbance(
                config=self.config,
                model=model,
                initial_control=sim.initial_control,
                initial_state=sim.initial_state,
                true_control=sim.true_control,
            )

            stability_gains.append(np.array([gamma_2]))
            controls.append(control)
            pred_states.append(pred_state)
            true_states.append(sim.true_state)

        elif self.evaluation_sequence == 'all':
            logger.info(f'Test incremental stability for {self.evaluation_sequence} sequences')
            for idx_data, sim in enumerate(
                split_simulations(self.window_size, self.horizon_size, simulations)
            ):
                logger.info(f'Sequence number: {idx_data}')

                gamma_2, control, pred_state = optimize_input_disturbance(
                    config=self.config,
                    model=model,
                    initial_control=sim.initial_control,
                    initial_state=sim.initial_state,
                    true_control=sim.true_control,
                )

                stability_gains.append(np.array([gamma_2]))
                controls.append(control)
                pred_states.append(pred_state)
                true_states.append(sim.true_state)

        sequences = []
        for c, ps, ts, sg in zip(controls, pred_states, true_states, stability_gains):
            sequences.append(
                TestSequenceResult(
                    inputs=dict(control=c),
                    outputs=dict(pred_state=ps, true_state=ts),
                    metadata=dict(stability_gains=sg),
                )
            )
        return TestResult(sequences=sequences, metadata=dict())


def optimize_input_disturbance(
    config: StabilityTestConfig,
    model: Union[RnnInit, LtiRnnInit, LSTMInitModel, ConstrainedRnn],
    initial_control: NDArray[np.float64],
    initial_state: NDArray[np.float64],
    true_control: NDArray[np.float64],
) -> Tuple[np.float64, NDArray[np.float64], NDArray[np.float64]]:
    if (
        model.state_mean is None
        or model.state_std is None
        or model.control_mean is None
        or model.control_std is None
    ):
        raise ValueError('Mean and standard deviation is not initialized in the model')

    model.predictor.train()

    # normalize data
    u_init_norm_numpy = utils.normalize(
        initial_control, model.control_mean, model.control_std
    )
    u_norm_numpy = utils.normalize(true_control, model.control_mean, model.control_std)
    y_init_norm_numpy = utils.normalize(
        initial_state, model.state_mean, model.state_std
    )

    # convert to tensors
    u_init_norm = (
        torch.from_numpy(np.hstack((u_init_norm_numpy[1:], y_init_norm_numpy[:-1])))
        .unsqueeze(0)
        .float()
        .to(model.device_name)
    )
    u_norm = torch.from_numpy(u_norm_numpy).unsqueeze(0).float().to(model.device_name)

    # disturb input
    delta = torch.normal(
        config.initial_mean_delta,
        config.initial_std_delta,
        size=(config.horizon_size, model.control_dim),
        requires_grad=True,
        device=model.device_name,
    )

    # optimizer
    opt = torch.optim.Adam(  # type: ignore
        [delta], lr=config.optimization_lr, maximize=True
    )

    gamma_2: Optional[np.float64] = None
    control: Optional[NDArray[np.float64]] = None
    pred_state: Optional[NDArray[np.float64]] = None
    for step_idx in range(config.optimization_steps):
        u_a = u_norm + delta
        u_b = u_norm.clone()
        # model prediction
        _, hx = model.initializer(u_init_norm)
        # TODO set initial state to zero should be good to find unstable sequences
        hx = (
            torch.zeros_like(hx[0]).to(model.device_name),
            torch.zeros_like(hx[1]).to(model.device_name),
        )
        y_hat_a, _ = model.predictor(u_a, hx=hx)
        y_hat_b, _ = model.predictor(u_b, hx=hx)
        y_hat_a = y_hat_a.squeeze()
        y_hat_b = y_hat_b.squeeze()

        # use log to avoid zero in the denominator (goes to -inf)
        # since we maximize this results in a punishment
        regularization = config.regularization_scale * torch.log(
            utils.sequence_norm(u_a - u_b)
        )
        gamma_2_torch = utils.sequence_norm(y_hat_a - y_hat_b) / utils.sequence_norm(
            u_a - u_b
        )
        L = gamma_2_torch + regularization
        L.backward()
        torch.nn.utils.clip_grad_norm_(delta, config.clip_gradient_norm)
        opt.step()

        control = utils.denormalize(
            (u_a - u_b).cpu().detach().numpy().squeeze(),
            model.control_mean,
            model.control_std,
        )
        pred_state = utils.denormalize(
            (y_hat_a - y_hat_b).cpu().detach().numpy(),
            model.state_mean,
            model.state_std,
        )

        gamma_2 = gamma_2_torch.cpu().detach().numpy()
        if step_idx % 100 == 0:
            logger.info(
                f'step: {step_idx} \t '
                f'gamma_2: {gamma_2:.3f} \t '
                f'gradient norm: {torch.norm(delta.grad):.3f} '
                f'\t -log(norm(denominator)): {regularization:.3f}'
            )

    if gamma_2 is None or control is None or pred_state is None:
        raise ValueError(
            'Computation of optimal input disturbance did not create any result.'
            'This might be caused by an unknown stability type.'
            f'Your stability type is {config.type}.'
        )

    return gamma_2, control, pred_state
