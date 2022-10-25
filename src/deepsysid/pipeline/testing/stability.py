import logging
from typing import List, Literal, Tuple, Union, Protocol, runtime_checkable, Optional

import numpy as np
import torch
from numpy.typing import NDArray
from torch import nn

from deepsysid.models import utils
from deepsysid.models.base import DynamicIdentificationModel
from deepsysid.models.recurrent import LSTMInitModel, ConstrainedRnn
from deepsysid.pipeline.testing import TestSimulation, split_simulations, BaseTest, TestResult, BaseTestConfig

logger = logging.getLogger(__name__)


@runtime_checkable
class SupportsRNNForward(Protocol):
    def forward(
            self,
            x_pred: torch.Tensor,
            hx: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            return_state: bool = False
    ):
        pass

@runtime_checkable
class SupportsStabilityTest(Protocol):
    device_name: str
    control_dim: int
    state_mean: NDArray[np.float64]
    state_std: NDArray[np.float64]
    control_mean: NDArray[np.float64]
    control_std: NDArray[np.float64]
    initializer: nn.Module
    predictor: nn.Module


class StabilityTestConfig(BaseTestConfig):
    type: Literal['incremental', 'bibo']
    optimization_steps: int
    optimization_lr: float
    initial_mean_delta: float
    initial_std_delta: float
    clip_gradient_norm: float
    regularization_scale: float
    evaluation_sequence: Union[Literal['all'], int]


class StabilityTest(BaseTest):
    CONFIG = StabilityTestConfig

    def __init__(self, config: StabilityTestConfig) -> None:
        super().__init__(config)

        self.config = config
        self.window_size = config.window_size
        self.horizon_size = config.horizon_size
        self.evaluation_sequence = config.evaluation_sequence

    def test(self, model: DynamicIdentificationModel, simulations: List[TestSimulation]) -> TestResult:
        if not (isinstance(model, LSTMInitModel) or isinstance(model, ConstrainedRnn)):
            return dict()

        controls = []
        pred_states = []
        stability_gains = []

        if isinstance(self.evaluation_sequence, int):
            logger.info(
                f'Test stability for sequence number {self.evaluation_sequence}'
            )
            dataset = list(split_simulations(self.window_size, self.horizon_size, simulations))
            (
                initial_control,
                initial_state,
                true_control,
                _,
                _,
            ) = dataset[self.evaluation_sequence]
            gamma_2, control, pred_state = optimize_input_disturbance(
                config=self.config,
                model=model,
                initial_control=initial_control,
                initial_state=initial_state,
                true_control=true_control,
            )
            stability_gains.append(gamma_2)
            controls.append(control)
            pred_states.append(pred_state)

        elif self.evaluation_sequence == 'all':
            logger.info(
                f'Test stability for {self.evaluation_sequence} sequences'
            )
            for (
                    idx_data,
                    (
                            initial_control,
                            initial_state,
                            true_control,
                            _,
                            _,
                    ),
            ) in enumerate(split_simulations(self.window_size, self.horizon_size, simulations)):
                logger.info(f'Sequence number: {idx_data}')

                gamma_2, control, pred_state = optimize_input_disturbance(
                    config=self.config,
                    model=model,
                    initial_control=initial_control,
                    initial_state=initial_state,
                    true_control=true_control,
                )

                stability_gains.append(gamma_2)
                controls.append(control)
                pred_states.append(pred_state)


def optimize_input_disturbance(
    config: StabilityTestConfig,
    model: DynamicIdentificationModel,
    initial_control: NDArray[np.float64],
    initial_state: NDArray[np.float64],
    true_control: NDArray[np.float64],
) -> Optional[Tuple[np.float64, NDArray[np.float64], NDArray[np.float64]]]:
    if not isinstance(model, SupportsStabilityTest):
        return None

    if not isinstance(model.initializer, SupportsRNNForward):
        return None

    if not isinstance(model.predictor, SupportsRNNForward):
        return None

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

    for step_idx in range(config.optimization_steps):

        # calculate stability gain
        def sequence_norm(x):
            norm = torch.tensor(0).float().to(model.device_name)
            for x_k in x:
                norm += (torch.linalg.norm(x_k) ** 2).float()
            return norm

        u_a = u_norm + delta
        if config.type == 'incremental':
            u_b = u_norm.clone()
            # model prediction
            _, hx = model.initializer(u_init_norm, return_state=True)
            # TODO set initial state to zero should be good to find unstable sequences
            hx = (
                torch.zeros_like(hx[0]).to(model.device_name),
                torch.zeros_like(hx[1]).to(model.device_name),
            )
            y_hat_a = model.predictor(u_a, hx=hx, return_state=False).squeeze()
            y_hat_b = model.predictor(u_b, hx=hx, return_state=False).squeeze()

            # use log to avoid zero in the denominator (goes to -inf)
            # since we maximize this results in a punishment
            regularization = config.regularization_scale * torch.log(
                sequence_norm(u_a - u_b)
            )
            gamma_2 = sequence_norm(y_hat_a - y_hat_b) / sequence_norm(u_a - u_b)
            L = gamma_2 + regularization
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

        elif config.type == 'bibo':
            # model prediction
            _, hx = model.initializer(u_init_norm, return_state=True)
            # TODO set initial state to zero should be good to find unstable sequences
            hx = (
                torch.zeros_like(hx[0]).to(model.device_name),
                torch.zeros_like(hx[1]).to(model.device_name),
            )
            y_hat_a = model.predictor(u_a, hx=hx, return_state=False).squeeze()

            # use log to avoid zero in the denominator (goes to -inf)
            # since we maximize this results in a punishment
            regularization = config.regularization_scale * torch.log(
                sequence_norm(u_a)
            )
            gamma_2 = sequence_norm(y_hat_a) / sequence_norm(u_a)
            L = gamma_2 + regularization
            L.backward()
            torch.nn.utils.clip_grad_norm_(delta, config.clip_gradient_norm)
            opt.step()

            control = utils.denormalize(
                u_a.cpu().detach().numpy().squeeze(),
                model.control_mean,
                model.control_std,
            )
            pred_state = utils.denormalize(
                y_hat_a.cpu().detach().numpy(), model.state_mean, model.state_std
            )
        else:
            raise ValueError(f'Stability type {config.type} does not exist.')

        gamma_2 = gamma_2.cpu().detach().numpy()
        if step_idx % 100 == 0:
            logger.info(
                f'step: {step_idx} \t '
                f'gamma_2: {gamma_2:.3f} \t '
                f'gradient norm: {torch.norm(delta.grad):.3f} '
                f'\t -log(norm(denominator)): {regularization:.3f}'
            )

    return gamma_2, control, pred_state


