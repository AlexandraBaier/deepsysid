from typing import List, Optional

import numpy as np
from numpy.typing import NDArray
from sklearn.linear_model import LassoCV

from ...models import utils
from ...models.base import DynamicIdentificationModel
from ..base import BaseExplainer, BaseExplainerConfig, Explanation, ModelInput
from .utils import denormalize_control_weights, denormalize_state_weights


class LIMEExplainerConfig(BaseExplainerConfig):
    num_samples: int
    cv_folds: Optional[int] = 5


class LIMEExplainer(BaseExplainer):
    CONFIG = LIMEExplainerConfig

    def __init__(self, config: LIMEExplainerConfig) -> None:
        super().__init__(config)

        self.num_samples = config.num_samples
        self.cv_folds = config.cv_folds

        self.state_mean: Optional[NDArray[np.float64]] = None
        self.state_std: Optional[NDArray[np.float64]] = None
        self.control_mean: Optional[NDArray[np.float64]] = None
        self.control_std: Optional[NDArray[np.float64]] = None
        self.input_mean: Optional[NDArray[np.float64]] = None
        self.input_std: Optional[NDArray[np.float64]] = None
        self.output_mean: Optional[NDArray[np.float64]] = None
        self.output_std: Optional[NDArray[np.float64]] = None

    def initialize(
        self,
        training_inputs: List[ModelInput],
        training_outputs: List[NDArray[np.float64]],
    ) -> None:
        window_size: int = training_inputs[0].initial_state.shape[0]
        horizon_size: int = training_inputs[0].control.shape[0]

        controls = [inp.initial_control for inp in training_inputs] + [
            inp.control for inp in training_inputs
        ]
        states = [inp.initial_state for inp in training_inputs] + [
            out for out in training_outputs
        ]

        self.control_mean, self.control_std = utils.mean_stddev(controls)
        self.state_mean, self.state_std = utils.mean_stddev(states)

        input_mean: List[NDArray[np.float64]] = (
            window_size * [self.control_mean]
            + window_size * [self.state_mean]
            + horizon_size * [self.control_mean]
        )
        self.input_mean = np.hstack(input_mean)

        input_std: List[NDArray[np.float64]] = (
            window_size * [self.control_std]
            + window_size * [self.state_std]
            + horizon_size * [self.control_std]
        )
        self.input_std = np.hstack(input_std)
        self.output_mean = self.state_mean
        self.output_std = self.state_std

    def explain(
        self,
        model: DynamicIdentificationModel,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> Explanation:
        if (
            self.state_mean is None
            or self.state_std is None
            or self.control_mean is None
            or self.control_std is None
            or self.input_mean is None
            or self.input_std is None
            or self.output_mean is None
            or self.output_std is None
        ):
            raise ValueError(
                'Explainer needs to be initialized with initialize' 'before explaining.'
            )
        x_ref = np.hstack(
            (initial_control.flatten(), initial_state.flatten(), control.flatten())
        )

        control_dim = initial_control.shape[1]
        state_dim = initial_state.shape[1]
        window_size = initial_state.shape[0]
        horizon_size = control.shape[0]
        input_dim = x_ref.shape[0]
        output_dim = state_dim

        x_ref = utils.normalize(x_ref, self.input_mean, self.input_std)
        disturbances = np.random.normal(0.0, 1.0, (self.num_samples, input_dim))
        x_all = x_ref + disturbances
        x_all_dist_to_ref = (1.0 + 1e-05) / (
            np.linalg.norm(disturbances, ord=2, axis=1) + 1e-05
        )
        x_all_den = utils.denormalize(x_all, self.input_mean, self.input_std)

        y_all = np.zeros((self.num_samples, output_dim))
        for idx in range(self.num_samples):
            initial_control_i = x_all_den[idx, : window_size * control_dim].reshape(
                (window_size, control_dim)
            )
            initial_state_i = x_all_den[
                idx, window_size * control_dim : window_size * (control_dim + state_dim)
            ].reshape((window_size, state_dim))
            control_i = x_all_den[
                idx, window_size * (state_dim + control_dim) :
            ].reshape((horizon_size, control_dim))

            model_output = model.simulate(
                initial_control_i, initial_state_i, control_i, None, None
            )
            if isinstance(model_output, np.ndarray):
                y_all[idx, :] = model_output[-1]
            else:
                y_all[idx, :] = model_output[0][-1]

            y_all[idx, :] = utils.normalize(
                y_all[idx, :], self.output_mean, self.output_std
            )

        weights_initial_control = np.zeros(
            (state_dim, window_size, control_dim), dtype=np.float64
        )
        weights_initial_state = np.zeros(
            (state_dim, window_size, state_dim), dtype=np.float64
        )
        weights_control = np.zeros(
            (state_dim, horizon_size, control_dim), dtype=np.float64
        )
        intercept = self.state_mean.copy()

        for out_idx in range(state_dim):
            local_estimator = LassoCV(fit_intercept=True, cv=self.cv_folds)
            local_estimator.fit(
                x_all, y_all[:, out_idx], sample_weight=x_all_dist_to_ref
            )

            weights_initial_control[out_idx, :, :] = local_estimator.coef_[
                : window_size * control_dim
            ].reshape((window_size, control_dim))
            weights_initial_state[out_idx, :, :] = local_estimator.coef_[
                window_size * control_dim : window_size * (state_dim + control_dim)
            ].reshape((window_size, state_dim))
            weights_control[out_idx, :, :] = local_estimator.coef_[
                window_size * (state_dim + control_dim) :
            ].reshape((horizon_size, control_dim))
            intercept[out_idx] = (
                intercept[out_idx]
                + local_estimator.intercept_ * self.state_std[out_idx]
            )

        for time in range(window_size):
            (
                weights_initial_control[:, time, :],
                intercept_delta,
            ) = denormalize_control_weights(
                state_std=self.state_std,
                control_mean=self.control_mean,
                control_std=self.control_std,
                control_matrix=weights_initial_control[:, time, :],
            )
            intercept = intercept + intercept_delta
            (
                weights_initial_state[:, time, :],
                intercept_delta,
            ) = denormalize_state_weights(
                state_mean=self.state_mean,
                state_std=self.state_std,
                state_matrix=weights_initial_state[:, time, :],
            )
            intercept = intercept + intercept_delta

        for time in range(horizon_size):
            weights_control[:, time, :], intercept_delta = denormalize_control_weights(
                state_std=self.state_std,
                control_mean=self.control_mean,
                control_std=self.control_std,
                control_matrix=weights_control[:, time, :],
            )
            intercept = intercept + intercept_delta

        return Explanation(
            weights_initial_control=weights_initial_control,
            weights_initial_state=weights_initial_state,
            weights_control=weights_control,
            intercept=intercept,
        )
