from typing import List, Optional, Tuple

import lime
import lime.lime_tabular
import numpy as np
from numpy.typing import NDArray

from deepsysid.explainability.base import (
    BaseExplainer,
    BaseExplainerConfig,
    ExplainerNotImplementedForModel,
    Explanation,
    ModelInput,
)
from deepsysid.models.base import DynamicIdentificationModel
from deepsysid.models.switching.switchrnn import SwitchingLSTMBaseModel


class SwitchingLSTMExplainer(BaseExplainer):
    def explain(
        self,
        model: DynamicIdentificationModel,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> Explanation:
        if not isinstance(model, SwitchingLSTMBaseModel):
            raise ExplainerNotImplementedForModel(
                'SwitchingLSTMExplainer can only explain models '
                'that subclass SwitchingLSTMBaseModel.'
            )

        if model.state_mean is None:
            raise ValueError('Model has not been trained and cannot explain.')

        state_dim = initial_state.shape[1]
        control_dim = initial_control.shape[1]
        window_size = initial_state.shape[0]
        horizon_size = control.shape[0]

        _, metadata = model.simulate(initial_control, initial_state, control)
        system_matrices = [mat for mat in metadata['system_matrices'].squeeze(0)]
        control_matrices = [mat for mat in metadata['control_matrices'].squeeze(0)]
        weights = self.construct_feature_weights(system_matrices, control_matrices)

        intercept = model.state_mean

        weights_initial_state = np.zeros(
            (state_dim, window_size, state_dim), dtype=np.float64
        )
        initial_state_den, initial_state_intercept = self.denormalize_state_weights(
            model, weights[:, :state_dim]
        )
        weights_initial_state[:, -1, :] = initial_state_den
        intercept = intercept + initial_state_intercept

        weights_initial_control = np.zeros(
            (state_dim, window_size, control_dim), dtype=np.float64
        )

        weights_true_control = np.zeros(
            (state_dim, horizon_size, control_dim), dtype=np.float64
        )
        for time in range(horizon_size):
            begin_idx = state_dim + time * control_dim
            end_idx = state_dim + (time + 1) * control_dim
            control_den, control_intercept = self.denormalize_control_weights(
                model, weights[:, begin_idx:end_idx]
            )
            weights_true_control[:, time, :] = control_den
            intercept = intercept + control_intercept

        return Explanation(
            weights_initial_control=weights_initial_control,
            weights_initial_state=weights_initial_state,
            weights_control=weights_true_control,
            intercept=intercept,
        )

    def construct_feature_weights(
        self,
        system_matrices: List[NDArray[np.float64]],
        control_matrices: List[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        weights: List[List[np.ndarray]] = []
        for At, Bt in zip(system_matrices, control_matrices):
            if len(weights) == 0:
                weights.append([At, Bt])
            else:
                weights.append([(At @ weight) for weight in weights[-1]] + [Bt])

        weights_np = np.hstack(weights[-1])
        return weights_np

    def denormalize_control_weights(
        self, model: SwitchingLSTMBaseModel, control_matrix: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        if (
            model.state_mean is None
            or model.state_std is None
            or model.control_mean is None
            or model.control_std is None
        ):
            raise ValueError(
                'SwitchingLSTMBaseModel needs to be trained '
                'prior to computing explanations.'
            )
        state_std = model.state_std
        control_mean = model.control_mean
        control_std = model.control_std
        state_dim = control_matrix.shape[0]
        control_dim = control_matrix.shape[1]

        control_matrix_den = np.zeros(control_matrix.shape)
        intercept = np.zeros((state_dim,))
        for out_idx in range(state_dim):
            for in_idx in range(control_dim):
                weight = (
                    state_std[out_idx]
                    / control_std[in_idx]
                    * control_matrix[out_idx, in_idx]
                )
                control_matrix_den[out_idx, in_idx] = weight
                intercept[out_idx] = intercept[out_idx] - weight * control_mean[in_idx]

        return control_matrix_den, intercept

    def denormalize_state_weights(
        self, model: SwitchingLSTMBaseModel, state_matrix: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        if (
            model.state_mean is None
            or model.state_std is None
            or model.control_mean is None
            or model.control_std is None
        ):
            raise ValueError(
                'SwitchingLSTMBaseModel needs to be trained '
                'prior to computing explanations.'
            )
        state_mean = model.state_mean
        state_std = model.state_std
        state_dim = state_mean.shape[0]

        state_matrix_den = np.zeros(state_matrix.shape)
        intercept = np.zeros((state_dim,))
        for out_idx in range(state_dim):
            for in_idx in range(state_dim):
                weight = (
                    state_std[out_idx]
                    / state_std[in_idx]
                    * state_matrix[out_idx, in_idx]
                )
                state_matrix_den[out_idx, in_idx] = weight
                intercept[out_idx] = intercept[out_idx] - weight * state_mean[in_idx]

        return state_matrix_den, intercept


class LimeExplainerConfig(BaseExplainerConfig):
    num_samples: int
    # If num_features is None, all features are used.
    num_features: Optional[int] = None


class LIMEExplainer(BaseExplainer):
    CONFIG = LimeExplainerConfig

    def __init__(self, config: LimeExplainerConfig):
        super().__init__(config)

        self.num_samples = config.num_samples
        self.lime_explainers: Optional[
            List[lime.lime_tabular.LimeTabularExplainer]
        ] = None
        self.num_features = config.num_features

    def initialize(
        self,
        training_inputs: List[ModelInput],
        training_outputs: List[NDArray[np.float64]],
    ) -> None:
        training_data = np.vstack(
            [
                np.hstack(
                    (
                        inp.initial_control.flatten(),
                        inp.initial_state.flatten(),
                        inp.control.flatten(),
                    )
                )
                for inp in training_inputs
            ]
        )
        training_labels = np.vstack([out[-1, :] for out in training_outputs])

        self.lime_explainers = [
            lime.lime_tabular.LimeTabularExplainer(
                training_data=training_data,
                mode='regression',
                training_labels=training_labels[:, out_idx],
            )
            for out_idx in range(training_labels.shape[1])
        ]

    def explain(
        self,
        model: DynamicIdentificationModel,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> Explanation:
        if self.lime_explainers is None:
            raise ValueError(
                'Explainer needs to be initialized with initialize' 'before explaining.'
            )

        control_dim = initial_control.shape[1]
        state_dim = initial_state.shape[1]
        window_size = initial_state.shape[0]
        horizon_size = control.shape[0]

        if self.num_features is None:
            num_features_lime = (
                control_dim * window_size
                + state_dim * window_size
                + control_dim * horizon_size
            )
        else:
            num_features_lime = self.num_features

        def predict_fn(x: NDArray[np.float64], out_idx: int) -> NDArray[np.float64]:
            if len(x.shape) != 2:
                raise ValueError(
                    'Expected 2D array from LIMEs explain_instance method.'
                )
            batch_size = x.shape[0]

            initial_control_index = window_size * control_dim
            initial_state_index = initial_control_index + window_size * state_dim
            x_initial_control = x[:, :initial_control_index].reshape(
                (batch_size, window_size, control_dim)
            )
            x_initial_state = x[:, initial_control_index:initial_state_index].reshape(
                (batch_size, window_size, state_dim)
            )
            x_control = x[:, initial_state_index:].reshape(
                (batch_size, horizon_size, control_dim)
            )
            output = np.zeros((batch_size,))
            for idx in range(batch_size):
                pred, _ = model.simulate(
                    x_initial_control[idx], x_initial_state[idx], x_control[idx]
                )
                output[idx] = pred[-1, out_idx]

            return output

        data_row = np.hstack(
            (initial_control.flatten(), initial_state.flatten(), control.flatten())
        )
        lime_explanations = [
            lime_explainer.explain_instance(
                data_row=data_row,
                predict_fn=lambda x: predict_fn(x, out_idx),
                num_samples=self.num_samples,
                num_features=num_features_lime,
            )
            for out_idx, lime_explainer in enumerate(self.lime_explainers)
        ]
        weights = np.zeros(
            (
                state_dim,
                window_size * control_dim
                + window_size * state_dim
                + horizon_size * control_dim,
            ),
            dtype=np.float64,
        )
        intercept = np.zeros((state_dim,), dtype=np.float64)
        for out_idx, explanation in enumerate(lime_explanations):
            intercept[out_idx] = explanation.intercept[0]
            for in_idx, weight in explanation.local_exp[0]:
                # According to
                # https://github.com/abacusai/xai-bench/blob/main/custom_explainers/lime.py#L69
                # The outputs of LIME for regression are negated.
                # So we have to invert them again.
                weights[out_idx, in_idx] = -weight

        initial_control_idx = window_size * control_dim
        initial_state_idx = initial_control_idx + window_size * state_dim

        return Explanation(
            weights_initial_control=weights[:, :initial_control_idx].reshape(
                (state_dim, window_size, control_dim)
            ),
            weights_initial_state=weights[
                :, initial_control_idx:initial_state_idx
            ].reshape((state_dim, window_size, state_dim)),
            weights_control=weights[:, initial_state_idx:].reshape(
                (state_dim, horizon_size, control_dim)
            ),
            intercept=intercept,
        )
