from typing import List, Tuple

import numpy as np
from numpy._typing import NDArray

from deepsysid.explainability.base import (
    BaseExplainer,
    ExplainerNotImplementedForModel,
    Explanation,
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
