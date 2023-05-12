from typing import List

import numpy as np
from numpy.typing import NDArray

from deepsysid.explainability.base import (
    BaseExplainer,
    ExplainerNotImplementedForModel,
    Explanation,
)

from ...models.switching.switchrnn import SwitchingLSTMBaseModel
from ..base import DynamicIdentificationModel
from .utils import denormalize_control_weights, denormalize_state_weights


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

        control_dim = initial_control.shape[1]
        state_dim = model.state_dim
        window_size = initial_state.shape[0]
        horizon_size = control.shape[0]

        _, metadata = model.simulate(initial_control, initial_state, control)
        system_matrices = [mat for mat in metadata['system_matrices'].squeeze(0)]
        control_matrices = [mat for mat in metadata['control_matrices'].squeeze(0)]
        output_matrix = model.predictor.output_matrix.cpu().detach().numpy()
        weights = self.construct_feature_weights(
            system_matrices, control_matrices, output_matrix
        )

        intercept = model.state_mean

        weights_initial_state = np.zeros(
            (state_dim, window_size, state_dim), dtype=np.float64
        )
        initial_state_den, initial_state_intercept = denormalize_state_weights(
            model.state_mean, model.state_std, weights[:, :state_dim]
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
            control_den, control_intercept = denormalize_control_weights(
                model.state_std,
                model.control_mean,
                model.control_std,
                weights[:, begin_idx:end_idx],
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
        output_matrix: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        weights: List[List[np.ndarray]] = []
        for At, Bt in zip(system_matrices, control_matrices):
            if len(weights) == 0:
                weights.append([At, Bt])
            else:
                weights.append([(At @ weight) for weight in weights[-1]] + [Bt])

        # Currently, weights map to state space. We need them in output space.
        # Map them to output space as follows:
        feature_weights = [output_matrix @ weight for weight in weights[-1]]
        feature_weights_np = np.hstack(feature_weights)
        return feature_weights_np
