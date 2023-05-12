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
            model.output_mean is None
            or model.output_std is None
            or model.control_mean is None
            or model.control_std is None
        ):
            raise ValueError(
                'SwitchingLSTMBaseModel needs to be trained '
                'prior to computing explanations.'
            )

        control_dim = initial_control.shape[1]
        state_dim = model.predictor.state_dimension
        output_dim = model.predictor.output_dimension
        window_size = initial_state.shape[0]
        horizon_size = control.shape[0]

        _, metadata = model.simulate(initial_control, initial_state, control)
        system_matrices = [mat for mat in metadata['system_matrices'].squeeze(0)]
        control_matrices = [mat for mat in metadata['control_matrices'].squeeze(0)]
        output_matrix = model.predictor.output_matrix.cpu().detach().numpy()
        weights = self.construct_feature_weights(
            system_matrices, control_matrices, output_matrix
        )

        intercept = model.output_mean

        # Warning: Naming ambiguity here.
        # output_dim refers to len(state_names) and therefore what
        # we call initial_state. However, state_dim refers to
        # the switched system state, which is larger than output_dim.
        # For the initial state, we set the first len(state_names)
        # values to initial_state and the remainder to 0. Accordingly,
        # we truncate the weights by output_dim, but later for the
        # control weights, we start at state_dim. The section
        # output_dim:state_dim is ignored for time step 0 by the
        # switched system.
        weights_initial_state = np.zeros(
            (output_dim, window_size, output_dim), dtype=np.float64
        )
        initial_state_den, initial_state_intercept = denormalize_state_weights(
            model.output_mean, model.output_std, weights[:, :output_dim]
        )
        weights_initial_state[:, -1, :] = initial_state_den
        intercept = intercept + initial_state_intercept

        weights_initial_control = np.zeros(
            (output_dim, window_size, control_dim), dtype=np.float64
        )

        weights_true_control = np.zeros(
            (output_dim, horizon_size, control_dim), dtype=np.float64
        )
        for time in range(horizon_size):
            begin_idx = state_dim + time * control_dim
            end_idx = state_dim + (time + 1) * control_dim
            control_den, control_intercept = denormalize_control_weights(
                model.output_std,
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
