import logging
from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from deepsysid.explainability.base import (
    BaseExplainer,
    BaseExplanationMetric,
    BaseExplanationMetricConfig,
    ModelInput,
)
from deepsysid.models.base import DynamicIdentificationModel

logger = logging.getLogger(__name__)


class NMSEInfidelityMetric(BaseExplanationMetric):
    def __init__(self, config: BaseExplanationMetricConfig):
        super().__init__(config)

        self.state_dim = len(config.state_names)

    def measure(
        self,
        model: DynamicIdentificationModel,
        explainer: BaseExplainer,
        model_inputs: List[ModelInput],
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:

        model_predictions = [
            model.simulate(
                model_input.initial_control,
                model_input.initial_state,
                model_input.control,
                model_input.x0,
                model_input.initial_x0,
            )[0]
            for model_input in model_inputs
        ]
        explanations = []
        for idx, model_input in enumerate(model_inputs):
            explanations.append(
                explainer.explain(
                    model,
                    model_input.initial_control,
                    model_input.initial_state,
                    model_input.control,
                )
            )
            logger.info(
                f'NMSEInfidelityMetric: '
                f'Computed {(idx + 1) / len(model_inputs):.2%} explanations.'
            )

        n = len(model_inputs)
        std_y = np.std(np.vstack(model_predictions), axis=0)

        nmse = np.zeros((self.state_dim,), dtype=np.float64)
        for model_input, y, expl in zip(model_inputs, model_predictions, explanations):
            initial_control_contr = (
                expl.weights_initial_control.reshape(self.state_dim, -1)
                @ model_input.initial_control.flatten()
            )
            initial_state_contr = (
                expl.weights_initial_state.reshape(self.state_dim, -1)
                @ model_input.initial_state.flatten()
            )
            control_contr = (
                expl.weights_control.reshape(self.state_dim, -1)
                @ model_input.control.flatten()
            )
            yhat = (
                initial_control_contr
                + initial_state_contr
                + control_contr
                + expl.intercept
            )
            nmse = nmse + (yhat - y[-1]) * (yhat - y[-1])

        nmse = (1.0 / ((std_y**2) * n)) * nmse

        return nmse, dict(
            initial_controls=np.array(
                [inp.initial_control for inp in model_inputs], dtype=np.float64
            ),
            initial_states=np.array(
                [inp.initial_state for inp in model_inputs], dtype=np.float64
            ),
            controls=np.array([inp.control for inp in model_inputs], dtype=np.float64),
            weights_initial_control=np.array(
                [expl.weights_initial_control for expl in explanations],
                dtype=np.float64,
            ),
            weights_initial_state=np.array(
                [expl.weights_initial_state for expl in explanations], dtype=np.float64
            ),
            weights_control=np.array(
                [expl.weights_control for expl in explanations], dtype=np.float64
            ),
            intercepts=np.array(
                [expl.intercept for expl in explanations], dtype=np.float64
            ),
        )


class LipschitzEstimateMetricConfig(BaseExplanationMetricConfig):
    n_disturbances: int
    control_error_std: List[float]
    state_error_std: List[float]


class LipschitzEstimateMetric(BaseExplanationMetric):
    CONFIG = LipschitzEstimateMetricConfig

    def __init__(self, config: LipschitzEstimateMetricConfig) -> None:
        super().__init__(config)

        self.n_disturbances = config.n_disturbances
        self.control_error_std = np.array(config.control_error_std, dtype=np.float64)
        self.state_error_std = np.array(config.state_error_std, dtype=np.float64)

    def measure(
        self,
        model: DynamicIdentificationModel,
        explainer: BaseExplainer,
        model_inputs: List[ModelInput],
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        window_size = model_inputs[0].initial_state.shape[0]
        horizon_size = model_inputs[0].control.shape[0]
        state_dim = model_inputs[0].initial_state.shape[1]
        control_dim = model_inputs[0].initial_control.shape[1]

        lipschitz_estimates: List[float] = []
        for idx, model_input in enumerate(model_inputs):
            explanation_orig = explainer.explain(
                model,
                model_input.initial_control,
                model_input.initial_state,
                model_input.control,
            )
            weights_orig = np.hstack(
                (
                    explanation_orig.weights_initial_control.flatten(),
                    explanation_orig.weights_initial_state.flatten(),
                    explanation_orig.weights_control.flatten(),
                )
            )

            for disturbance_idx in range(self.n_disturbances):
                initial_control_error = np.random.normal(
                    0.0, self.control_error_std, size=(window_size, control_dim)
                )
                initial_state_error = np.random.normal(
                    0.0, self.state_error_std, size=(window_size, state_dim)
                )
                control_error = np.random.normal(
                    0.0, self.control_error_std, size=(horizon_size, control_dim)
                )

                initial_control_dist = (
                    model_input.initial_control + initial_control_error
                )
                initial_state_dist = model_input.initial_state + initial_state_error
                control_dist = model_input.control + control_error

                explanation_dist = explainer.explain(
                    model,
                    initial_control_dist,
                    initial_state_dist,
                    control_dist,
                )
                error = np.hstack(
                    (
                        initial_control_error.flatten(),
                        initial_state_error.flatten(),
                        control_error.flatten(),
                    )
                )
                weights_dist = np.hstack(
                    (
                        explanation_dist.weights_initial_control.flatten(),
                        explanation_dist.weights_initial_state.flatten(),
                        explanation_dist.weights_control.flatten(),
                    )
                )

                error_norm = np.linalg.norm(error, ord=2)
                weights_distance = np.linalg.norm(weights_orig - weights_dist, ord=2)
                lipschitz_estimate = float(weights_distance / error_norm)
                lipschitz_estimates.append(lipschitz_estimate)

            logger.info(
                f'LipschitzEstimateMetric: '
                f'Computed {(idx + 1) / len(model_inputs):.2%} explanations.'
            )

        return np.array(lipschitz_estimates, dtype=np.float64), dict(
            largest_lipschitz_estimate=np.array(
                [max(lipschitz_estimates)], dtype=np.float64
            )
        )


class ExplanationComplexityMetricConfig(BaseExplanationMetricConfig):
    relevance_threshold: float


class ExplanationComplexityMetric(BaseExplanationMetric):
    CONFIG = ExplanationComplexityMetricConfig

    def __init__(self, config: ExplanationComplexityMetricConfig) -> None:
        super().__init__(config)

        self.relevance_threshold = config.relevance_threshold

    def measure(
        self,
        model: DynamicIdentificationModel,
        explainer: BaseExplainer,
        model_inputs: List[ModelInput],
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        complexities: List[float] = []
        for idx, model_input in enumerate(model_inputs):
            explanation = explainer.explain(
                model,
                model_input.initial_control,
                model_input.initial_state,
                model_input.control,
            )
            weights = np.hstack(
                (
                    explanation.weights_initial_control.flatten(),
                    explanation.weights_initial_state.flatten(),
                    explanation.weights_control.flatten(),
                )
            )
            relative_importance = np.abs(weights) / np.sum(np.abs(weights))
            complexity = float(np.mean(relative_importance >= self.relevance_threshold))
            complexities.append(complexity)

        simplicity = float(1.0 - np.mean(complexities))

        return np.array(complexities, dtype=np.float64), dict(
            simplicity=np.array([simplicity], dtype=np.float64)
        )
