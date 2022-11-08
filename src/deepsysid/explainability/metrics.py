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

        nmse = (1.0 / (std_y * n)) * nmse

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
