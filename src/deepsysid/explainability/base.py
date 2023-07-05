import abc
import dataclasses
import warnings
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from deepsysid.models.base import DynamicIdentificationModel


class ExplainerNotImplementedForModel(ValueError):
    pass


@dataclasses.dataclass
class ModelInput:
    initial_control: NDArray[np.float64]
    initial_state: NDArray[np.float64]
    control: NDArray[np.float64]
    x0: Optional[NDArray[np.float64]]


@dataclasses.dataclass
class AdditiveFeatureAttributionExplanation:
    """
    Explanation weights should be similar to Shapley values (or Shapley values).
    This mean the sum of all weights and the intercept
    should reconstruct as closely as possible the original prediction.
    For Shapley values, which are efficient/faithful, the reconstruction is perfect.
    For LIME, the reconstruction might not be exact.

    weights_initial_control (N, W, M)
    weights_initial_state (N, W, N)
    weights_control (N, H, M)
    intercept (N,)

    with N = state_dim, M = control_dim, W = window, H = horizon
    """

    weights_initial_control: NDArray[np.float64]
    weights_initial_state: NDArray[np.float64]
    weights_control: NDArray[np.float64]
    intercept: NDArray[np.float64]

    def __post_init__(self) -> None:
        if not (
            self.weights_initial_state.shape[0]
            == self.weights_initial_control.shape[0]
            == self.weights_control.shape[0]
            == self.intercept.shape[0]
            == self.weights_initial_state.shape[2]
        ):
            raise ValueError(
                'State dimension must match across all weights and intercept. '
                f'But it does not match: {self.weights_initial_control.shape}, '
                f'{self.weights_initial_state.shape}, {self.weights_control.shape}, '
                f'{self.intercept.shape}.'
            )
        if not (self.weights_initial_control.shape[2] == self.weights_control.shape[2]):
            raise ValueError(
                'Control dimension must match for initial_control and control. '
                f'But it does not match: {self.weights_initial_control.shape}, '
                f'{self.weights_control.shape}.'
            )
        if not (
            self.weights_initial_state.shape[1] == self.weights_initial_control.shape[1]
        ):
            raise ValueError(
                'Window dimension of initial feature weights must match. '
                f'But it does not match: {self.weights_initial_control.shape}, '
                f'{self.weights_initial_state.shape}.'
            )


class AdditiveFeatureAttributionExplainerConfig(BaseModel):
    pass


class AdditiveFeatureAttributionExplainer(metaclass=abc.ABCMeta):
    CONFIG = AdditiveFeatureAttributionExplainerConfig

    def __init__(
        self,
        config: AdditiveFeatureAttributionExplainerConfig,
    ):
        pass

    def initialize(
        self,
        training_inputs: List[ModelInput],
        training_outputs: List[NDArray[np.float64]],
    ) -> None:
        """
        :param training_inputs: list of length n_samples with ModelInput
            with initial_control (window, control)
            with initial_state (window, state)
            with control (horizon, control)
        :param training_outputs: list of length n_samples
            with arrays (state_dim,). The array corresponds to the output
            at the last prediction time step.
        :return:
        """
        pass

    @abc.abstractmethod
    def explain(
        self,
        model: DynamicIdentificationModel,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> AdditiveFeatureAttributionExplanation:
        """
        Return feature weights for the most recent
            initial state and sequence of control inputs.
        Returned feature weights should be able to reproduce
            unnormalized model prediction to some degree of accuracy.
        May raise ExplainerNotImplementedForModel
            if it does not support a specific type of model.

        W = initial window
        M = control dimension
        N = state dimension
        H = horizon
        :param model:
        :param initial_control: (W, M)
        :param initial_state: (W, N)
        :param control: (H, M)
        :return: an Explanation
        """
        pass


class BaseExplanationMetricConfig(BaseModel):
    state_names: List[str]


class BaseExplanationMetric(metaclass=abc.ABCMeta):
    CONFIG: Type[BaseExplanationMetricConfig] = BaseExplanationMetricConfig

    def __init__(self, config: BaseExplanationMetricConfig):
        pass

    @abc.abstractmethod
    def measure(
        self,
        model: DynamicIdentificationModel,
        explainer: AdditiveFeatureAttributionExplainer,
        model_inputs: List[ModelInput],
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        pass


def retrieve_explainer_class(
    explainer_class_string: str,
) -> Type[AdditiveFeatureAttributionExplainer]:
    # https://stackoverflow.com/a/452981
    parts = explainer_class_string.split('.')
    module_string = '.'.join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, AdditiveFeatureAttributionExplainer):
        raise ValueError(f'{cls} is not a subclass of BaseExplainer.')
    return cls  # type: ignore


def retrieve_explanation_metric_class(
    explanation_metric_class_string: str,
) -> Type[BaseExplanationMetric]:
    # https://stackoverflow.com/a/452981
    parts = explanation_metric_class_string.split('.')
    module_string = '.'.join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, BaseExplanationMetric):
        raise ValueError(f'{cls} is not a subclass of BaseExplanationMetric.')
    return cls  # type: ignore


class Explanation(AdditiveFeatureAttributionExplanation):
    def __post_init__(self) -> None:
        warnings.warn(
            'deepsysid.explainability.explainers.base.Explanation has been deprecated '
            'and replaced with deepsysid.explainability.explainers.base.'
            'AdditiveFeatureAttributionExplanation.',
            category=DeprecationWarning,
        )
        super().__post_init__()


class BaseExplainerConfig(AdditiveFeatureAttributionExplainerConfig):
    def __init__(self, **data: Dict[str, Any]) -> None:
        warnings.warn(
            'deepsysid.explainability.explainers.base.'
            'BaseExplainerConfig has been deprecated '
            'and replaced with deepsysid.explainability.explainers.base.'
            'AdditiveFeatureAttributionExplainerConfig.',
            category=DeprecationWarning,
        )
        super().__init__(**data)


class BaseExplainer(AdditiveFeatureAttributionExplainer, abc.ABC):
    def __init__(self, config: AdditiveFeatureAttributionExplainerConfig) -> None:
        warnings.warn(
            'deepsysid.explainability.explainers.base.BaseExplainer '
            'has been deprecated and replaced with deepsysid.'
            'explainability.explainers.base.'
            'AdditiveFeatureAttributionExplainer.',
            category=DeprecationWarning,
        )
        super().__init__(config)
