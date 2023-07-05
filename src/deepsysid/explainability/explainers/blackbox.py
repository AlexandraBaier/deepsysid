import abc
from typing import Any, Callable, Dict, List, Optional, Type

import numpy as np
from interpret.api.templates import FeatureValueExplanation
from interpret.blackbox import LimeTabular, ShapKernel
from numpy.typing import NDArray

from ...models.base import DynamicIdentificationModel
from ..base import (
    AdditiveFeatureAttributionExplainer,
    AdditiveFeatureAttributionExplainerConfig,
    AdditiveFeatureAttributionExplanation,
    ModelInput,
)

Predict = Callable[
    [NDArray[np.float64], Optional[NDArray[np.float64]]], NDArray[np.float64]
]


class TabularExplainer(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self, model: Predict, data: NDArray[np.float64], **kwargs: Dict[str, Any]
    ) -> None:
        pass

    @abc.abstractmethod
    def explain_local(
        self, X: NDArray[np.float64], **kwargs: Dict[str, Any]
    ) -> FeatureValueExplanation:
        pass


class BlackboxExplainer(AdditiveFeatureAttributionExplainer, abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        config: AdditiveFeatureAttributionExplainerConfig,
        tabular_explainer_cls: Type[TabularExplainer],
        init_kwargs: Dict[str, Any],
        explain_local_kwargs: Dict[str, Any],
        intercept_key: str,
    ) -> None:
        super().__init__(config)

        self.tabular_explainer_cls = tabular_explainer_cls
        self.init_kwargs = init_kwargs
        self.explain_local_kwargs = explain_local_kwargs
        self.intercept_key = intercept_key

        self._x_train: Optional[NDArray[np.float64]] = None
        self._y_train: Optional[NDArray[np.float64]] = None
        self._explainers: Optional[List[LimeTabular]] = None
        self._model: Optional[DynamicIdentificationModel] = None

    def initialize(
        self,
        training_inputs: List[ModelInput],
        training_outputs: List[NDArray[np.float64]],
    ) -> None:
        assert len(training_inputs) == len(
            training_outputs
        ), 'Expecting the same number of inputs and outputs.'
        assert len(training_inputs) > 0, 'Expecting at least one input/output pair.'

        window_size: int = training_inputs[0].initial_state.shape[0]
        horizon_size: int = training_inputs[0].control.shape[0]
        input_size: int = training_inputs[0].control.shape[1]
        output_size: int = training_inputs[0].initial_state.shape[1]

        input_array = np.zeros(
            (
                len(training_inputs),
                window_size * (output_size + input_size) + horizon_size * input_size,
            ),
            dtype=np.float64,
        )
        output_array = np.zeros((len(training_outputs), output_size), dtype=np.float64)
        for idx, (x, y) in enumerate(zip(training_inputs, training_outputs)):
            input_array[idx, :] = _flatten_model_input(x)
            output_array[idx, :] = y

        self._x_train = input_array
        self._y_train = output_array
        self._explainers = None
        self._model = None

    def explain(
        self,
        model: DynamicIdentificationModel,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
    ) -> AdditiveFeatureAttributionExplanation:
        if self._x_train is None or self._y_train is None:
            raise ValueError(
                'Explainer needs to be initialized with initialize' 'before explaining.'
            )

        window_size = initial_control.shape[0]
        horizon_size = control.shape[0]
        input_size = initial_control.shape[1]
        output_size = initial_state.shape[1]

        # If a different model (including model reference) is passed
        # to the explainer, we should just initialize the explainers again.
        if self._explainers is None or self._model != model:
            self._model = model
            self._explainers = [
                self.tabular_explainer_cls(
                    model=_construct_predict(
                        model=model,
                        output_idx=output_idx,
                        input_size=input_size,
                        output_size=output_size,
                        window_size=window_size,
                        horizon_size=horizon_size,
                    ),
                    data=self._x_train,
                    **self.init_kwargs,
                )
                for output_idx in range(output_size)
            ]

        self._model = model

        x = _flatten_model_input(
            ModelInput(
                initial_control=initial_control,
                initial_state=initial_state,
                control=control,
                x0=None,
            )
        ).reshape(1, -1)

        weights_initial_state = np.zeros(
            (output_size, window_size, output_size), dtype=np.float64
        )
        weights_initial_control = np.zeros(
            (output_size, window_size, input_size), dtype=np.float64
        )
        weights_control = np.zeros(
            (output_size, horizon_size, input_size), dtype=np.float64
        )
        intercept = np.zeros((output_size,), dtype=np.float64)
        for output_idx, explainer in enumerate(self._explainers):
            explanation = explainer.explain_local(X=x, **self.explain_local_kwargs)
            intercept[output_idx] = _explanation_to_intercept(
                explanation, intercept_key=self.intercept_key
            )
            attributions = _explanation_to_attributions(
                explanation, array_size=x.shape[1]
            )
            input_weights = _recover_flattened_model_input(
                x=attributions,
                window_size=window_size,
                horizon_size=horizon_size,
                input_size=input_size,
                output_size=output_size,
            )
            weights_initial_state[output_idx, :, :] = input_weights.initial_state
            weights_initial_control[output_idx, :, :] = input_weights.initial_control
            weights_control[output_idx, :, :] = input_weights.control

        return AdditiveFeatureAttributionExplanation(
            weights_initial_control=weights_initial_control,
            weights_initial_state=weights_initial_state,
            weights_control=weights_control,
            intercept=intercept,
        )


class TabularLIMEExplainer(TabularExplainer):
    def __init__(
        self, model: Predict, data: NDArray[np.float64], **kwargs: Dict[str, Any]
    ) -> None:
        self._lime = LimeTabular(model=model, data=data, **kwargs)

    def explain_local(
        self, X: NDArray[np.float64], **kwargs: Dict[str, Any]
    ) -> FeatureValueExplanation:
        return self._lime.explain_local(
            X=X,
            num_samples=kwargs['num_samples'],
            num_features=(
                kwargs['num_features']
                if kwargs['num_features'] is not None
                else X.shape[1]
            ),
        )


class LIMEExplainerConfig(AdditiveFeatureAttributionExplainerConfig):
    num_samples: int
    num_features: Optional[int] = None


class LIMEExplainer(BlackboxExplainer):
    CONFIG = LIMEExplainerConfig

    def __init__(self, config: LIMEExplainerConfig) -> None:
        super().__init__(
            config,
            tabular_explainer_cls=TabularLIMEExplainer,
            init_kwargs=dict(),
            explain_local_kwargs=dict(
                num_samples=config.num_samples, num_features=config.num_features
            ),
            intercept_key='Intercept',
        )


class KernelSHAPExplainer(BlackboxExplainer):
    def __init__(self, config: AdditiveFeatureAttributionExplainerConfig) -> None:
        super().__init__(
            config,
            tabular_explainer_cls=ShapKernel,
            init_kwargs=dict(),
            explain_local_kwargs=dict(),
            intercept_key='Base Value',
        )


def _explanation_to_attributions(
    explanation: FeatureValueExplanation, array_size: int
) -> NDArray[np.float64]:
    names = explanation.data(key=0)['names']
    scores = explanation.data(key=0)['scores']
    attributions = np.zeros((array_size,), dtype=np.float64)
    for name, score in zip(names, scores):
        try:
            original_idx = int(name.split('_')[1])
        except ValueError:
            raise ValueError(
                'Expected FeatureValueExplanation.data(key=?)["names"] '
                'to only contain strings of format "feature_XXXX" where '
                f'X are digits. Encountered {name} instead. This is '
                f'a bug in deepsysid.'
            )
        attributions[original_idx] = score
    return attributions


def _explanation_to_intercept(
    explanation: FeatureValueExplanation, intercept_key: str
) -> float:
    intercept_idx = explanation.data(key=0)['extra']['names'].index(intercept_key)
    return float(explanation.data(key=0)['extra']['scores'][intercept_idx])


def _construct_predict(
    model: DynamicIdentificationModel,
    output_idx: int,
    window_size: int,
    horizon_size: int,
    input_size: int,
    output_size: int,
) -> Callable[
    [NDArray[np.float64], Optional[NDArray[np.float64]]], NDArray[np.float64]
]:
    def _predict(
        x: NDArray[np.float64], sample_weight: Optional[NDArray[np.float64]] = None
    ) -> NDArray[np.float64]:
        assert len(x.shape) == 2, 'Expecting x to be 2-dimensional.'

        n_samples = x.shape[0]
        y = np.zeros((n_samples,))
        for sample_idx in range(n_samples):
            model_input = _recover_flattened_model_input(
                x=x[sample_idx, :],
                input_size=input_size,
                output_size=output_size,
                window_size=window_size,
                horizon_size=horizon_size,
            )
            model_output = model.simulate(
                initial_state=model_input.initial_state,
                initial_control=model_input.initial_control,
                control=model_input.control,
                x0=None,
            )
            if isinstance(model_output, np.ndarray):
                y[sample_idx] = model_output[-1, output_idx]
            else:
                y[sample_idx] = model_output[0][-1, output_idx]
        return y

    return _predict


def _flatten_model_input(model_input: ModelInput) -> NDArray[np.float64]:
    return np.hstack(
        (
            model_input.initial_state.flatten(),
            model_input.initial_control.flatten(),
            model_input.control.flatten(),
        )
    )


def _recover_flattened_model_input(
    x: NDArray[np.float64],
    window_size: int,
    horizon_size: int,
    input_size: int,
    output_size: int,
) -> ModelInput:
    expected_size = window_size * (input_size + output_size) + horizon_size * input_size
    assert x.shape == (expected_size,), (
        f'Expects ndarray of shape ({expected_size},).'
        f'Received ndarray of shape {x.shape}.'
    )

    initial_control_offset = window_size * output_size
    control_offset = initial_control_offset + window_size * input_size
    return ModelInput(
        initial_state=x[:initial_control_offset].reshape(window_size, output_size),
        initial_control=x[initial_control_offset:control_offset].reshape(
            window_size, input_size
        ),
        control=x[control_offset:].reshape(horizon_size, input_size),
        x0=None,
    )
