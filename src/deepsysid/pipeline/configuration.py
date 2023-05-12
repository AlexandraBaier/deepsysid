import itertools
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, root_validator

from ..explainability.base import (
    BaseExplainerConfig,
    BaseExplanationMetricConfig,
    retrieve_explainer_class,
    retrieve_explanation_metric_class,
)
from ..models.base import DynamicIdentificationModel, DynamicIdentificationModelConfig
from .metrics import BaseMetricConfig, retrieve_metric_class
from .testing.base import BaseTestConfig, retrieve_test_class


class ExperimentModelConfiguration(BaseModel):
    model_class: str
    parameters: DynamicIdentificationModelConfig


class ExperimentMetricConfiguration(BaseModel):
    metric_class: str
    parameters: BaseMetricConfig


class ExperimentTestConfiguration(BaseModel):
    test_class: str
    parameters: BaseTestConfig


class ExperimentExplainerConfiguration(BaseModel):
    explainer_class: str
    explained_super_classes: Optional[List[str]]
    parameters: BaseExplainerConfig


class ExperimentExplanationMetricConfiguration(BaseModel):
    metric_class: str
    parameters: BaseExplanationMetricConfig


class GridSearchMetricConfiguration(BaseModel):
    metric_class: str
    parameters: Dict[str, Any]


class GridSearchExplanationMetricConfiguration(BaseModel):
    metric_class: str
    parameters: Dict[str, Any]


class GridSearchTestConfiguration(BaseModel):
    test_class: str
    parameters: Dict[str, Any]


class SessionConfiguration(BaseModel):
    total_runs_for_best_models: int
    training_trajectory: Optional[List[str]]
    training_scalar: Optional[List[str]]


class ExperimentGridSearchSettings(BaseModel):
    time_delta: float
    window_size: int
    horizon_size: int
    control_names: List[str]
    state_names: List[str]
    initial_state_names: Optional[List[str]] = None
    additional_tests: Dict[str, GridSearchTestConfiguration]
    target_metric: str
    metrics: Dict[str, GridSearchMetricConfiguration]
    explanation_metrics: Optional[Dict[str, GridSearchExplanationMetricConfiguration]]
    session: Optional[SessionConfiguration]


class ModelGridSearchTemplate(BaseModel):
    model_base_name: str
    model_class: str
    static_parameters: Dict[str, Any]
    flexible_parameters: Dict[str, List[Any]]


class GridSearchExplainerConfiguration(BaseModel):
    explainer_class: str
    explained_super_classes: Optional[List[str]]
    parameters: Dict[str, Any]


class ExperimentGridSearchTemplate(BaseModel):
    settings: ExperimentGridSearchSettings
    models: List[ModelGridSearchTemplate]
    explainers: Optional[Dict[str, GridSearchExplainerConfiguration]]


class ExperimentConfiguration(BaseModel):
    time_delta: float
    window_size: int
    horizon_size: int
    control_names: List[str]
    state_names: List[str]
    initial_state_names: List[str]
    metrics: Dict[str, ExperimentMetricConfiguration]
    explanation_metrics: Optional[Dict[str, ExperimentExplanationMetricConfiguration]]
    target_metric: str
    additional_tests: Dict[str, ExperimentTestConfiguration]
    models: Dict[str, ExperimentModelConfiguration]
    explainers: Optional[Dict[str, ExperimentExplainerConfiguration]]
    session: Optional[SessionConfiguration]

    @root_validator
    def check_target_metric_in_metrics(cls, values):
        target_metric = values.get('target_metric')
        metrics = values.get('metrics')
        if target_metric not in metrics:
            raise ValueError('target_metric should be a metric in metrics.')
        return values

    @classmethod
    def from_grid_search_template(
        cls, template: ExperimentGridSearchTemplate, device_name: str = 'cpu'
    ) -> 'ExperimentConfiguration':
        if template.settings.initial_state_names is None:
            initial_state_names = template.settings.state_names
        else:
            initial_state_names = template.settings.initial_state_names

        models: Dict[str, ExperimentModelConfiguration] = dict()
        for model_template in template.models:
            model_class_str = model_template.model_class
            model_class = retrieve_model_class(model_class_str)

            base_model_params = dict(
                device_name=device_name,
                control_names=template.settings.control_names,
                state_names=template.settings.state_names,
                initial_state_names=initial_state_names,
                time_delta=template.settings.time_delta,
                window_size=template.settings.window_size,
                horizon_size=template.settings.horizon_size,
            )
            static_model_params = model_template.static_parameters

            for combination in itertools.product(
                *list(model_template.flexible_parameters.values())
            ):
                model_name = model_template.model_base_name
                flexible_model_params = dict()
                for param_name, param_value in zip(
                    model_template.flexible_parameters.keys(), combination
                ):
                    flexible_model_params[param_name] = param_value
                    if issubclass(type(param_value), list):
                        model_name += '-' + '_'.join(map(str, param_value))
                    else:
                        if isinstance(param_value, float):
                            model_name += f'-{param_value:f}'.replace('.', '')
                        else:
                            model_name += f'-{param_value}'.replace('.', '')

                model_config = ExperimentModelConfiguration(
                    model_class=model_class_str,
                    parameters=model_class.CONFIG.parse_obj(
                        # Merge dictionaries
                        {
                            **base_model_params,
                            **static_model_params,
                            **flexible_model_params,
                        }
                    ),
                )
                models[model_name] = model_config

        metrics = dict()
        base_metric_params = dict(
            state_names=template.settings.state_names,
            sample_time=template.settings.time_delta,
        )
        for name, metric in template.settings.metrics.items():
            metric_class = retrieve_metric_class(metric.metric_class)
            metrics[name] = ExperimentMetricConfiguration(
                metric_class=metric.metric_class,
                parameters=metric_class.CONFIG.parse_obj(
                    {**metric.parameters, **base_metric_params}
                ),
            )

        tests = dict()
        base_test_params = dict(
            control_names=template.settings.control_names,
            state_names=template.settings.state_names,
            window_size=template.settings.window_size,
            horizon_size=template.settings.horizon_size,
        )
        for name, test in template.settings.additional_tests.items():
            test_class = retrieve_test_class(test.test_class)
            tests[name] = ExperimentTestConfiguration(
                test_class=test.test_class,
                parameters=test_class.CONFIG.parse_obj(
                    {**test.parameters, **base_test_params}
                ),
            )

        explainers = dict()
        if template.explainers is not None:
            for name, explainer in template.explainers.items():
                explainer_cls = retrieve_explainer_class(explainer.explainer_class)

                # Check whether all stated classes can be imported.
                if explainer.explained_super_classes is not None:
                    for explained_model_class in explainer.explained_super_classes:
                        _ = retrieve_model_class(explained_model_class)

                explainers[name] = ExperimentExplainerConfiguration(
                    explainer_class=explainer.explainer_class,
                    explained_super_classes=explainer.explained_super_classes,
                    parameters=explainer_cls.CONFIG.parse_obj(explainer.parameters),
                )

        explanation_metrics = dict()
        base_explanation_metric_params = dict(state_names=template.settings.state_names)
        if template.settings.explanation_metrics is not None:
            for (
                name,
                explanation_metric,
            ) in template.settings.explanation_metrics.items():
                explanation_metric_cls = retrieve_explanation_metric_class(
                    explanation_metric.metric_class
                )
                explanation_metrics[name] = ExperimentExplanationMetricConfiguration(
                    metric_class=explanation_metric.metric_class,
                    parameters=explanation_metric_cls.CONFIG.parse_obj(
                        {
                            **base_explanation_metric_params,
                            **explanation_metric.parameters,
                        }
                    ),
                )

        return cls(
            time_delta=template.settings.time_delta,
            window_size=template.settings.window_size,
            horizon_size=template.settings.horizon_size,
            control_names=template.settings.control_names,
            state_names=template.settings.state_names,
            initial_state_names=initial_state_names,
            models=models,
            target_metric=template.settings.target_metric,
            metrics=metrics,
            explanation_metrics=explanation_metrics,
            additional_tests=tests,
            explainers=explainers,
            session=template.settings.session,
        )


def initialize_model(
    experiment_config: ExperimentConfiguration, model_name: str, device_name: str
) -> DynamicIdentificationModel:
    model_config = experiment_config.models[model_name]

    model_class = retrieve_model_class(model_config.model_class)

    parameters = model_config.parameters
    parameters.device_name = device_name

    model = model_class(config=parameters)

    return model


def retrieve_model_class(model_class_string: str) -> Type[DynamicIdentificationModel]:
    # https://stackoverflow.com/a/452981
    parts = model_class_string.split('.')
    module_string = '.'.join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, DynamicIdentificationModel):
        raise ValueError(f'{cls} is not a subclass of DynamicIdentificationModel')
    return cls  # type: ignore
