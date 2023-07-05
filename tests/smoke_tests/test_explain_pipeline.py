import pathlib

from deepsysid.explainability.explainers.blackbox import (
    KernelSHAPExplainer,
    LIMEExplainer,
)
from deepsysid.explainability.explainers.switching import SwitchingLSTMExplainer
from deepsysid.explainability.metrics import (
    ExplanationComplexityMetric,
    LipschitzEstimateMetric,
    NMSEInfidelityMetric,
)
from deepsysid.metrics import NormalizedRootMeanSquaredErrorMetric
from deepsysid.models.linear import LinearLag
from deepsysid.models.switching.switchrnn import StableSwitchingLSTMModel
from deepsysid.pipeline.configuration import (
    ExperimentConfiguration,
    ExperimentExplainerConfiguration,
    ExperimentExplanationMetricConfiguration,
    ExperimentMetricConfiguration,
    ExperimentModelConfiguration,
    SessionConfiguration,
)

from . import pipeline
from .pipeline import (
    get_4dof_ship_control_names,
    get_4dof_ship_data,
    get_4dof_ship_state_names,
    get_horizon_size,
    get_time_delta,
    get_window_size,
    run_generic_pipeline,
)


def test_kernel_shap_explainer(tmp_path: pathlib.Path) -> None:
    control_names = get_4dof_ship_control_names()
    state_names = get_4dof_ship_state_names()

    configuration = ExperimentConfiguration(
        time_delta=get_time_delta(),
        window_size=get_window_size(),
        horizon_size=get_horizon_size(),
        control_names=control_names,
        state_names=state_names,
        initial_state_names=state_names,
        session=SessionConfiguration(total_runs_for_best_models=3),
        tracker=dict(),
        additional_tests=dict(),
        models={
            'LinearLag': ExperimentModelConfiguration(
                model_class='deepsysid.models.linear.LinearLag',
                parameters=LinearLag.CONFIG(
                    control_names=pipeline.get_4dof_ship_control_names(),
                    state_names=pipeline.get_4dof_ship_state_names(),
                    device_name=pipeline.get_cpu_device_name(),
                    time_delta=pipeline.get_time_delta(),
                    lag=pipeline.get_window_size(),
                ),
            )
        },
        metrics=dict(
            nrmse=ExperimentMetricConfiguration(
                metric_class='deepsysid.pipeline.metrics.'
                'NormalizedRootMeanSquaredErrorMetric',
                parameters=NormalizedRootMeanSquaredErrorMetric.CONFIG(
                    state_names=state_names, sample_time=get_time_delta()
                ),
            ),
        ),
        target_metric='nrmse',
        explanation_metrics=dict(
            infidelity=ExperimentExplanationMetricConfiguration(
                metric_class='deepsysid.explainability.metrics.NMSEInfidelityMetric',
                parameters=NMSEInfidelityMetric.CONFIG(state_names=state_names),
            )
        ),
        explainers=dict(
            shap_explainer=ExperimentExplainerConfiguration(
                explainer_class='deepsysid.explainability'
                '.explainers.blackbox.KernelSHAPExplainer',
                parameters=KernelSHAPExplainer.CONFIG(),
            )
        ),
    )

    run_generic_pipeline(
        base_path=tmp_path,
        model_name='LinearLag',
        config=configuration,
        get_data_func=get_4dof_ship_data,
    )


def test_lime_explainer(tmp_path: pathlib.Path) -> None:
    control_names = get_4dof_ship_control_names()
    state_names = get_4dof_ship_state_names()

    configuration = ExperimentConfiguration(
        time_delta=get_time_delta(),
        window_size=get_window_size(),
        horizon_size=get_horizon_size(),
        control_names=control_names,
        state_names=state_names,
        initial_state_names=state_names,
        session=SessionConfiguration(total_runs_for_best_models=3),
        tracker=dict(),
        additional_tests=dict(),
        models={
            'LinearLag': ExperimentModelConfiguration(
                model_class='deepsysid.models.linear.LinearLag',
                parameters=LinearLag.CONFIG(
                    control_names=pipeline.get_4dof_ship_control_names(),
                    state_names=pipeline.get_4dof_ship_state_names(),
                    device_name=pipeline.get_cpu_device_name(),
                    time_delta=pipeline.get_time_delta(),
                    lag=pipeline.get_window_size(),
                ),
            )
        },
        metrics=dict(
            nrmse=ExperimentMetricConfiguration(
                metric_class='deepsysid.pipeline.metrics.'
                'NormalizedRootMeanSquaredErrorMetric',
                parameters=NormalizedRootMeanSquaredErrorMetric.CONFIG(
                    state_names=state_names, sample_time=get_time_delta()
                ),
            ),
        ),
        target_metric='nrmse',
        explanation_metrics=dict(
            infidelity=ExperimentExplanationMetricConfiguration(
                metric_class='deepsysid.explainability.metrics.NMSEInfidelityMetric',
                parameters=NMSEInfidelityMetric.CONFIG(state_names=state_names),
            ),
            robustness=ExperimentExplanationMetricConfiguration(
                metric_class='deepsysid.explainability.metrics.LipschitzEstimateMetric',
                parameters=LipschitzEstimateMetric.CONFIG(
                    state_names=state_names,
                    n_disturbances=5,
                    control_error_std=[0.1 for _ in control_names],
                    state_error_std=[0.1 for _ in state_names],
                ),
            ),
            simplicity=ExperimentExplanationMetricConfiguration(
                metric_class='deepsysid.explainability.metrics'
                '.ExplanationComplexityMetric',
                parameters=ExplanationComplexityMetric.CONFIG(
                    state_names=state_names, relevance_threshold=0.1
                ),
            ),
        ),
        explainers=dict(
            lime_explainer=ExperimentExplainerConfiguration(
                explainer_class='deepsysid.explainability'
                '.explainers.blackbox.LIMEExplainer',
                explained_super_classes=[
                    'deepsysid.models.base.DynamicIdentificationModel',
                ],
                parameters=LIMEExplainer.CONFIG(num_samples=6),
            )
        ),
    )

    run_generic_pipeline(
        base_path=tmp_path,
        model_name='LinearLag',
        config=configuration,
        get_data_func=get_4dof_ship_data,
    )


def test_switching_lstm_explainer(tmp_path: pathlib.Path) -> None:
    control_names = get_4dof_ship_control_names()
    state_names = get_4dof_ship_state_names()

    configuration = ExperimentConfiguration(
        time_delta=get_time_delta(),
        window_size=get_window_size(),
        horizon_size=get_horizon_size(),
        control_names=control_names,
        state_names=state_names,
        initial_state_names=state_names,
        session=SessionConfiguration(total_runs_for_best_models=3),
        tracker=dict(),
        additional_tests=dict(),
        models={
            'StableReLiNet': ExperimentModelConfiguration(
                model_class='deepsysid.models.switching.switchrnn'
                '.StableSwitchingLSTMModel',
                parameters=StableSwitchingLSTMModel.CONFIG(
                    control_names=pipeline.get_4dof_ship_control_names(),
                    state_names=pipeline.get_4dof_ship_state_names(),
                    device_name=pipeline.get_cpu_device_name(),
                    time_delta=pipeline.get_time_delta(),
                    switched_system_state_dim=15,
                    recurrent_dim=10,
                    num_recurrent_layers=2,
                    dropout=0.25,
                    sequence_length=3,
                    learning_rate=0.1,
                    batch_size=2,
                    epochs_initializer=2,
                    epochs_predictor=2,
                    loss='mse',
                ),
            )
        },
        metrics=dict(
            nrmse=ExperimentMetricConfiguration(
                metric_class='deepsysid.pipeline.metrics.'
                'NormalizedRootMeanSquaredErrorMetric',
                parameters=NormalizedRootMeanSquaredErrorMetric.CONFIG(
                    state_names=state_names, sample_time=get_time_delta()
                ),
            ),
        ),
        target_metric='nrmse',
        explanation_metrics=dict(
            infidelity=ExperimentExplanationMetricConfiguration(
                metric_class='deepsysid.explainability.metrics.NMSEInfidelityMetric',
                parameters=NMSEInfidelityMetric.CONFIG(state_names=state_names),
            )
        ),
        explainers=dict(
            switching_lstm_explainer=ExperimentExplainerConfiguration(
                explainer_class='deepsysid.explainability'
                '.explainers.switching.SwitchingLSTMExplainer',
                parameters=SwitchingLSTMExplainer.CONFIG(),
            )
        ),
    )

    run_generic_pipeline(
        base_path=tmp_path,
        model_name='StableReLiNet',
        config=configuration,
        get_data_func=get_4dof_ship_data,
    )
