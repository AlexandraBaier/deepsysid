import pathlib

from deepsysid.pipeline.configuration import (
    ExperimentConfiguration,
    ExperimentGridSearchSettings,
    ExperimentGridSearchTemplate,
    GridSearchMetricConfiguration,
    GridSearchTrackingConfiguration,
    GridSearchTestConfiguration,
    ModelGridSearchTemplate,
)
from deepsysid.pipeline.gridsearch import ExperimentSessionManager, SessionAction

from .pipeline import (
    get_4dof_ship_control_names,
    get_4dof_ship_data,
    get_4dof_ship_state_names,
    get_cpu_device_name,
    get_horizon_size,
    get_time_delta,
    get_window_size,
    prepare_directories,
)


def test_experiment_session_manager_new_and_test_best_successful(
    tmp_path: pathlib.Path,
) -> None:
    paths = prepare_directories(tmp_path)

    # Setup dataset directory.
    paths['train'].joinpath('train-0.csv').write_text(data=get_4dof_ship_data(0))
    paths['validation'].joinpath('validation-0.csv').write_text(
        data=get_4dof_ship_data(1)
    )
    paths['test'].joinpath('test-0.csv').write_text(data=get_4dof_ship_data(2))

    template = ExperimentGridSearchTemplate(
        settings=ExperimentGridSearchSettings(
            time_delta=get_time_delta(),
            window_size=get_window_size(),
            horizon_size=get_horizon_size(),
            control_names=get_4dof_ship_control_names(),
            state_names=get_4dof_ship_state_names(),
            initial_state_names=get_4dof_ship_state_names(),
            target_metric='d1',
            metrics=dict(
                d1=GridSearchMetricConfiguration(
                    metric_class='deepsysid.pipeline.metrics.IndexOfAgreementMetric',
                    parameters=dict(j=1),
                )
            ),
            tracker=dict(
                mlflow=GridSearchTrackingConfiguration(
                    tracking_class='deepsysid.tracker.mlflow.MlFlowTracker',
                    parameters=dict(),
                )
            ),
            additional_tests=dict(
                Bibo_stability=GridSearchTestConfiguration(
                    test_class='deepsysid.pipeline.testing.stability.'
                    'bibo.BiboStabilityTest',
                    parameters=dict(
                        optimization_steps=10,
                        optimization_lr=1e-3,
                        initial_mean_delta=0.0,
                        initial_std_delta=1e-3,
                        evaluation_sequence=1,
                        clip_gradient_norm=100.0,
                        regularization_scale=0.25,
                    ),
                ),
                incremental_stability=GridSearchTestConfiguration(
                    test_class='deepsysid.pipeline.testing.stability.'
                    'incremental.IncrementalStabilityTest',
                    parameters=dict(
                        optimization_steps=10,
                        optimization_lr=1e-3,
                        initial_mean_delta=0.0,
                        initial_std_delta=1e-3,
                        evaluation_sequence=1,
                        clip_gradient_norm=100.0,
                        regularization_scale=0.25,
                    ),
                ),
            ),
        ),
        models=[
            ModelGridSearchTemplate(
                model_base_name='Lag',
                model_class='deepsysid.models.linear.LinearLag',
                static_parameters=dict(),
                flexible_parameters=dict(lag=list(range(1, get_window_size() + 1))),
            ),
            ModelGridSearchTemplate(
                # Naming it LagQuadratic checks whether the base name extraction
                # works properly.
                # TODO: The functionality should be unit-tested.
                model_base_name='LagQuadratic',
                model_class='deepsysid.models.linear.QuadraticControlLag',
                static_parameters=dict(),
                flexible_parameters=dict(lag=list(range(1, get_window_size() + 1))),
            ),
            ModelGridSearchTemplate(
                model_base_name='LSTM',
                model_class='deepsysid.models.recurrent.LSTMInitModel',
                static_parameters=dict(
                    dropout=0.25,
                    sequence_length=3,
                    learning_rate=0.0001,
                    batch_size=2,
                    epochs_initializer=2,
                    epochs_predictor=2,
                ),
                flexible_parameters=dict(
                    recurrent_dim=[16, 32, 64],
                    num_recurrent_layers=[1, 2],
                    loss=['mse', 'msge'],
                ),
            ),
        ],
    )

    n_models = len(
        ExperimentConfiguration.from_grid_search_template(
            template, device_name=get_cpu_device_name()
        ).models
    )

    manager = ExperimentSessionManager(
        config=template,
        device_name=get_cpu_device_name(),
        session_action=SessionAction.NEW,
        dataset_directory=str(paths['data']),
        models_directory=str(paths['models']),
        results_directory=str(paths['result']),
    )
    manager.run_session()
    session_report = manager.get_session_report()
    assert len(session_report.unfinished_models) == 0, 'No models left to validate.'
    assert (
        len(session_report.validated_models) == n_models
    ), 'All generated models should be validated.'

    manager = ExperimentSessionManager(
        config=template,
        device_name=get_cpu_device_name(),
        session_action=SessionAction.TEST_BEST,
        dataset_directory=str(paths['data']),
        models_directory=str(paths['models']),
        results_directory=str(paths['result']),
        session_report=session_report,
    )
    manager.run_session()
    session_report = manager.get_session_report()

    assert (
        len(session_report.tested_models) == 3
    ), 'Only 3 models should be tested (one per class/base name).'
    assert (
        len(session_report.best_per_base_name.items()) == 3
    ), 'There are 3 unique base names.'
    assert (
        len(session_report.best_per_class.items()) == 3
    ), 'There are 3 unique classes.'
