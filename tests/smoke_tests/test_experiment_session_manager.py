import pathlib

from deepsysid.pipeline.configuration import (
    ExperimentGridSearchSettings,
    ExperimentGridSearchTemplate,
    ModelGridSearchTemplate,
)
from deepsysid.pipeline.gridsearch import ExperimentSessionManager, SessionAction

from .pipeline import (
    get_control_names,
    get_cpu_device_name,
    get_data,
    get_horizon_size,
    get_state_names,
    get_thresholds,
    get_time_delta,
    get_train_fraction,
    get_validation_fraction,
    get_window_size,
    prepare_directories,
)


def test_experiment_session_manager_session_action_new_successful(
    tmp_path: pathlib.Path,
) -> None:
    paths = prepare_directories(tmp_path)

    # Setup dataset directory.
    paths['train'].joinpath('train-0.csv').write_text(data=get_data(0))
    paths['validation'].joinpath('validation-0.csv').write_text(data=get_data(1))
    paths['test'].joinpath('test-0.csv').write_text(data=get_data(2))

    template = ExperimentGridSearchTemplate(
        settings=ExperimentGridSearchSettings(
            train_fraction=get_train_fraction(),
            validation_fraction=get_validation_fraction(),
            time_delta=get_time_delta(),
            window_size=get_window_size(),
            horizon_size=get_horizon_size(),
            control_names=get_control_names(),
            state_names=get_state_names(),
            thresholds=get_thresholds(),
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

    manager = ExperimentSessionManager(
        config=template,
        device_name=get_cpu_device_name(),
        session_action=SessionAction.NEW,
        dataset_directory=str(paths['data']),
        models_directory=str(paths['models']),
        results_directory=str(paths['result']),
        target_metric='rmse',
    )
    manager.run_session()
