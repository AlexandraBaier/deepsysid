import pathlib

from deepsysid.models.linear import LinearModel, LinearLag, QuadraticControlLag
from deepsysid.models.narx import NARXDenseNetwork
from deepsysid.models.recurrent import LSTMInitModel, ConstrainedRnn
from . import pipeline


def test_linear_model_cpu(tmp_path: pathlib.Path):
    model_name = 'LinearModel'
    model_class = 'deepsysid.models.linear.LinearModel'
    config = LinearModel.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta()
    )
    pipeline.run_pipeline(
        tmp_path, model_name, model_class, config=config
    )


def test_linear_lag(tmp_path: pathlib.Path):
    model_name = 'LinearLag'
    model_class = 'deepsysid.models.linear.LinearLag'
    config = LinearLag.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        window_size=pipeline.get_window_size()
    )
    pipeline.run_pipeline(
        tmp_path, model_name, model_class, config=config
    )


def test_quadratic_control_lag(tmp_path: pathlib.Path):
    model_name = 'QuadraticControlLag'
    model_class = 'deepsysid.models.linear.QuadraticControlLag'
    config = QuadraticControlLag.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        window_size=pipeline.get_window_size()
    )
    pipeline.run_pipeline(
        tmp_path, model_name, model_class, config=config
    )


def test_narx(tmp_path: pathlib.Path):
    model_name = 'NARXDenseNetwork'
    model_class = 'deepsysid.models.narx.NARXDenseNetwork'
    config = NARXDenseNetwork.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        window_size=pipeline.get_window_size(),
        learning_rate=0.01,
        batch_size=3,
        epochs=2,
        layers=[5, 10, 5],
        dropout=0.25
    )
    pipeline.run_pipeline(
        tmp_path, model_name, model_class, config=config
    )


def test_lstm_init_model(tmp_path: pathlib.Path):
    model_name = 'LSTMInitModel'
    model_class = 'deepsysid.models.recurrent.LSTMInitModel'
    config = LSTMInitModel.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=10,
        num_recurrent_layers=2,
        dropout=0.25,
        sequence_length=3,
        learning_rate=0.1,
        batch_size=3,
        epochs_initializer=2,
        epochs_predictor=2,
        loss='mse'
    )
    pipeline.run_pipeline(
        tmp_path, model_name, model_class, config=config
    )


def test_constrained_rnn(tmp_path: pathlib.Path):
    model_name = 'ConstrainedRnn'
    model_class = 'deepsysid.models.recurrent.ConstrainedRnn'
    config = ConstrainedRnn.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        # TODO: nx and nw need to be the same value?
        #  Why are the separate?
        nx=2,
        nw=2,
        gamma=0.1,
        beta=0.05,
        decay_parameter=0.01,
        num_recurrent_layers=3,
        dropout=0.25,
        sequence_length=3,
        learning_rate=0.1,
        batch_size=3,
        epochs_initializer=2,
        epochs_predictor=2,
        loss='mse'
    )
    pipeline.run_pipeline(
        tmp_path, model_name, model_class, config=config
    )
