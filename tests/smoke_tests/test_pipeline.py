import pathlib

from deepsysid.models.hybrid.bounded_residual import (
    HybridBlankeModel,
    HybridLinearModel,
    HybridMinimalManeuveringModel,
    HybridPropulsionManeuveringModel,
)
from deepsysid.models.linear import LinearLag, LinearModel, QuadraticControlLag
from deepsysid.models.narx import NARXDenseNetwork
from deepsysid.models.recurrent import (
    ConstrainedRnn,
    LSTMCombinedInitModel,
    LSTMInitModel,
)
from deepsysid.models.switching.switchrnn import StableSwitchingLSTMModel

from . import pipeline


def test_linear_model_cpu(tmp_path: pathlib.Path):
    model_name = 'LinearModel'
    model_class = 'deepsysid.models.linear.LinearModel'
    config = LinearModel.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)


def test_linear_lag(tmp_path: pathlib.Path):
    model_name = 'LinearLag'
    model_class = 'deepsysid.models.linear.LinearLag'
    config = LinearLag.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        window_size=pipeline.get_window_size(),
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)


def test_quadratic_control_lag(tmp_path: pathlib.Path):
    model_name = 'QuadraticControlLag'
    model_class = 'deepsysid.models.linear.QuadraticControlLag'
    config = QuadraticControlLag.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        window_size=pipeline.get_window_size(),
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)


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
        batch_size=2,
        epochs=2,
        layers=[5, 10, 5],
        dropout=0.25,
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)


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
        batch_size=2,
        epochs_initializer=2,
        epochs_predictor=2,
        loss='mse',
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)


def test_constrained_rnn(tmp_path: pathlib.Path):
    model_name = 'ConstrainedRnn'
    model_class = 'deepsysid.models.recurrent.ConstrainedRnn'
    config = ConstrainedRnn.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        nx=3,
        recurrent_dim=2,
        gamma=0.1,
        beta=0.05,
        initial_decay_parameter=1e-3,
        decay_rate=10,
        epochs_with_const_decay=1,
        num_recurrent_layers_init=3,
        dropout=0.25,
        sequence_length=3,
        learning_rate=0.1,
        batch_size=2,
        epochs_initializer=2,
        epochs_predictor=2,
        loss='mse',
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)


def test_constrained_rnn_stable(tmp_path: pathlib.Path):
    model_name = 'ConstrainedRnn'
    model_class = 'deepsysid.models.recurrent.ConstrainedRnn'
    config = ConstrainedRnn.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        nx=2,
        recurrent_dim=3,
        gamma=0.0,
        beta=0.05,
        initial_decay_parameter=1e-3,
        decay_rate=10,
        epochs_with_const_decay=1,
        num_recurrent_layers_init=3,
        dropout=0.25,
        sequence_length=3,
        learning_rate=0.1,
        batch_size=2,
        epochs_initializer=2,
        epochs_predictor=2,
        loss='mse',
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)


def test_hybrid_minimal_maneuvering_model(tmp_path: pathlib.Path):
    model_name = 'HybridMinimalManeuveringModel'
    model_class = (
        'deepsysid.models.hybrid.bounded_residual.HybridMinimalManeuveringModel'
    )
    config = HybridMinimalManeuveringModel.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=10,
        num_recurrent_layers=2,
        dropout=0.25,
        sequence_length=3,
        learning_rate=0.01,
        batch_size=2,
        epochs_initializer=2,
        epochs_parallel=3,
        epochs_feedback=3,
        loss='mse',
        Xud=-0.1,
        Yvd=0.2,
        Ypd=-0.1,
        Yrd=0.1,
        Kvd=0.1,
        Kpd=0.3,
        Krd=-0.4,
        Nvd=0.1,
        Npd=0.2,
        Nrd=0.1,
        m=10.0,
        Ixx=0.5,
        Izz=0.1,
        xg=0.3,
        zg=0.5,
        rho_water=0.5,
        disp=20.0,
        gm=4,
        g=9.81,
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)


def test_hybrid_propulsion_maneuvering_model(tmp_path: pathlib.Path):
    model_name = 'HybridPropulsionManeuveringModel'
    model_class = (
        'deepsysid.models.hybrid.bounded_residual.HybridPropulsionManeuveringModel'
    )
    config = HybridPropulsionManeuveringModel.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=10,
        num_recurrent_layers=2,
        dropout=0.25,
        sequence_length=3,
        learning_rate=0.01,
        batch_size=2,
        epochs_initializer=2,
        epochs_parallel=3,
        epochs_feedback=3,
        loss='mse',
        Xud=-0.1,
        Yvd=0.2,
        Ypd=-0.1,
        Yrd=0.1,
        Kvd=0.1,
        Kpd=0.3,
        Krd=-0.4,
        Nvd=0.1,
        Npd=0.2,
        Nrd=0.1,
        m=10.0,
        Ixx=0.5,
        Izz=0.1,
        xg=0.3,
        zg=0.5,
        rho_water=0.5,
        disp=20.0,
        gm=4,
        g=9.81,
        wake_factor=0.2,
        diameter=2.0,
        Kt=(2.0, -1.0, 0.5),
        lx=10,
        ly=4,
        lz=3,
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)


def test_hybrid_linear_model(tmp_path: pathlib.Path):
    model_name = 'HybridLinearModel'
    model_class = 'deepsysid.models.hybrid.bounded_residual.HybridLinearModel'
    config = HybridLinearModel.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=10,
        num_recurrent_layers=2,
        dropout=0.25,
        sequence_length=3,
        learning_rate=0.01,
        batch_size=2,
        epochs_initializer=2,
        epochs_parallel=3,
        epochs_feedback=3,
        loss='mse',
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)


def test_hybrid_blanke_model(tmp_path: pathlib.Path):
    model_name = 'HybridBlankeModel'
    model_class = 'deepsysid.models.hybrid.bounded_residual.HybridBlankeModel'
    config = HybridBlankeModel.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=10,
        num_recurrent_layers=2,
        dropout=0.25,
        sequence_length=3,
        learning_rate=0.01,
        batch_size=2,
        epochs_initializer=2,
        epochs_parallel=3,
        epochs_feedback=3,
        loss='mse',
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)


def test_lstm_combined_init_model(tmp_path: pathlib.Path):
    model_name = 'LSTMCombinedInitModel'
    model_class = 'deepsysid.models.recurrent.LSTMCombinedInitModel'
    config = LSTMCombinedInitModel.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=10,
        num_recurrent_layers=2,
        dropout=0.25,
        sequence_length=3,
        learning_rate=0.1,
        batch_size=2,
        epochs=2,
        loss='msge',
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)


def test_stable_switching_lstm_model(tmp_path: pathlib.Path):
    model_name = 'StableSwitchingLSTMModel'
    model_class = 'deepsysid.models.switching.switchrnn.StableSwitchingLSTMModel'
    config = StableSwitchingLSTMModel.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=10,
        num_recurrent_layers=2,
        dropout=0.25,
        sequence_length=3,
        learning_rate=0.1,
        batch_size=2,
        epochs_initializer=2,
        epochs_predictor=2,
        loss='mse',
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)


def test_unconstrained_switching_lstm_model(tmp_path: pathlib.Path):
    model_name = 'UnconstrainedSwitchingLSTMModel'
    model_class = 'deepsysid.models.switching.switchrnn.UnconstrainedSwitchingLSTMModel'
    config = StableSwitchingLSTMModel.CONFIG(
        control_names=pipeline.get_control_names(),
        state_names=pipeline.get_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=10,
        num_recurrent_layers=2,
        dropout=0.25,
        sequence_length=3,
        learning_rate=0.1,
        batch_size=2,
        epochs_initializer=2,
        epochs_predictor=2,
        loss='mse',
    )
    pipeline.run_pipeline(tmp_path, model_name, model_class, model_config=config)
