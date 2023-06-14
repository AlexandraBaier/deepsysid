import pathlib

from deepsysid.models.hybrid.bounded_residual import (
    HybridBlankeModel,
    HybridLinearModel,
    HybridMinimalManeuveringModel,
    HybridPropulsionManeuveringModel,
)
from deepsysid.models.hybrid.serial import (
    SerialParallel4DOFShipModel,
    SerialParallelQuadcopterModel,
)
from deepsysid.models.linear import (
    LinearLag,
    LinearModel,
    QuadraticControlLag,
    RidgeRegressionCVModel,
)
from deepsysid.models.narx import NARXDenseNetwork
from deepsysid.models.recurrent import (
    ConstrainedRnn,
    GRUInitModel,
    HybridConstrainedRnn,
    JointInitializerGRUModel,
    JointInitializerLSTMModel,
    JointInitializerRNNModel,
    LSTMInitModel,
    LtiRnnInit,
    RnnInit,
    RnnInitFlexibleNonlinearity,
    WashoutInitializerGRUModel,
    WashoutInitializerLSTMModel,
    WashoutInitializerRNNModel,
)
from deepsysid.models.switching.klinreg import KLinearRegressionARXModel
from deepsysid.models.switching.switchrnn import (
    StableIdentityOutputSwitchingLSTMModel,
    StableSwitchingLSTMModel,
    UnconstrainedIdentityOutputSwitchingLSTMModel,
    UnconstrainedSwitchingLSTMModel,
)

from ..unit_tests import test_networks
from . import pipeline


def test_linear_model_cpu(tmp_path: pathlib.Path) -> None:
    model_name = 'LinearModel'
    model_class = 'deepsysid.models.linear.LinearModel'
    config = LinearModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_linear_lag(tmp_path: pathlib.Path) -> None:
    model_name = 'LinearLag'
    model_class = 'deepsysid.models.linear.LinearLag'
    config = LinearLag.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        lag=pipeline.get_window_size(),
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_quadratic_control_lag(tmp_path: pathlib.Path) -> None:
    model_name = 'QuadraticControlLag'
    model_class = 'deepsysid.models.linear.QuadraticControlLag'
    config = QuadraticControlLag.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        lag=pipeline.get_window_size(),
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_ridge_regression_cv_model(tmp_path: pathlib.Path) -> None:
    model_name = 'RidgeCV'
    model_class = 'deepsysid.models.linear.RidgeRegressionCVModel'
    config = RidgeRegressionCVModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        window_size=pipeline.get_window_size(),
        folds=3,
        repeats=2,
        c_grid=[0.1, 1.0],
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_narx(tmp_path: pathlib.Path) -> None:
    model_name = 'NARXDenseNetwork'
    model_class = 'deepsysid.models.narx.NARXDenseNetwork'
    config = NARXDenseNetwork.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        window_size=pipeline.get_window_size(),
        learning_rate=0.01,
        batch_size=2,
        epochs=2,
        layers=[5, 10, 5],
        dropout=0.25,
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_lstm_init_model(tmp_path: pathlib.Path) -> None:
    model_name = 'LSTMInitModel'
    model_class = 'deepsysid.models.recurrent.LSTMInitModel'
    config = LSTMInitModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
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
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_constrained_rnn(tmp_path: pathlib.Path) -> None:
    model_name = 'ConstrainedRnn'
    model_class = 'deepsysid.models.recurrent.ConstrainedRnn'
    config = ConstrainedRnn.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
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
        bias=True,
        nonlinearity='torch.nn.Tanh()',
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_constrained_hybrid_rnn(tmp_path: pathlib.Path) -> None:
    model_name = 'HybridConstrainedRnn'
    model_class = 'deepsysid.models.recurrent.HybridConstrainedRnn'
    A_lin, B_lin, C_lin, D_lin = test_networks.get_linear_matrices()
    config = HybridConstrainedRnn.CONFIG(
        control_names=pipeline.get_cartpole_control_names(),
        state_names=pipeline.get_cartpole_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        A_lin=A_lin.tolist(),
        B_lin=B_lin.tolist(),
        C_lin=C_lin.tolist(),
        D_lin=D_lin.tolist(),
        nwu=10,
        nzu=10,
        alpha=0.0,
        beta=1.0,
        gamma=1.0,
        initial_decay_parameter=1e-3,
        decay_rate=10,
        epochs_with_const_decay=1,
        num_recurrent_layers_init=3,
        dropout=0.25,
        sequence_length=[3],
        learning_rate=0.1,
        batch_size=2,
        epochs_initializer=2,
        epochs_predictor=2,
        loss='mse',
        enforce_constraints_method='barrier',
        epochs_without_projection=50,
    )
    pipeline.run_cartpole_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_constrained_rnn_stable(tmp_path: pathlib.Path) -> None:
    model_name = 'ConstrainedRnn'
    model_class = 'deepsysid.models.recurrent.ConstrainedRnn'
    config = ConstrainedRnn.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
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
        bias=True,
        loss='mse',
        nonlinearity='torch.nn.Softshrink(0.5)',
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_lti_rnn_init(tmp_path: pathlib.Path) -> None:
    model_name = 'LtiRnnInit'
    model_class = 'deepsysid.models.recurrent.LtiRnnInit'
    config = LtiRnnInit.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        nx=2,
        recurrent_dim=3,
        num_recurrent_layers_init=3,
        dropout=0.25,
        sequence_length=3,
        learning_rate=0.1,
        batch_size=2,
        epochs_initializer=2,
        epochs_predictor=2,
        loss='mse',
        clip_gradient_norm=10,
        nonlinearity='torch.nn.Tanh()',
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_rnn_init(tmp_path: pathlib.Path) -> None:
    model_name = 'RnnInit'
    model_class = 'deepsysid.models.recurrent.RnnInit'
    config = RnnInit.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=5,
        num_recurrent_layers=3,
        dropout=0.25,
        bias=False,
        sequence_length=3,
        learning_rate=0.1,
        batch_size=2,
        epochs_initializer=2,
        epochs_predictor=2,
        loss='mse',
        clip_gradient_norm=10,
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_gru_init_model(tmp_path: pathlib.Path) -> None:
    model_name = 'GruInit'
    model_class = 'deepsysid.models.recurrent.GRUInitModel'
    config = GRUInitModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=5,
        num_recurrent_layers=3,
        dropout=0.25,
        bias=False,
        sequence_length=3,
        learning_rate=0.1,
        batch_size=2,
        epochs_initializer=2,
        epochs_predictor=2,
        loss='mse',
        clip_gradient_norm=10,
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_washout_initializer_rnn_model(tmp_path: pathlib.Path) -> None:
    model_name = 'WashoutRNN'
    model_class = 'deepsysid.models.recurrent.WashoutInitializerRNNModel'
    config = WashoutInitializerRNNModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=5,
        num_recurrent_layers=3,
        dropout=0.25,
        bias=True,
        sequence_length=2,
        learning_rate=0.1,
        batch_size=2,
        epochs=4,
        loss='mse',
        clip_gradient_norm=10,
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_washout_initializer_gru_model(tmp_path: pathlib.Path) -> None:
    model_name = 'WashoutGRU'
    model_class = 'deepsysid.models.recurrent.WashoutInitializerGRUModel'
    config = WashoutInitializerGRUModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=5,
        num_recurrent_layers=3,
        dropout=0.25,
        bias=True,
        sequence_length=2,
        learning_rate=0.1,
        batch_size=2,
        epochs=4,
        loss='mse',
        clip_gradient_norm=10,
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_washout_initializer_lstm_model(tmp_path: pathlib.Path) -> None:
    model_name = 'WashoutLSTM'
    model_class = 'deepsysid.models.recurrent.WashoutInitializerLSTMModel'
    config = WashoutInitializerLSTMModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=5,
        num_recurrent_layers=3,
        dropout=0.25,
        bias=False,
        sequence_length=2,
        learning_rate=0.1,
        batch_size=2,
        epochs=4,
        loss='mse',
        clip_gradient_norm=10,
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_joint_initializer_rnn_model(tmp_path: pathlib.Path) -> None:
    model_name = 'JointRNN'
    model_class = 'deepsysid.models.recurrent.JointInitializerRNNModel'
    config = JointInitializerRNNModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=5,
        num_recurrent_layers=3,
        dropout=0.25,
        bias=False,
        sequence_length=2,
        learning_rate=0.1,
        batch_size=2,
        epochs=4,
        loss='mse',
        clip_gradient_norm=10,
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_joint_initializer_gru_model(tmp_path: pathlib.Path) -> None:
    model_name = 'JointGRU'
    model_class = 'deepsysid.models.recurrent.JointInitializerGRUModel'
    config = JointInitializerGRUModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=5,
        num_recurrent_layers=3,
        dropout=0.25,
        bias=False,
        sequence_length=2,
        learning_rate=0.1,
        batch_size=2,
        epochs=4,
        loss='mse',
        clip_gradient_norm=10,
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_joint_initializer_lstm_model(tmp_path: pathlib.Path) -> None:
    model_name = 'JointLSTM'
    model_class = 'deepsysid.models.recurrent.JointInitializerLSTMModel'
    config = JointInitializerLSTMModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=5,
        num_recurrent_layers=3,
        dropout=0.25,
        bias=False,
        sequence_length=2,
        learning_rate=0.1,
        batch_size=2,
        epochs=4,
        loss='mse',
        clip_gradient_norm=10,
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_rnn_init_flexible_nonlinearity(tmp_path: pathlib.Path) -> None:
    model_name = 'RnnInitFlexibleNonlinearity'
    model_class = 'deepsysid.models.recurrent.RnnInitFlexibleNonlinearity'
    config = RnnInitFlexibleNonlinearity.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        recurrent_dim=5,
        bias=True,
        nonlinearity="torch.nn.Softshrink()",
        sequence_length=3,
        learning_rate=0.1,
        batch_size=2,
        epochs_initializer=2,
        epochs_predictor=2,
        loss='mse',
        clip_gradient_norm=10,
        num_recurrent_layers_init=2,
        dropout_init=0.25,
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_hybrid_minimal_maneuvering_model(tmp_path: pathlib.Path) -> None:
    model_name = 'HybridMinimalManeuveringModel'
    model_class = (
        'deepsysid.models.hybrid.bounded_residual.HybridMinimalManeuveringModel'
    )
    config = HybridMinimalManeuveringModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
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
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_hybrid_propulsion_maneuvering_model(tmp_path: pathlib.Path) -> None:
    model_name = 'HybridPropulsionManeuveringModel'
    model_class = (
        'deepsysid.models.hybrid.bounded_residual.HybridPropulsionManeuveringModel'
    )
    config = HybridPropulsionManeuveringModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
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
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_hybrid_linear_model(tmp_path: pathlib.Path) -> None:
    model_name = 'HybridLinearModel'
    model_class = 'deepsysid.models.hybrid.bounded_residual.HybridLinearModel'
    config = HybridLinearModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
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
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_hybrid_blanke_model(tmp_path: pathlib.Path) -> None:
    model_name = 'HybridBlankeModel'
    model_class = 'deepsysid.models.hybrid.bounded_residual.HybridBlankeModel'
    config = HybridBlankeModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
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
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_stable_switching_lstm_model(tmp_path: pathlib.Path) -> None:
    model_name = 'StableSwitchingLSTMModel'
    model_class = 'deepsysid.models.switching.switchrnn.StableSwitchingLSTMModel'
    config = StableSwitchingLSTMModel.CONFIG(
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
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_unconstrained_switching_lstm_model(tmp_path: pathlib.Path) -> None:
    model_name = 'UnconstrainedSwitchingLSTMModel'
    model_class = 'deepsysid.models.switching.switchrnn.UnconstrainedSwitchingLSTMModel'
    config = UnconstrainedSwitchingLSTMModel.CONFIG(
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
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_unconstrained_identity_output_switching_lstm_model(
    tmp_path: pathlib.Path,
) -> None:
    model_name = 'UnconstrainedSwitchingLSTMModel'
    model_class = (
        'deepsysid.models.switching.switchrnn.'
        'UnconstrainedIdentityOutputSwitchingLSTMModel'
    )
    config = UnconstrainedIdentityOutputSwitchingLSTMModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
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
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_stable_identity_output_switching_lstm_model(tmp_path: pathlib.Path) -> None:
    model_name = 'UnconstrainedSwitchingLSTMModel'
    model_class = (
        'deepsysid.models.switching.switchrnn.StableIdentityOutputSwitchingLSTMModel'
    )
    config = StableIdentityOutputSwitchingLSTMModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
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
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_k_linear_regression_model(tmp_path: pathlib.Path) -> None:
    model_name = 'KLinearRegressionARXModel'
    model_class = 'deepsysid.models.switching.klinreg.KLinearRegressionARXModel'
    config = KLinearRegressionARXModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        n_modes=2,
        lag=pipeline.get_window_size() - 1,
        probability_failure=0.001,
        initialization_bound=100.0,
        zero_probability_restarts=100,
        use_max_restarts=False,
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_serial_parallel_4dof_ship_model(tmp_path: pathlib.Path) -> None:
    model_name = 'SerialParallel4DOFShipModel'
    model_class = 'deepsysid.models.hybrid.serial.SerialParallel4DOFShipModel'
    config = SerialParallel4DOFShipModel.CONFIG(
        control_names=pipeline.get_4dof_ship_control_names(),
        state_names=pipeline.get_4dof_ship_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        epochs=3,
        batch_size=2,
        sequence_length=2,
        learning_rate=0.1,
        recurrent_dim=5,
        num_recurrent_layers=2,
        dropout=0.25,
        m=10.0,
        g=9.81,
        rho_water=0.5,
        disp=20.0,
        gm=4.0,
        ixx=0.5,
        izz=0.1,
        xg=0.3,
        zg=0.5,
        xud=0.1,
        yvd=0.2,
        ypd=-0.1,
        yrd=0.1,
        kvd=0.1,
        kpd=0.3,
        krd=-0.4,
        nvd=0.1,
        npd=0.2,
        nrd=0.1,
    )
    pipeline.run_4dof_ship_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )


def test_serial_parallel_quadcopter_model(tmp_path: pathlib.Path) -> None:
    model_name = 'SerialParallelQuadcopterModel'
    model_class = 'deepsysid.models.hybrid.serial.SerialParallelQuadcopterModel'
    config = SerialParallelQuadcopterModel.CONFIG(
        control_names=pipeline.get_quadcopter_control_names(),
        state_names=pipeline.get_quadcopter_state_names(),
        device_name=pipeline.get_cpu_device_name(),
        time_delta=pipeline.get_time_delta(),
        epochs=3,
        batch_size=2,
        sequence_length=2,
        learning_rate=0.1,
        recurrent_dim=5,
        num_recurrent_layers=2,
        dropout=0.25,
        m=0.5,
        g=9.81,
        l=0.2,
        d=0.01,
        ixx=0.25,
        izz=0.25,
        kr=1.0,
        kt=1.0,
    )
    pipeline.run_quadcopter_pipeline(
        tmp_path, model_name, model_class, model_config=config
    )
