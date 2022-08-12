import numpy as np

from deepsysid import execution


def test_linear_model_train_simulate_cpu():
    control_dim = 2
    state_dim = 3
    window_size = 10
    horizon_size = 20
    n_training_sequences = 3

    config = execution.ExperimentConfiguration.parse_obj(dict(
        train_fraction=0.6,
        validation_fraction=0.5,
        time_delta=0.5,
        window_size=window_size,
        horizon_size=horizon_size,
        control_names=[f'u{i}' for i in range(control_dim)],
        state_names=[f'x{i}' for i in range(state_dim)],
        models=dict(
            LinearModel=dict(
                model_class='deepsysid.models.linear.LinearModel',
                location='.tmp',
                parameters=dict(
                    control_names=[f'u{i}' for i in range(control_dim)],
                    state_names=[f'x{i}' for i in range(state_dim)],
                    time_delta=0.5
                )
            )
        )
    ))
    model_name = 'LinearModel'
    device_name = 'cpu'

    control_seqs = [
        np.random.normal(0, 1, (100, control_dim))
        for _ in range(n_training_sequences)
    ]
    state_seqs = [
        np.random.normal(0, 1, (100, state_dim))
        for _ in range(n_training_sequences)
    ]

    initial_control = np.random.normal(0, 1, (window_size, control_dim))
    initial_state = np.random.normal(0, 1, (window_size, state_dim))
    control = np.random.normal(0, 1, (horizon_size, control_dim))

    model = execution.initialize_model(config, model_name, device_name)
    model.train(control_seqs, state_seqs)
    model.simulate(initial_control, initial_state, control)
    model.get_file_extension()
    model.get_parameter_count()
