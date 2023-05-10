from typing import List

import numpy as np

from deepsysid.models.linear import LinearLag, LinearModel, QuadraticControlLag


def get_control_names(dim: int) -> List[str]:
    return [f'u{i}' for i in range(dim)]


def get_state_names(dim: int) -> List[str]:
    return [f'x{i}' for i in range(dim)]


def get_time_delta() -> float:
    return 0.5


def test_linear_model():
    np.random.seed(1)

    batch_size = 10
    control_dim = 3
    state_dim = 2

    parameter_count = (control_dim + state_dim) * state_dim + state_dim

    model = LinearModel(
        LinearModel.CONFIG(
            control_names=get_control_names(control_dim),
            state_names=get_state_names(state_dim),
            time_delta=get_time_delta(),
        )
    )
    assert model.get_parameter_count() == parameter_count

    model.train(
        control_seqs=[np.random.normal(0, 1, (batch_size, control_dim))],
        state_seqs=[np.random.normal(0, 1, (batch_size, state_dim))],
        initial_seqs=None,
    )
    assert model.get_parameter_count() == parameter_count


def test_linear_lag():
    np.random.seed(1)

    batch_size = 10
    control_dim = 3
    state_dim = 2
    lag = 3

    parameter_count = (
        state_dim * (control_dim + lag * (control_dim + state_dim)) + state_dim
    )

    model = LinearLag(
        LinearLag.CONFIG(
            control_names=get_control_names(control_dim),
            state_names=get_state_names(state_dim),
            time_delta=get_time_delta(),
            lag=lag,
        )
    )
    assert model.get_parameter_count() == parameter_count

    model.train(
        control_seqs=[np.random.normal(0, 1, (batch_size, control_dim))],
        state_seqs=[np.random.normal(0, 1, (batch_size, state_dim))],
        initial_seqs=None,
    )
    assert model.get_parameter_count() == parameter_count


def test_quadratic_control_lag():
    np.random.seed(1)

    batch_size = 10
    control_dim = 3
    state_dim = 2
    lag = 3

    parameter_count = (
        state_dim * (2 * control_dim + lag * (2 * control_dim + state_dim)) + state_dim
    )

    model = QuadraticControlLag(
        QuadraticControlLag.CONFIG(
            control_names=get_control_names(control_dim),
            state_names=get_state_names(state_dim),
            time_delta=get_time_delta(),
            lag=lag,
        )
    )
    assert model.get_parameter_count() == parameter_count

    model.train(
        control_seqs=[np.random.normal(0, 1, (batch_size, control_dim))],
        state_seqs=[np.random.normal(0, 1, (batch_size, state_dim))],
        initial_seqs=None,
    )
    assert model.get_parameter_count() == parameter_count
