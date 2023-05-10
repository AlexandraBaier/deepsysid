import pathlib

from deepsysid.pipeline.data_io import load_simulation_data

from ..smoke_tests import pipeline


def test_load_simulation_data(tmp_path: pathlib.Path) -> None:
    paths = pipeline.prepare_directories(base_path=tmp_path)

    # Setup dataset directory.
    paths['train'].joinpath('train-0.csv').write_text(
        data=pipeline.get_cartpole_data(0)
    )
    paths['validation'].joinpath('validation-0.csv').write_text(
        data=pipeline.get_cartpole_data(1)
    )
    paths['test'].joinpath('test-0.csv').write_text(data=pipeline.get_cartpole_data(2))

    N = 26
    state_names = pipeline.get_cartpole_state_names()
    control_names = pipeline.get_cartpole_control_names()
    initial_state_names = pipeline.get_cartpole_initial_state_names()

    controls, states, initial_states = load_simulation_data(
        directory=paths['train'],
        control_names=control_names,
        state_names=state_names,
        initial_state_names=initial_state_names,
    )

    assert initial_states[0].shape == (N, len(initial_state_names))
    assert controls[0].shape == (N, len(control_names))
    assert states[0].shape == (N, len(state_names))
