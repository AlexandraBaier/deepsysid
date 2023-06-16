import pathlib

import numpy as np

from deepsysid.pipeline.data_io import load_simulation_data
from deepsysid.pipeline.testing.base import TestSimulation
from deepsysid.pipeline.testing.io import split_simulations

from ..smoke_tests import pipeline


def test_split_simulation(tmp_path: pathlib.Path) -> None:
    paths = pipeline.prepare_directories(base_path=tmp_path)

    # Setup dataset directory.
    paths['train'].joinpath('train-0.csv').write_text(
        data=pipeline.get_cartpole_data(0)
    )
    paths['validation'].joinpath('validation-0.csv').write_text(
        data=pipeline.get_cartpole_data(1)
    )
    paths['test'].joinpath('test-0.csv').write_text(data=pipeline.get_cartpole_data(2))

    file_names = ['test-0.csv']
    window_size = 4
    horizon_isze = 8

    controls, states, initial_states = load_simulation_data(
        directory=str(paths['test']),
        control_names=pipeline.get_cartpole_control_names(),
        state_names=pipeline.get_cartpole_state_names(),
        initial_state_names=pipeline.get_cartpole_initial_state_names(),
    )

    simulations = [
        TestSimulation(control, state, initial_state, file_name)
        for control, state, initial_state, file_name in zip(
            controls, states, initial_states, file_names
        )
    ]
    x0_list = [
        np.array(
            [
                -0.004342489483927282,
                -0.10859271339593855,
                0.0065125973112120945,
                0.16348772749345103,
            ]
        ),
        np.array(
            [
                -0.06995460409853099,
                -0.4406927563078292,
                0.1148971973101082,
                0.7982591762446896,
            ]
        ),
    ]

    init_wp_list = [
        np.array(
            [
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
            ]
        ).reshape((4, 1)),
        np.array(
            [
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
            ]
        ).reshape((4, 1)),
    ]

    init_zp_list = [
        np.array(
            [0.0, 0.0004064256313086258, 0.0016247181384796887, 0.00365755129657772]
        ).reshape((4, 1)),
        np.array(
            [
                0.06156217665390031,
                0.07303095250779881,
                0.0856964111432312,
                0.09962711113954151,
            ]
        ).reshape((4, 1)),
    ]

    zp_list = [
        np.array(
            [
                0.0065125973112120945,
                0.010202456620048565,
                0.014744444987805354,
                0.020160025212792122,
                0.02647824765797335,
                0.03373415349848603,
                0.041968370143501695,
                0.05122711123622628,
            ]
        ).reshape((8, 1)),
        np.array(
            [
                0.1148971973101082,
                0.13157946092654152,
                0.149761556901757,
                0.16955435982217,
                0.19107520751295737,
                0.2144478182659306,
                0.2398022908395357,
                0.2672751044588533,
            ]
        ).reshape((8, 1)),
    ]

    wp_list = [
        np.array(
            [
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
            ]
        ).reshape((8, 1)),
        np.array(
            [
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
                -1.3909720801108492,
            ]
        ).reshape((8, 1)),
    ]

    for idx, sim in enumerate(
        split_simulations(
            window_size=window_size, horizon_size=horizon_isze, simulations=simulations
        )
    ):
        assert np.linalg.norm(sim.x0 - x0_list[idx]) < 1e-10
        assert np.linalg.norm(sim.initial_control - init_wp_list[idx]) < 1e-10
        assert np.linalg.norm(sim.initial_state - init_zp_list[idx]) < 1e-10
        assert np.linalg.norm(sim.true_control - wp_list[idx]) < 1e-10
        assert np.linalg.norm(sim.true_state - zp_list[idx]) < 1e-10
