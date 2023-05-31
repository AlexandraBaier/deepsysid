import os
from typing import List, Literal, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def load_control_and_state(
    file_path: str,
    control_names: List[str],
    state_names: List[str],
    initial_state_names: List[str],
) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    sim = pd.read_csv(file_path)
    control_df = sim[control_names]
    state_df = sim[state_names]
    initial_state_df = sim[initial_state_names]

    return (
        control_df.values.astype(np.float64),
        state_df.values.astype(np.float64),
        initial_state_df.values.astype(np.float64),
    )


def load_simulation_data(
    directory: str,
    control_names: List[str],
    state_names: List[str],
    initial_state_names: List[str],
) -> Tuple[
    List[NDArray[np.float64]], List[NDArray[np.float64]], List[NDArray[np.float64]]
]:
    file_names = load_file_names(directory)

    controls = []
    states = []
    initial_states = []
    for fn in file_names:
        control, state, initial_state = load_control_and_state(
            file_path=fn,
            control_names=control_names,
            state_names=state_names,
            initial_state_names=initial_state_names,
        )
        controls.append(control)
        states.append(state)
        initial_states.append(initial_state)
    return controls, states, initial_states


def load_file_names(directory: str) -> List[str]:
    file_names = []
    for fn in os.listdir(directory):
        if fn.endswith('.csv'):
            file_names.append(os.path.join(directory, fn))
    return sorted(file_names)


def build_score_file_name(
    mode: Literal['train', 'validation', 'test'],
    window_size: int,
    horizon_size: int,
    extension: str,
) -> str:
    return f'scores-{mode}-w_{window_size}-h_{horizon_size}.{extension}'


def build_result_file_name(
    mode: Literal['train', 'validation', 'test'],
    window_size: int,
    horizon_size: int,
    extension: str,
) -> str:
    return f'{mode}-w_{window_size}-h_{horizon_size}.{extension}'


def build_explanation_result_file_name(
    mode: Literal['train', 'validation', 'test'],
    window_size: int,
    horizon_size: int,
    extension: str,
) -> str:
    return f'{mode}-explanation-w_{window_size}-h_{horizon_size}.{extension}'


def build_tracker_config_file_name(tracker_name: str, model_name: str) -> str:
    return f'tracker-{tracker_name}-{model_name}.json'
