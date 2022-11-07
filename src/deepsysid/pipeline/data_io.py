import os
from typing import List, Literal, Optional, Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray


def load_control_and_state(
    file_path: str, control_names: List[str], state_names: List[str]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    sim = pd.read_csv(file_path)
    control_df = sim[control_names]
    state_df = sim[state_names]

    return control_df.values.astype(np.float64), state_df.values.astype(np.float64)


def load_simulation_data(
    directory: str, control_names: List[str], state_names: List[str]
) -> Tuple[List[NDArray[np.float64]], List[NDArray[np.float64]]]:
    file_names = load_file_names(directory)

    controls = []
    states = []
    for fn in file_names:
        control, state = load_control_and_state(
            file_path=fn, control_names=control_names, state_names=state_names
        )
        controls.append(control)
        states.append(state)
    return controls, states


def load_file_names(directory: str) -> List[str]:
    file_names = []
    for fn in os.listdir(directory):
        if fn.endswith('.csv'):
            file_names.append(os.path.join(directory, fn))
    return sorted(file_names)


def build_trajectory_file_name(
    mode: Literal['train', 'validation', 'test'],
    window_size: int,
    horizon_size: int,
    extension: str,
) -> str:
    return f'trajectory-{mode}-w_{window_size}-h_{horizon_size}.{extension}'


def build_score_file_name(
    mode: Literal['train', 'validation', 'test'],
    window_size: int,
    horizon_size: int,
    extension: str,
    threshold: Optional[float] = None,
) -> str:
    if threshold is None:
        return f'scores-{mode}-w_{window_size}-h_{horizon_size}.{extension}'

    threshold_str = f'{threshold:f}'.replace('.', '')
    return (
        f'scores-{mode}-w_{window_size}-h_{horizon_size}-t_{threshold_str}.{extension}'
    )


def build_result_file_name(
    mode: Literal['train', 'validation', 'test'],
    window_size: int,
    horizon_size: int,
    extension: str,
    threshold: Optional[float] = None,
) -> str:
    if threshold is None:
        return f'{mode}-w_{window_size}-h_{horizon_size}.{extension}'

    threshold_str = f'{threshold:f}'.replace('.', '')
    return (
        f'threshold_hybrid_{mode}-w_{window_size}'
        f'-h_{horizon_size}-t_{threshold_str}.{extension}'
    )


def build_explanation_result_file_name(
    mode: Literal['train', 'validation', 'test'],
    window_size: int,
    horizon_size: int,
    extension: str,
) -> str:
    return f'{mode}-explanation-w_{window_size}-h_{horizon_size}.{extension}'
