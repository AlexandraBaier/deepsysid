import os
from typing import List, Tuple

import numpy as np
import pandas as pd


def load_control_and_state(
    file_path: str, control_names: List[str], state_names: List[str]
) -> Tuple[np.ndarray, np.ndarray]:
    sim = pd.read_csv(file_path)
    control_df = sim[control_names]
    state_df = sim[state_names]

    return control_df.values, state_df.values


def load_simulation_data(
    directory: str, control_names: List[str], state_names: List[str]
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
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
