import os
from typing import List, Tuple, Type

import numpy as np
import pandas as pd

from .models.base import DynamicIdentificationModel


def load_control_and_state(file_path: str, control_names: List[str], state_names: List[str]) -> Tuple[
    np.array, np.array]:
    sim = pd.read_csv(file_path)
    control_df = sim[control_names]
    state_df = sim[state_names]

    return control_df.values, state_df.values


def load_simulation_data(
        directory: str, control_names: List[str], state_names: List[str]
) -> Tuple[List[np.array], List[np.array]]:
    file_names = load_file_names(directory)

    controls = []
    states = []
    for fn in file_names:
        control, state = load_control_and_state(file_path=fn, control_names=control_names, state_names=state_names)
        controls.append(control)
        states.append(state)
    return controls, states


def load_file_names(directory: str) -> List[str]:
    file_names = []
    for fn in os.listdir(directory):
        if fn.endswith('.csv'):
            file_names.append(os.path.join(directory, fn))
    return sorted(file_names)


def load_model(model: DynamicIdentificationModel, directory: str, model_name: str):
    extension = model.get_file_extension()
    model.load(tuple(os.path.join(directory, f'{model_name}.{ext}') for ext in extension))


def save_model(model: DynamicIdentificationModel, directory: str, model_name: str):
    extension = model.get_file_extension()
    model.save(tuple(os.path.join(directory, f'{model_name}.{ext}') for ext in extension))


def retrieve_model_class(model_class_string: str) -> Type[DynamicIdentificationModel]:
    # https://stackoverflow.com/a/452981
    parts = model_class_string.split('.')
    module_string = '.'.join(parts[:-1])
    module = __import__(module_string)

    for component in parts[1:]:
        module = getattr(module, component)
    return module
