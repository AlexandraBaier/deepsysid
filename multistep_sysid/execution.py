from collections import namedtuple
import os
import typing as T

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from .models import base
from . import utils


def load_control_and_state(file_path: str, control_names: T.List[str], state_names: T.List[str]) -> T.Tuple[
    np.array, np.array]:
    sim = pd.read_csv(file_path)
    control_df = sim[control_names]
    state_df = sim[state_names]

    return control_df.values, state_df.values


def load_simulation_data(directory: str, control_names: T.List[str], state_names: T.List[str]) -> T.Tuple[
    T.List[np.array], T.List[np.array]]:
    file_names = load_file_names(directory)

    controls = []
    states = []
    for fn in file_names:
        control, state = load_control_and_state(file_path=fn, control_names=control_names, state_names=state_names)
        controls.append(control)
        states.append(state)
    return controls, states


def load_file_names(directory: str) -> T.List[str]:
    file_names = []
    for fn in os.listdir(directory):
        if fn.endswith('.csv'):
            file_names.append(os.path.join(directory, fn))
    return sorted(file_names)


def load_model(model: base.DynamicIdentificationModel, directory: str, model_name: str):
    extension = model.get_file_extension()
    if type(extension) == str:
        model.load(os.path.join(directory, f'{model_name}.{extension}'))
    else:
        model.load([os.path.join(directory, f'{model_name}.{ext}') for ext in extension])


def save_model(model: base.DynamicIdentificationModel, directory: str, model_name: str):
    extension = model.get_file_extension()
    if type(extension) == str:
        model.save(os.path.join(directory, f'{model_name}.{extension}'))
    else:
        model.save([os.path.join(directory, f'{model_name}.{ext}') for ext in extension])


def retrieve_model_class(model_class_string: str) -> T.Type[base.DynamicIdentificationModel]:
    # https://stackoverflow.com/a/452981
    parts = model_class_string.split('.')
    module_string = '.'.join(parts[:-1])
    module = __import__(module_string)

    for component in parts[1:]:
        module = getattr(module, component)
    return module


class MAEEpochValidator(object):
    ValidationSet = namedtuple('ValidationSet', ('initial_control', 'initial_state', 'control', 'state'))

    def __init__(self, controls: T.List[np.array], states: T.List[np.array], window_size: int, horizon_size: int,
                 trigger_after_epoch: int = 5):
        self.after_epoch = trigger_after_epoch  # type: int
        self.state_dim = states[0].shape[1]

        self.validation_set = []  # type: T.List[MAEEpochValidator.ValidationSet]

        total_length = window_size + horizon_size
        for control, state in zip(controls, states):
            for i in range(total_length, control.shape[0], total_length):
                initial_control = control[i - total_length:i - total_length + window_size]
                initial_state = state[i - total_length:i - total_length + window_size]
                true_control = control[i - total_length + window_size:i]
                true_state = state[i - total_length + window_size:i]

                self.validation_set.append(
                    MAEEpochValidator.ValidationSet(initial_control, initial_state, true_control, true_state))

    def __call__(self, model: base.DynamicIdentificationModel, epoch: int) -> T.Optional[T.List[float]]:
        if epoch == 0 or ((epoch+1) % self.after_epoch) != 0:
            return None

        try:
            error = np.mean(utils.score_on_sequence(
                [vs.state for vs in self.validation_set],
                [model.simulate(vs.initial_control, vs.initial_state, vs.control) for vs in self.validation_set],
                lambda t, p: mean_absolute_error(t, p, multioutput='raw_values')
            ), axis=0)
        except ValueError:
            error = []

        return list(error)
