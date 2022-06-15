import itertools
import os
from typing import List, Tuple, Type, Optional, Dict, Any

import numpy as np
import pandas as pd
from pydantic import BaseModel

from .models.base import DynamicIdentificationModel


class ExperimentModelConfiguration(BaseModel):
    model_class: str
    location: str
    parameters: Dict[str, Any]


class ExperimentGridSearchSettings(BaseModel):
    train_fraction: float
    validation_fraction: float
    time_delta: float
    window_size: int
    horizon_size: int
    control_names: List[str]
    state_names: List[str]
    thresholds: Optional[List[float]]


class ModelGridSearchTemplate(BaseModel):
    model_base_name: str
    model_class: str
    static_parameters: Dict[str, Any]
    flexible_parameters: Dict[str, List[Any]]


class ExperimentGridSearchTemplate(BaseModel):
    base_path: str
    settings: ExperimentGridSearchSettings
    models: List[ModelGridSearchTemplate]


class ExperimentConfiguration(BaseModel):
    train_fraction: float
    validation_fraction: float
    time_delta: float
    window_size: int
    horizon_size: int
    control_names: List[str]
    state_names: List[str]
    thresholds: Optional[List[float]]
    models: Dict[str, ExperimentModelConfiguration]

    @classmethod
    def from_grid_search_template(
        cls, template: ExperimentGridSearchTemplate, device_name: str
    ) -> 'ExperimentConfiguration':
        models: Dict[str, ExperimentModelConfiguration] = dict()
        for model_template in template.models:
            model_class_str = model_template.model_class
            model_class = retrieve_model_class(model_class_str)
            base_model_params = dict(
                device_name=device_name,
                control_names=template.settings.control_names,
                state_names=template.settings.state_names,
                time_delta=template.settings.time_delta,
                window_size=template.settings.window_size,
                horizon_size=template.settings.horizon_size,
            )
            static_model_params = model_template.static_parameters

            for combination in itertools.product(
                *list(model_template.flexible_parameters.values())
            ):
                model_name = model_template.model_base_name
                flexible_model_params = dict()
                for param_name, param_value in zip(
                    model_template.flexible_parameters.keys(), combination
                ):
                    flexible_model_params[param_name] = param_value
                    if issubclass(type(param_value), list):
                        model_name += '-' + '_'.join(map(str, param_value))
                    else:
                        if isinstance(param_value, float):
                            model_name += f'-{param_value:f}'.replace('.', '')
                        else:
                            model_name += f'-{param_value}'.replace('.', '')

                model_config = ExperimentModelConfiguration.parse_obj(
                    dict(
                        model_class=model_class_str,
                        location=os.path.join(template.base_path, model_name),
                        parameters=model_class.CONFIG.parse_obj(
                            # Merge dictionaries
                            {
                                **base_model_params,
                                **static_model_params,
                                **flexible_model_params,
                            }
                        ),
                    )
                )
                models[model_name] = model_config

        return cls(
            train_fraction=template.settings.train_fraction,
            validation_fraction=template.settings.validation_fraction,
            time_delta=template.settings.time_delta,
            window_size=template.settings.window_size,
            horizon_size=template.settings.horizon_size,
            control_names=template.settings.control_names,
            state_names=template.settings.state_names,
            thresholds=template.settings.thresholds,
            models=models,
        )


def initialize_model(
    experiment_config: ExperimentConfiguration, model_name: str, device_name: str
) -> DynamicIdentificationModel:
    model_config = experiment_config.models[model_name]

    model_class = retrieve_model_class(model_config.model_class)

    parameters = model_config.parameters
    parameters['device_name'] = device_name

    config = model_class.CONFIG.parse_obj(parameters)

    model = model_class(config=config)

    return model


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


def load_model(model: DynamicIdentificationModel, directory: str, model_name: str):
    extension = model.get_file_extension()
    model.load(
        tuple(os.path.join(directory, f'{model_name}.{ext}') for ext in extension)
    )


def save_model(model: DynamicIdentificationModel, directory: str, model_name: str):
    extension = model.get_file_extension()
    model.save(
        tuple(os.path.join(directory, f'{model_name}.{ext}') for ext in extension)
    )


def retrieve_model_class(model_class_string: str) -> Type[DynamicIdentificationModel]:
    # https://stackoverflow.com/a/452981
    parts = model_class_string.split('.')
    module_string = '.'.join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[2])
    for component in parts[1:]:
        cls = getattr(cls, component)

    if not issubclass(cls, DynamicIdentificationModel):
        raise ValueError(f'{cls} is not a subclass of DynamicIdentificationModel')
    return cls
