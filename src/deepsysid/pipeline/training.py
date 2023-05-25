import logging
import os
from typing import Dict, Any, Optional, Callable
from types import ModuleType
import importlib

import h5py
import numpy as np
from numpy.typing import NDArray
import time

from ..pipeline.configuration import ExperimentConfiguration, initialize_model
from .data_io import load_simulation_data
from .model_io import save_model
from ..tracker.base import retrieve_tracker_class

logger = logging.getLogger(__name__)
tracking: Optional[ModuleType] = None


def train_model(
    configuration: ExperimentConfiguration,
    model_name: str,
    device_name: str,
    dataset_directory: str,
    models_directory: str,
) -> None:
    dataset_directory = os.path.join(dataset_directory, 'processed', 'train')
    model_directory = os.path.expanduser(
        os.path.normpath(os.path.join(models_directory, model_name))
    )
    os.makedirs(model_directory, exist_ok=True)

    # Load dataset
    controls, states, initial_states = load_simulation_data(
        directory=dataset_directory,
        control_names=configuration.control_names,
        state_names=configuration.state_names,
        initial_state_names=configuration.initial_state_names,
    )
    # set tracking
    if configuration.tracker is not None:
        for tracker_config in configuration.tracker.values():
            tracker_class = retrieve_tracker_class(tracker_config.tracking_class)
            callback = tracker_class(tracker_config.parameters)

    # Initialize model
    model = initialize_model(configuration, model_name, device_name)
    # Train model
    logger.info(f'Training model {model_name} on {device_name} if implemented.')
    start_training = time.time()
    metadata = model.train(
        control_seqs=controls,
        state_seqs=states,
        initial_seqs=initial_states,
        callback=callback,
    )
    # Save model metadata
    if metadata is not None:
        save_training_metadata(metadata, model_directory, model_name)
    # Save model
    save_model(model, model_directory, model_name)
    training_duration = time.time() - start_training
    logger.info(
        f'training time: {time.strftime("%H:%M:%S", time.gmtime(float(training_duration)))}'
    )
    # Save model configuration
    with open(
        os.path.join(model_directory, f'config-{model_name}.json'), mode='w'
    ) as f:
        f.write(configuration.models[model_name].json())


def save_training_metadata(
    metadata: Dict[str, NDArray[np.float64]], directory: str, model_name: str
) -> None:
    with h5py.File(os.path.join(directory, f'{model_name}-metadata.hdf5'), 'w') as f:
        for name, data in metadata.items():
            f.create_dataset(name, data=data)
