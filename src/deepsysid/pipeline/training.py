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
        global tracking
        try:
            tracking = importlib.import_module(configuration.tracker.module_name)
        except ImportError:
            tracking = None
        if configuration.tracker.tracking_uri is not None:
            tracking.set_tracking_uri(configuration.tracker.tracking_uri)
        start_run = time.time()
        tracking.start_run()
        duration_run = time.time() - start_run
        logger.info(f'Ml flow initialize session: {time.strftime("%H:%M:%S", time.gmtime(float(duration_run)))}')
        callback: Callable[[Dict[str, Any]], None] = lambda _:None
        try:
            callback = eval(configuration.tracker.callback_name)
        except SyntaxError:
            callback = lambda _:None

    # Initialize model
    model = initialize_model(configuration, model_name, device_name)
    # Train model
    logger.info(f'Training model {model_name} on {device_name} if implemented.')
    start_training = time.time()
    metadata = model.train(
        control_seqs=controls, state_seqs=states, initial_seqs=initial_states, callback=callback
    )
    # Save model metadata
    if metadata is not None:
        save_training_metadata(metadata, model_directory, model_name)
    # Save model
    save_model(model, model_directory, model_name)
    training_duration = time.time() - start_training
    logger.info(f'training time: {time.strftime("%H:%M:%S", time.gmtime(float(training_duration)))}')
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

def mlflow_tracking(logs: Dict[str, Any])-> None:
    if 'metric' in logs:
        for metric_list in logs['metric']:
            tracking.log_metric(metric_list[0], metric_list[1], step=metric_list[2])
    if 'figures' in logs:
        for figures_list in logs['figures']:
            tracking.log_figure(figure=figures_list[0], artifact_file=figures_list[1])
