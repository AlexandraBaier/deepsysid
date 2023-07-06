import logging
import os
import time
from types import ModuleType
from typing import Dict, Optional

import h5py
import numpy as np
from numpy.typing import NDArray

from ..pipeline.configuration import (
    ExperimentConfiguration,
    initialize_model,
    initialize_tracker,
)
from ..tracker.event_data import (
    SaveTrackingConfiguration,
    SetExperiment,
    SetTags,
    StopRun,
    TrackArtifacts
)
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

    # set tracking
    tracker = initialize_tracker(experiment_config=configuration)
    tracker(SetExperiment('Start new experiment', dataset_directory))

    # Load dataset
    controls, states, initial_states = load_simulation_data(
        directory=dataset_directory,
        control_names=configuration.control_names,
        state_names=configuration.state_names,
        initial_state_names=configuration.initial_state_names,
    )

    # Initialize model
    model = initialize_model(configuration, model_name, device_name)
    # Train model
    logger.info(f'Training model {model_name} on {device_name} if implemented.')
    start_training = time.time()
    metadata = model.train(
        control_seqs=controls,
        state_seqs=states,
        initial_seqs=initial_states,
        tracker=tracker,
    )
    # Save model metadata
    if metadata is not None:
        save_training_metadata(metadata, model_directory, model_name)
    # Save model
    save_model(model, model_directory, model_name, tracker)
    training_duration = time.time() - start_training
    logger.info(
        f'training time: '
        f'{time.strftime("%H:%M:%S", time.gmtime(float(training_duration)))}'
    )
    tracker(SetTags('Trained, not validated', {'trained': True, 'validated': False}))
    if configuration.tracker is not None:
        tracker(
            SaveTrackingConfiguration(
                'Write tracking config to json file',
                configuration.tracker,
                model_name,
                model_directory,
            )
        )
    
    # Save model configuration
    config_file_path = os.path.join(model_directory, f'config-{model_name}.json')
    with open(
        config_file_path, mode='w'
    ) as f:
        f.write(configuration.models[model_name].json())

    tracker(TrackArtifacts(
        'Save model configuration',
        {'model configuration': config_file_path}
    ))
    tracker(StopRun('stop current run', None))


def save_training_metadata(
    metadata: Dict[str, NDArray[np.float64]], directory: str, model_name: str
) -> None:
    with h5py.File(os.path.join(directory, f'{model_name}-metadata.hdf5'), 'w') as f:
        for name, data in metadata.items():
            f.create_dataset(name, data=data)
