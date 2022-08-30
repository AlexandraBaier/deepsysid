import logging
import os
from typing import Dict

import h5py
import numpy as np
from numpy.typing import NDArray

from ..pipeline.configuration import ExperimentConfiguration, initialize_model
from .data_io import load_simulation_data
from .model_io import save_model

logger = logging.getLogger(__name__)


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
    try:
        os.mkdir(model_directory)
    except FileExistsError:
        pass

    logger.addHandler(
        logging.FileHandler(
            filename=os.path.join(model_directory, 'training.log'), mode='a'
        )
    )

    # Load dataset
    controls, states = load_simulation_data(
        directory=dataset_directory,
        control_names=configuration.control_names,
        state_names=configuration.state_names,
    )

    # Initialize model
    model = initialize_model(configuration, model_name, device_name)
    # Train model
    logger.info(f'Training model on {device_name} if implemented.')
    metadata = model.train(control_seqs=controls, state_seqs=states)
    # Save model metadata
    if metadata is not None:
        save_training_metadata(metadata, model_directory, model_name)
    # Save model
    save_model(model, model_directory, model_name)
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
