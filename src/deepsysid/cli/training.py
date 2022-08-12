import json
import logging
import os
import sys

from deepsysid import execution


def train_model(
    model_name: str,
    device_name: str,
    configuration_path: str,
    dataset_directory: str,
    disable_stdout: bool,
):
    # Load configuration
    with open(configuration_path, mode='r') as f:
        config = json.load(f)

    config = execution.ExperimentConfiguration.parse_obj(config)

    dataset_directory = os.path.join(dataset_directory, 'processed', 'train')
    model_directory = os.path.expanduser(
        os.path.normpath(config.models[model_name].location)
    )
    try:
        os.mkdir(model_directory)
    except FileExistsError:
        pass

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(
        logging.FileHandler(
            filename=os.path.join(model_directory, 'training.log'), mode='a'
        )
    )
    if not disable_stdout:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    # Load dataset
    controls, states = execution.load_simulation_data(
        directory=dataset_directory,
        control_names=config.control_names,
        state_names=config.state_names,
    )

    # Initialize model
    model = execution.initialize_model(config, model_name, device_name)
    # Train model
    logger.info(f'Training model on {device_name} if implemented.')
    model.train(control_seqs=controls, state_seqs=states)
    # Save model
    execution.save_model(model, model_directory, model_name)
