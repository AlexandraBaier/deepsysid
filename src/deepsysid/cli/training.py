import logging
import os
import sys

from deepsysid import execution


def train_model(
    configuration: execution.ExperimentConfiguration,
    model_name: str,
    device_name: str,
    dataset_directory: str,
    models_directory: str,
    disable_stdout: bool,
):
    dataset_directory = os.path.join(dataset_directory, 'processed', 'train')
    model_directory = os.path.expanduser(
        os.path.normpath(os.path.join(models_directory, model_name))
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
        control_names=configuration.control_names,
        state_names=configuration.state_names,
    )

    # Initialize model
    model = execution.initialize_model(configuration, model_name, device_name)
    # Train model
    logger.info(f'Training model on {device_name} if implemented.')
    metadata = model.train(control_seqs=controls, state_seqs=states)
    # Save model metadata
    if metadata is not None:
        execution.save_training_metadata(metadata, model_directory, model_name)
    # Save model
    execution.save_model(model, model_directory, model_name)
