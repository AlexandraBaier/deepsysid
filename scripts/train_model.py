import argparse
import json
import logging
import os
import sys

import deepsysid.execution as execution

logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('model', help='model in experiment')
    parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--device-idx', action='store', help='Index of GPU')
    parser.add_argument('--disable-stdout', action='store_true', help='Prevent logging to reach STDOUT')
    args = parser.parse_args()

    model_name = args.model
    if args.enable_cuda:
        logger.info('Training model on CUDA if implemented.')
        if args.device_idx:
            device_name = f'cuda:{args.device_idx}'
        else:
            device_name = 'cuda'
    else:
        device_name = 'cpu'

    # Load configuration
    with open(os.environ['CONFIGURATION'], mode='r') as f:
        config = json.load(f)

    config = execution.ExperimentConfiguration.parse_obj(config)

    # Load dataset
    dataset_directory = os.path.join(os.environ['DATASET_DIRECTORY'], 'processed', 'train')
    controls, states = execution.load_simulation_data(
        directory=dataset_directory, control_names=config.control_names, state_names=config.state_names
    )

    # Initialize model
    model_directory = os.path.expanduser(os.path.normpath(config.models[model_name].location))
    try:
        os.mkdir(model_directory)
    except FileExistsError:
        pass

    handlers = [logging.FileHandler(filename=os.path.join(model_directory, 'training.log'), mode='a')]
    if not args.disable_stdout:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        format='%(asctime)s : %(levelname)s : %(message)s',
        level=logging.INFO,
        handlers=handlers
    )

    model = execution.initialize_model(config, model_name, device_name)
    model.train(control_seqs=controls, state_seqs=states)
    execution.save_model(model, model_directory, model_name)


if __name__ == '__main__':
    main()
