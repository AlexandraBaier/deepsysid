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

    time_delta = config['time_delta']
    window_size = config['window']
    horizon_size = config['horizon']
    control_names = config['control_names']
    state_names = config['state_names']

    # Load dataset
    dataset_directory = os.path.join(os.environ['DATASET_DIRECTORY'], 'processed', 'train')
    controls, states = execution.load_simulation_data(
        directory=dataset_directory, control_names=control_names, state_names=state_names)

    # Initialize model
    model_config = config['models'][model_name]
    model_directory = os.path.expanduser(os.path.normpath(model_config['location']))
    model_class = execution.retrieve_model_class(model_config['model_class'])

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

    parameters = model_config['parameters']
    parameters['control_names'] = control_names
    parameters['state_names'] = state_names
    parameters['time_delta'] = time_delta
    parameters['window_size'] = window_size
    parameters['horizon_size'] = horizon_size
    parameters['device_name'] = device_name

    if model_class.CONFIG is not None:
        config = model_class.CONFIG.parse_obj(parameters)
    else:
        config = None

    model = model_class(config=config)
    model.train(control_seqs=controls, state_seqs=states)

    execution.save_model(model, model_directory, model_name)


if __name__ == '__main__':
    main()
