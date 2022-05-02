import argparse
import json
import logging
import os

import h5py
import numpy as np

import sysid.execution as execution
import sysid.models.base as base

logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('model', help='model')
    parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--device-idx', action='store', help='Index of GPU')
    parser.add_argument('--mode', action='store', help='either "validation" or "test"')
    args = parser.parse_args()

    model_name = args.model
    mode = args.mode
    if mode not in {'validation', 'test'}:
        raise ValueError('Argument to --mode must be either "validation" or "test"')

    if args.enable_cuda:
        logger.info('Training model on CUDA if implemented.')
        if args.device_idx:
            device_name = f'cuda:{args.device_idx}'
        else:
            device_name = 'cuda'
    else:
        device_name = 'cpu'

    with open(os.environ['CONFIGURATION'], mode='r') as f:
        config = json.load(f)

    time_delta = config['time_delta']
    window_size = config['window']
    horizon_size = config['horizon']
    control_names = config['control_names']
    state_names = config['state_names']

    # Initialize and load model
    model_config = config['models'][model_name]
    model_directory = os.path.expanduser(os.path.normpath(model_config['location']))

    model_class = execution.retrieve_model_class(model_config['model_class'])
    params = model_config['parameters']
    params['time_delta'] = time_delta
    params['window_size'] = window_size
    params['horizon_size'] = horizon_size
    params['control_names'] = control_names
    params['state_names'] = state_names
    params['verbose'] = True
    params['device_name'] = device_name

    model = model_class(**params)  # type: base.DynamicIdentificationModel
    execution.load_model(model, model_directory, model_name)

    # Prepare test data
    dataset_directory = os.path.join(os.environ['DATASET_DIRECTORY'], 'processed', mode)

    file_names = list(map(lambda fn: os.path.basename(fn).split('.')[0],
                          execution.load_file_names(dataset_directory)))
    controls, states = execution.load_simulation_data(
        directory=dataset_directory, control_names=control_names, state_names=state_names)

    simulations = list(zip(controls, states, file_names))

    # Execute predictions on test data
    control = []
    pred_states = []
    true_states = []
    file_names = []
    for initial_control, initial_state, true_control, true_state, file_name \
            in split_simulations(window_size, horizon_size, simulations):
        pred_target = model.simulate(initial_control, initial_state, true_control)

        control.append(true_control)
        pred_states.append(pred_target)
        true_states.append(true_state)
        file_names.append(file_name)

    # Save true and predicted time series
    result_directory = os.path.join(os.environ['RESULT_DIRECTORY'], model_name)
    try:
        os.mkdir(result_directory)
    except FileExistsError:
        pass

    result_file_path = os.path.join(result_directory, f'{mode}-w_{window_size}-h_{horizon_size}.hdf5')
    with h5py.File(result_file_path, 'w') as f:
        f.attrs['control_names'] = np.array([np.string_(name) for name in control_names])
        f.attrs['state_names'] = np.array([np.string_(name) for name in state_names])

        f.create_dataset('file_names', data=np.array(list(map(np.string_, file_names))))

        control_grp = f.create_group('control')
        pred_grp = f.create_group('predicted')
        true_grp = f.create_group('true')
        for i, (control, pred_state, true_state) in enumerate(zip(control, pred_states, true_states)):
            control_grp.create_dataset(str(i), data=control)
            pred_grp.create_dataset(str(i), data=pred_state)
            true_grp.create_dataset(str(i), data=true_state)


def split_simulations(window_size, horizon_size, simulations):
    total_length = window_size + horizon_size
    for control, state, file_name in simulations:
        for i in range(total_length, control.shape[0], total_length):
            initial_control = control[i-total_length:i - total_length + window_size]
            initial_state = state[i-total_length:i - total_length + window_size]
            true_control = control[i - total_length + window_size:i]
            true_state = state[i - total_length + window_size:i]

            yield initial_control, initial_state, true_control, true_state, file_name


if __name__ == '__main__':
    main()
