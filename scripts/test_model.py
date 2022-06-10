import argparse
import json
import logging
import os

import h5py
import numpy as np

import deepsysid.execution as execution

logger = logging.getLogger()


def main():
    parser = argparse.ArgumentParser(description='Test model')
    parser.add_argument('model', help='model')
    parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
    parser.add_argument('--device-idx', action='store', help='Index of GPU')
    parser.add_argument('--mode', action='store', help='either "train" or "validation" or "test"')
    args = parser.parse_args()

    model_name = args.model
    mode = args.mode
    if mode not in {'train', 'validation', 'test'}:
        raise ValueError('Argument to --mode must be either "train", "validation" or "test"')

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

    config = execution.ExperimentConfiguration.parse_obj(config)

    # Initialize and load model
    model_directory = os.path.expanduser(os.path.normpath(config.models[model_name].location))

    model = execution.initialize_model(config, model_name, device_name)
    execution.load_model(model, model_directory, model_name)

    # Prepare test data
    dataset_directory = os.path.join(os.environ['DATASET_DIRECTORY'], 'processed', mode)

    file_names = list(map(lambda fn: os.path.basename(fn).split('.')[0],
                          execution.load_file_names(dataset_directory)))
    controls, states = execution.load_simulation_data(
        directory=dataset_directory, control_names=config.control_names, state_names=config.state_names)

    simulations = list(zip(controls, states, file_names))

    # Execute predictions on test data
    control = []
    pred_states = []
    true_states = []
    file_names = []
    whiteboxes = []
    blackboxes = []
    for initial_control, initial_state, true_control, true_state, file_name \
            in split_simulations(config.window_size, config.horizon_size, simulations):
        try:
            # Hybrid models may return physical and LSTM output separately
            # If a model supports this output, we would like to record this.
            # If a model does not support it, a TypeError is thrown as we have provided an unknown argument.
            # Then we run the simulate-call without setting return_whitebox_blackbox
            pred_target, whitebox, blackbox = model.simulate(
                initial_control, initial_state, true_control, return_whitebox_blackbox=True
            )
            whiteboxes.append(whitebox)
            blackboxes.append(blackbox)
        except TypeError:
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

    result_file_path = os.path.join(result_directory, f'{mode}-w_{config.window_size}-h_{config.horizon_size}.hdf5')
    with h5py.File(result_file_path, 'w') as f:
        f.attrs['control_names'] = np.array([np.string_(name) for name in config.control_names])
        f.attrs['state_names'] = np.array([np.string_(name) for name in config.state_names])

        f.create_dataset('file_names', data=np.array(list(map(np.string_, file_names))))

        control_grp = f.create_group('control')
        pred_grp = f.create_group('predicted')
        true_grp = f.create_group('true')
        if (len(whiteboxes) > 0) and (len(blackboxes) > 0):
            whitebox_grp = f.create_group('whitebox')
            blackbox_grp = f.create_group('blackbox')

            for i, (control, pred_state, true_state, whitebox, blackbox) \
                    in enumerate(zip(control, pred_states, true_states, whiteboxes, blackboxes)):
                control_grp.create_dataset(str(i), data=control)
                pred_grp.create_dataset(str(i), data=pred_state)
                true_grp.create_dataset(str(i), data=true_state)
                whitebox_grp.create_dataset(str(i), data=whitebox)
                blackbox_grp.create_dataset(str(i), data=blackbox)
        else:
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
