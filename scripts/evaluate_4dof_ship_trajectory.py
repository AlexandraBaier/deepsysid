import argparse
import json
import os

import h5py
import numpy as np

import deepsysid.utils as utils


def main():
    parser = argparse.ArgumentParser(description='Evaluate trajectory prediction')
    parser.add_argument('model', help='model')
    parser.add_argument('--mode', action='store', help='either "validation" or "test"')
    args = parser.parse_args()

    model_name = args.model
    mode = args.mode
    if mode not in {'validation', 'test'}:
        raise ValueError('Argument to --mode must be either "validation" or "test"')

    with open(os.environ['CONFIGURATION'], mode='r') as f:
        config = json.load(f)

    window_size = config['window']
    horizon_size = config['horizon']
    state_names = config['state_names']

    test_directory = os.environ['RESULT_DIRECTORY']
    test_file_path = os.path.join(
        test_directory, model_name, f'{mode}-w_{window_size}-h_{horizon_size}.hdf5')
    if mode == 'test':
        scores_file_path = os.path.join(
            test_directory, model_name, f'trajectory-w_{window_size}-h_{horizon_size}.hdf5')
        readable_scores_file_path = os.path.join(
            test_directory, model_name, f'trajectory-w_{window_size}-h_{horizon_size}.json')
    else:
        scores_file_path = os.path.join(
            test_directory, model_name, f'validation_trajectory-w_{window_size}-h_{horizon_size}.hdf5')
        readable_scores_file_path = os.path.join(
            test_directory, model_name, f'validation_trajectory-w_{window_size}-h_{horizon_size}.json')

    pred = []
    true = []
    steps = []

    # Load predicted and true states for each multi-step sequence.
    with h5py.File(test_file_path, 'r') as f:
        file_names = [fn.decode('UTF-8') for fn in f['file_names'][:].tolist()]
        for i in range(len(file_names)):
            pred.append(f['predicted'][str(i)][:])
            true.append(f['true'][str(i)][:])
            steps.append(f['predicted'][str(i)][:].shape[0])

    traj_rmse_per_step_seq = []

    for pred_state, true_state in zip(pred, true):
        pred_x, pred_y, _, _ = utils.compute_trajectory_4dof(pred_state, state_names, config['time_delta'])
        true_x, true_y, _, _ = utils.compute_trajectory_4dof(true_state, state_names, config['time_delta'])

        traj_rmse_per_step = np.sqrt((pred_x - true_x)**2 + (pred_y - true_y)**2)

        traj_rmse_per_step_seq.append(traj_rmse_per_step)

    traj_rmse = np.mean(np.concatenate(traj_rmse_per_step_seq))
    traj_stddev = np.std(np.concatenate(traj_rmse_per_step_seq))
    n_samples = np.concatenate(traj_rmse_per_step_seq).size

    with h5py.File(scores_file_path, 'w') as f:
        f.attrs['state_names'] = np.array(list(map(np.string_, state_names)))
        f.create_dataset('file_names', data=np.array(list(map(np.string_, file_names))))
        f.create_dataset('rmse_mean', data=traj_rmse)
        f.create_dataset('rmse_stddev', data=traj_stddev)
        f.create_dataset('n_samples', data=n_samples)

        rmse_grp = f.create_group('rmse_per_step')
        for idx, traj_rmse_per_step in enumerate(traj_rmse_per_step_seq):
            rmse_grp.create_dataset(str(idx), data=traj_rmse_per_step)

    with open(readable_scores_file_path, mode='w') as f:
        obj = dict()
        obj['rmse_mean'] = traj_rmse
        obj['rmse_stddev'] = traj_stddev
        obj['n_samples'] = n_samples
        json.dump(obj, f)


if __name__ == '__main__':
    main()
