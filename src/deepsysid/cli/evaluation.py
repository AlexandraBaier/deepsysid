import dataclasses
import json
import os
from typing import Any, Dict, List, Literal, Optional

import h5py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from deepsysid import execution, utils
from deepsysid.cli.testing import build_result_file_name


@dataclasses.dataclass
class TrajectoryResult:
    file_names: List[str]
    rmse_mean: float
    rmse_stddev: float
    n_samples: int
    rmse_per_step: List[np.ndarray]


def evaluate_model(
    config: execution.ExperimentConfiguration,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
    result_directory: str,
    threshold: Optional[float] = None,
):
    test_file_path = os.path.join(
        result_directory,
        model_name,
        build_result_file_name(
            mode=mode,
            window_size=config.window_size,
            horizon_size=config.horizon_size,
            extension='hdf5',
            threshold=threshold,
        ),
    )

    scores_file_path = os.path.join(
        result_directory,
        model_name,
        build_score_file_name(
            mode=mode,
            window_size=config.window_size,
            horizon_size=config.horizon_size,
            extension='hdf5',
            threshold=threshold,
        ),
    )
    readable_scores_file_path = os.path.join(
        result_directory,
        model_name,
        build_score_file_name(
            mode=mode,
            window_size=config.window_size,
            horizon_size=config.horizon_size,
            extension='json',
            threshold=threshold,
        ),
    )

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

    def mse(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return mean_squared_error(t, p, multioutput='raw_values')

    def rmse(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.sqrt(mean_squared_error(t, p, multioutput='raw_values'))

    def rmse_std(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.std(
            np.sqrt(mean_squared_error(t, p, multioutput='raw_values')), axis=0
        )

    def mae(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return mean_absolute_error(t, p, multioutput='raw_values')

    def d1(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return utils.index_of_agreement(t, p, j=1)

    score_functions = (
        ('mse', mse),
        ('rmse', rmse),
        ('rmse-std', rmse_std),
        ('mae', mae),
        ('d1', d1),
    )

    scores = dict()
    for name, fct in score_functions:
        scores[name] = utils.score_on_sequence(true, pred, fct)

    with h5py.File(scores_file_path, 'w') as f:
        f.attrs['state_names'] = np.array(list(map(np.string_, config.state_names)))
        f.create_dataset('file_names', data=np.array(list(map(np.string_, file_names))))
        f.create_dataset('steps', data=np.array(steps))

        for name, _ in score_functions:
            f.create_dataset(name, data=scores[name])

    average_scores = dict()
    for name, _ in score_functions:
        # 1/60 * sum_1^60 RMSE
        average_scores[name] = np.average(scores[name], weights=steps, axis=0).tolist()

    with open(readable_scores_file_path, mode='w') as f:
        obj: Dict[str, Any] = dict()
        obj['scores'] = average_scores
        obj['state_names'] = config.state_names
        json.dump(obj, f)


def test_4dof_ship_trajectory(
    config: execution.ExperimentConfiguration, result_file_path: str
) -> TrajectoryResult:
    pred = []
    true = []
    steps = []

    # Load predicted and true states for each multi-step sequence.
    with h5py.File(result_file_path, 'r') as f:
        file_names = [fn.decode('UTF-8') for fn in f['file_names'][:].tolist()]
        for i in range(len(file_names)):
            pred.append(f['predicted'][str(i)][:])
            true.append(f['true'][str(i)][:])
            steps.append(f['predicted'][str(i)][:].shape[0])

    traj_rmse_per_step_seq = []

    for pred_state, true_state in zip(pred, true):
        pred_x, pred_y, _, _ = utils.compute_trajectory_4dof(
            pred_state, config.state_names, config.time_delta
        )
        true_x, true_y, _, _ = utils.compute_trajectory_4dof(
            true_state, config.state_names, config.time_delta
        )

        traj_rmse_per_step = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)

        traj_rmse_per_step_seq.append(traj_rmse_per_step)

    traj_rmse = float(np.mean(np.concatenate(traj_rmse_per_step_seq)))
    traj_stddev = float(np.std(np.concatenate(traj_rmse_per_step_seq)))
    n_samples = np.concatenate(traj_rmse_per_step_seq).size

    return TrajectoryResult(
        file_names=file_names,
        rmse_mean=traj_rmse,
        rmse_stddev=traj_stddev,
        n_samples=n_samples,
        rmse_per_step=traj_rmse_per_step_seq,
    )


def test_quadcopter_trajectory(
    config: execution.ExperimentConfiguration, result_file_path: str
) -> TrajectoryResult:
    pred = []
    true = []
    steps = []

    # Load predicted and true states for each multi-step sequence.
    with h5py.File(result_file_path, 'r') as f:
        file_names = [fn.decode('UTF-8') for fn in f['file_names'][:].tolist()]
        for i in range(len(file_names)):
            pred.append(f['predicted'][str(i)][:])
            true.append(f['true'][str(i)][:])
            steps.append(f['predicted'][str(i)][:].shape[0])

    traj_rmse_per_step_seq = []

    for pred_state, true_state in zip(pred, true):
        px, py, pz = utils.compute_trajectory_quadcopter(
            pred_state, config.state_names, config.time_delta
        )
        tx, ty, tz = utils.compute_trajectory_quadcopter(
            true_state, config.state_names, config.time_delta
        )

        traj_rmse_per_step = np.sqrt((px - tx) ** 2 + (py - ty) ** 2 + (pz - tz) ** 2)
        traj_rmse_per_step_seq.append(traj_rmse_per_step)

    traj_rmse = float(np.mean(np.concatenate(traj_rmse_per_step_seq)))
    traj_stddev = float(np.std(np.concatenate(traj_rmse_per_step_seq)))
    n_samples = np.concatenate(traj_rmse_per_step_seq).size

    return TrajectoryResult(
        file_names=file_names,
        rmse_mean=traj_rmse,
        rmse_stddev=traj_stddev,
        n_samples=n_samples,
        rmse_per_step=traj_rmse_per_step_seq,
    )


def build_trajectory_file_name(
    mode: Literal['train', 'validation', 'test'],
    window_size: int,
    horizon_size: int,
    extension: str,
) -> str:
    return f'trajectory-{mode}-w_{window_size}-h_{horizon_size}.{extension}'


def save_trajectory_results(
    result: TrajectoryResult,
    config: execution.ExperimentConfiguration,
    result_directory: str,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
):
    scores_file_path = os.path.join(
        result_directory,
        model_name,
        build_trajectory_file_name(
            mode=mode,
            window_size=config.window_size,
            horizon_size=config.horizon_size,
            extension='hdf5',
        ),
    )
    readable_scores_file_path = os.path.join(
        result_directory,
        model_name,
        build_trajectory_file_name(
            mode=mode,
            window_size=config.window_size,
            horizon_size=config.horizon_size,
            extension='json',
        ),
    )

    with h5py.File(scores_file_path, 'w') as f:
        f.attrs['state_names'] = np.array(list(map(np.string_, config.state_names)))
        f.create_dataset(
            'file_names', data=np.array(list(map(np.string_, result.file_names)))
        )
        f.create_dataset('rmse_mean', data=result.rmse_mean)
        f.create_dataset('rmse_stddev', data=result.rmse_stddev)
        f.create_dataset('n_samples', data=result.n_samples)

        rmse_grp = f.create_group('rmse_per_step')
        for idx, traj_rmse_per_step in enumerate(result.rmse_per_step):
            rmse_grp.create_dataset(str(idx), data=traj_rmse_per_step)

    with open(readable_scores_file_path, mode='w') as f:
        obj = dict()
        obj['rmse_mean'] = result.rmse_mean
        obj['rmse_stddev'] = result.rmse_stddev
        obj['n_samples'] = result.n_samples
        json.dump(obj, f)


def build_score_file_name(
    mode: Literal['train', 'validation', 'test'],
    window_size: int,
    horizon_size: int,
    extension: str,
    threshold: Optional[float] = None,
) -> str:
    if threshold is None:
        return f'scores-{mode}-w_{window_size}-h_{horizon_size}.{extension}'

    threshold_str = f'{threshold:.f}'.replace('.', '')
    return (
        f'scores-{mode}-w_{window_size}-h_{horizon_size}-t_{threshold_str}.{extension}'
    )
