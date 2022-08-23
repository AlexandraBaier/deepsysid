import dataclasses
import json
import os
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple

import h5py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from deepsysid import execution
from deepsysid.pipeline.testing import build_result_file_name


@dataclasses.dataclass
class EvaluationResult:
    file_names: List[str]
    steps: List[int]
    scores_per_step: Dict[str, np.ndarray]
    average_scores: Dict[str, np.ndarray]
    horizon_size: int


@dataclasses.dataclass
class TrajectoryResult:
    file_names: List[str]
    rmse_mean: float
    rmse_stddev: float
    n_samples: int
    rmse_per_step: List[np.ndarray]
    horizon_size: int


def evaluate_model(
    config: execution.ExperimentConfiguration,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
    result_directory: str,
    threshold: Optional[float] = None,
):
    results = []
    for horizon_size in range(1, config.horizon_size + 1):
        result = evaluate_model_specific_horizon(
            config=config,
            model_name=model_name,
            mode=mode,
            result_directory=result_directory,
            horizon_size=horizon_size,
            threshold=threshold,
        )
        results.append(result)

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
    with h5py.File(scores_file_path, 'w') as f:
        f.attrs['state_names'] = np.array(list(map(np.string_, config.state_names)))

        for result in results:
            horizon_grp = f.create_group(f'horizon-{result.horizon_size}')

            horizon_grp.create_dataset(
                'file_names', data=np.array(list(map(np.string_, result.file_names)))
            )
            horizon_grp.create_dataset('steps', data=np.array(result.steps))

            for name, score in result.scores_per_step.items():
                horizon_grp.create_dataset(name, data=score)

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
    obj: Dict[str, Any] = dict()
    obj['state_names'] = config.state_names
    for result in results:
        result_obj = dict()
        result_obj['scores'] = result.average_scores
        obj[f'horizon-{result.horizon_size}'] = result_obj
    with open(readable_scores_file_path, mode='w') as f:
        json.dump(obj, f)


def evaluate_model_specific_horizon(
    config: execution.ExperimentConfiguration,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
    result_directory: str,
    horizon_size: int,
    threshold: Optional[float] = None,
) -> EvaluationResult:
    # Load from the maximum horizon file.
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

    pred = []
    true = []
    steps = []

    # Load predicted and true states for each multi-step sequence.
    with h5py.File(test_file_path, 'r') as f:
        file_names = [fn.decode('UTF-8') for fn in f['file_names'][:].tolist()]
        for i in range(len(file_names)):
            # Only grab the first "horizon_size" predictions for evaluation.
            pred.append(f['predicted'][str(i)][:horizon_size])
            true.append(f['true'][str(i)][:horizon_size])
            steps.append(f['predicted'][str(i)][:horizon_size].shape[0])

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
        return index_of_agreement(t, p, j=1)

    score_functions = (
        ('mse', mse),
        ('rmse', rmse),
        ('rmse-std', rmse_std),
        ('mae', mae),
        ('d1', d1),
    )

    scores = dict()
    for name, fct in score_functions:
        scores[name] = score_on_sequence(true, pred, fct)

    average_scores = dict()
    for name, _ in score_functions:
        # 1/60 * sum_1^60 RMSE
        average_scores[name] = np.average(scores[name], weights=steps, axis=0).tolist()

    return EvaluationResult(
        file_names=file_names,
        steps=steps,
        scores_per_step=scores,
        average_scores=average_scores,
        horizon_size=horizon_size,
    )


def evaluate_4dof_ship_trajectory(
    configuration: execution.ExperimentConfiguration,
    result_directory: str,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
):
    result_file_path = os.path.join(
        result_directory,
        model_name,
        build_result_file_name(
            mode=mode,
            window_size=configuration.window_size,
            horizon_size=configuration.horizon_size,
            extension='hdf5',
        ),
    )

    results = test_4dof_ship_trajectory(
        config=configuration,
        result_file_path=result_file_path,
    )

    save_trajectory_results(
        results=results,
        config=configuration,
        result_directory=result_directory,
        model_name=model_name,
        mode=mode,
    )


def test_4dof_ship_trajectory(
    config: execution.ExperimentConfiguration, result_file_path: str
) -> List[TrajectoryResult]:
    pred = []
    true = []
    steps = []

    # Load predicted and true states for each multi-step sequence.
    with h5py.File(result_file_path, 'r') as f:
        file_names = [fn.decode('UTF-8') for fn in f['file_names'][:].tolist()]
        for i in range(len(file_names)):
            # Only load first horizon_size predictions.
            pred.append(f['predicted'][str(i)][:])
            true.append(f['true'][str(i)][:])
            steps.append(f['predicted'][str(i)][:].shape[0])

    traj_rmse_per_step_seq = []
    for pred_state, true_state in zip(pred, true):
        pred_x, pred_y, _, _ = compute_trajectory_4dof(
            pred_state, config.state_names, config.time_delta
        )
        true_x, true_y, _, _ = compute_trajectory_4dof(
            true_state, config.state_names, config.time_delta
        )
        traj_rmse_per_step = np.sqrt((pred_x - true_x) ** 2 + (pred_y - true_y) ** 2)
        traj_rmse_per_step_seq.append(traj_rmse_per_step)

    results = []
    for horizon_size in range(1, config.horizon_size + 1):
        seq = traj_rmse_per_step_seq[:horizon_size]
        traj_rmse = float(np.mean(np.concatenate(seq)))
        traj_stddev = float(np.std(np.concatenate(seq)))
        n_samples = np.concatenate(seq).size

        result = TrajectoryResult(
            file_names=file_names,
            rmse_mean=traj_rmse,
            rmse_stddev=traj_stddev,
            n_samples=n_samples,
            rmse_per_step=seq,
            horizon_size=horizon_size,
        )
        results.append(result)

    return results


def evaluate_quadcopter_trajectory(
    configuration: execution.ExperimentConfiguration,
    result_directory: str,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
):
    result_file_path = os.path.join(
        result_directory,
        model_name,
        build_result_file_name(
            mode=mode,
            window_size=configuration.window_size,
            horizon_size=configuration.horizon_size,
            extension='hdf5',
        ),
    )

    results = []
    for horizon_size in range(1, configuration.horizon_size + 1):
        result = test_quadcopter_trajectory(
            config=configuration,
            result_file_path=result_file_path,
            horizon_size=horizon_size,
        )
        results.append(result)

    save_trajectory_results(
        results=results,
        config=configuration,
        result_directory=result_directory,
        model_name=model_name,
        mode=mode,
    )


def test_quadcopter_trajectory(
    config: execution.ExperimentConfiguration, result_file_path: str, horizon_size: int
) -> TrajectoryResult:
    pred = []
    true = []
    steps = []

    # Load predicted and true states for each multi-step sequence.
    with h5py.File(result_file_path, 'r') as f:
        file_names = [fn.decode('UTF-8') for fn in f['file_names'][:].tolist()]
        for i in range(len(file_names)):
            pred.append(f['predicted'][str(i)][:horizon_size])
            true.append(f['true'][str(i)][:horizon_size])
            steps.append(f['predicted'][str(i)][:horizon_size].shape[0])

    traj_rmse_per_step_seq = []

    for pred_state, true_state in zip(pred, true):
        px, py, pz = compute_trajectory_quadcopter(
            pred_state, config.state_names, config.time_delta
        )
        tx, ty, tz = compute_trajectory_quadcopter(
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
        horizon_size=horizon_size,
    )


def save_trajectory_results(
    results: List[TrajectoryResult],
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
    with h5py.File(scores_file_path, 'w') as f:
        f.attrs['state_names'] = np.array(list(map(np.string_, config.state_names)))

        for result in results:
            horizon_grp = f.create_group(f'horizon-{result.horizon_size}')
            horizon_grp.create_dataset(
                'file_names', data=np.array(list(map(np.string_, result.file_names)))
            )
            horizon_grp.create_dataset('rmse_mean', data=result.rmse_mean)
            horizon_grp.create_dataset('rmse_stddev', data=result.rmse_stddev)
            horizon_grp.create_dataset('n_samples', data=result.n_samples)

            rmse_grp = horizon_grp.create_group('rmse_per_step')
            for idx, traj_rmse_per_step in enumerate(result.rmse_per_step):
                rmse_grp.create_dataset(str(idx), data=traj_rmse_per_step)

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
    obj = dict()
    for result in results:
        result_obj = dict()
        result_obj['rmse_mean'] = result.rmse_mean
        result_obj['rmse_stddev'] = result.rmse_stddev
        result_obj['n_samples'] = result.n_samples
        obj[f'horizon-{result.horizon_size}'] = result_obj
    with open(readable_scores_file_path, mode='w') as f:
        json.dump(obj, f)


def build_trajectory_file_name(
    mode: Literal['train', 'validation', 'test'],
    window_size: int,
    horizon_size: int,
    extension: str,
) -> str:
    return f'trajectory-{mode}-w_{window_size}-h_{horizon_size}.{extension}'


def build_score_file_name(
    mode: Literal['train', 'validation', 'test'],
    window_size: int,
    horizon_size: int,
    extension: str,
    threshold: Optional[float] = None,
) -> str:
    if threshold is None:
        return f'scores-{mode}-w_{window_size}-h_{horizon_size}.{extension}'

    threshold_str = f'{threshold:f}'.replace('.', '')
    return (
        f'scores-{mode}-w_{window_size}-h_{horizon_size}-t_{threshold_str}.{extension}'
    )


def index_of_agreement(true: np.ndarray, pred: np.ndarray, j: int = 1) -> np.ndarray:
    if true.shape != pred.shape:
        raise ValueError('true and pred need to have the same shape.')

    if true.size == 0:
        raise ValueError('true and pred cannot be empty.')

    if j < 1:
        raise ValueError('exponent j must be equal or greater than 1.')

    error_sum = np.sum(np.power(np.abs(true - pred), j), axis=0)
    partial_diff_true = np.abs(true - np.mean(true, axis=0))
    partial_diff_pred = np.abs(pred - np.mean(true, axis=0))
    partial_diff_sum = np.sum(
        np.power(partial_diff_true + partial_diff_pred, j), axis=0
    )
    return 1 - (error_sum / partial_diff_sum)


def score_on_sequence(
    true_seq: List[np.ndarray],
    pred_seq: List[np.ndarray],
    score_fnc: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> np.ndarray:
    score = np.zeros((len(pred_seq), pred_seq[0].shape[1]))
    for i, (true, pred) in enumerate(zip(true_seq, pred_seq)):
        score[i, :] = score_fnc(true, pred)
    return score


def euler_method(ic, dx, dt):
    return ic + dx * dt


def two_step_adam_bashford(ic, dx0, dx1, dt):
    return ic + 1.5 * dt * dx1 - 0.5 * dt * dx0


def compute_trajectory_4dof(
    state: np.ndarray,
    state_names: List[str],
    sample_time: float,
    x0=0.0,
    y0=0.0,
    psi0=0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    name2idx = dict((name, idx) for idx, name in enumerate(state_names))

    u = state[:, name2idx['u']]
    v = state[:, name2idx['v']]
    r = state[:, name2idx['r']]
    phi = state[:, name2idx['phi']]

    shape = u.shape

    psi = np.zeros(shape)
    psi[0] = psi0
    x = np.zeros(shape)
    x[0] = x0
    y = np.zeros(shape)
    y[0] = y0
    old_xd = None
    old_yd = None

    for i in range(1, shape[0]):
        if i == 1:
            psi[i] = euler_method(psi[i - 1], r[i], sample_time)
        else:
            psi[i] = two_step_adam_bashford(psi[i - 1], r[i - 1], r[i], sample_time)

        xd = np.cos(psi[i]) * u[i] - (np.sin(psi[i]) * np.cos(phi[i]) * v[i])
        yd = np.sin(psi[i]) * u[i] + (np.cos(psi[i]) * np.cos(phi[i]) * v[i])

        if i == 1:
            x[i] = euler_method(x[i - 1], xd, sample_time)
            y[i] = euler_method(y[i - 1], yd, sample_time)
        else:
            x[i] = two_step_adam_bashford(x[i - 1], old_xd, xd, sample_time)
            y[i] = two_step_adam_bashford(y[i - 1], old_yd, yd, sample_time)

        old_xd = xd
        old_yd = yd

    return x, y, phi, psi


def compute_trajectory_quadcopter(
    state: np.ndarray, state_names: List[str], sample_time: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    name2idx = dict((name, idx) for idx, name in enumerate(state_names))

    xdot = state[:, name2idx['vx']]
    ydot = state[:, name2idx['vy']]
    zdot = state[:, name2idx['vz']]

    shape = xdot.shape

    x = np.zeros(shape)
    y = np.zeros(shape)
    z = np.zeros(shape)
    x[0] = 0.0
    y[0] = 0.0
    z[0] = 0.0

    for i in range(1, shape[0]):
        x[i] = x[i - 1] + sample_time * xdot[i]
        y[i] = y[i - 1] + sample_time * ydot[i]
        z[i] = z[i - 1] + sample_time * zdot[i]

    return x, y, z
