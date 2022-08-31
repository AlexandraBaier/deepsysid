import dataclasses
import os
from typing import Dict, List, Literal, Optional

import h5py
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .configuration import ExperimentConfiguration
from .data_io import (
    build_result_file_name,
    build_score_file_name,
    build_trajectory_file_name,
)
from .metrics import (
    compute_trajectory_4dof,
    compute_trajectory_quadcopter,
    index_of_agreement,
    score_on_sequence,
)


class PerHorizonAverageScores(BaseModel):
    mse: List[float]
    rmse: List[float]
    rmse_std: List[float]
    mae: List[float]
    d1: List[float]


class ReadableEvaluationScores(BaseModel):
    state_names: List[str]
    scores_per_horizon: Dict[int, PerHorizonAverageScores]


class PerHorizonTrajectoryScores(BaseModel):
    rmse_mean: float
    rmse_std: float
    n_samples: int


class ReadableTrajectoryScores(BaseModel):
    scores_per_horizon: Dict[int, PerHorizonTrajectoryScores]


@dataclasses.dataclass
class EvaluationResult:
    file_names: List[str]
    steps: List[int]
    scores_per_step: Dict[str, NDArray[np.float64]]
    average_scores: Dict[str, List[float]]
    horizon_size: int


@dataclasses.dataclass
class TrajectoryResult:
    file_names: List[str]
    rmse_mean: float
    rmse_stddev: float
    n_samples: int
    rmse_per_step: List[NDArray[np.float64]]
    horizon_size: int


def evaluate_model(
    config: ExperimentConfiguration,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
    result_directory: str,
    threshold: Optional[float] = None,
) -> None:
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
    readable_evaluation_result = ReadableEvaluationScores(
        state_names=config.state_names,
        scores_per_horizon=dict(
            (
                result.horizon_size,
                PerHorizonAverageScores(
                    mse=result.average_scores['mse'],
                    rmse=result.average_scores['rmse'],
                    rmse_std=result.average_scores['rmse_std'],
                    mae=result.average_scores['mae'],
                    d1=result.average_scores['d1'],
                ),
            )
            for result in results
        ),
    )

    with open(readable_scores_file_path, mode='w') as f:
        f.write(readable_evaluation_result.json())


def evaluate_model_specific_horizon(
    config: ExperimentConfiguration,
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
            pred.append(f['predicted'][str(i)][:horizon_size].astype(np.float64))
            true.append(f['true'][str(i)][:horizon_size].astype(np.float64))
            steps.append(f['predicted'][str(i)][:horizon_size].shape[0])

    def mse(t: NDArray[np.float64], p: NDArray[np.float64]) -> NDArray[np.float64]:
        return mean_squared_error(t, p, multioutput='raw_values')  # type: ignore

    def rmse(t: NDArray[np.float64], p: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.sqrt(  # type: ignore
            mean_squared_error(t, p, multioutput='raw_values')
        )

    def rmse_std(t: NDArray[np.float64], p: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.std(  # type: ignore
            np.sqrt(mean_squared_error(t, p, multioutput='raw_values')), axis=0
        )

    def mae(t: NDArray[np.float64], p: NDArray[np.float64]) -> NDArray[np.float64]:
        return mean_absolute_error(t, p, multioutput='raw_values')  # type: ignore

    def d1(t: NDArray[np.float64], p: NDArray[np.float64]) -> NDArray[np.float64]:
        return index_of_agreement(t, p, j=1)

    score_functions = (
        ('mse', mse),
        ('rmse', rmse),
        ('rmse_std', rmse_std),
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
    configuration: ExperimentConfiguration,
    result_directory: str,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
) -> None:
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
    config: ExperimentConfiguration,
    result_file_path: str,
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
    configuration: ExperimentConfiguration,
    result_directory: str,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
) -> None:
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
    config: ExperimentConfiguration,
    result_file_path: str,
    horizon_size: int,
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
    config: ExperimentConfiguration,
    result_directory: str,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
) -> None:
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
    trajectory_scores = ReadableTrajectoryScores(
        scores_per_horizon=dict(
            (
                result.horizon_size,
                PerHorizonTrajectoryScores(
                    rmse_mean=result.rmse_mean,
                    rmse_std=result.rmse_stddev,
                    n_samples=result.n_samples,
                ),
            )
            for result in results
        )
    )
    with open(readable_scores_file_path, mode='w') as f:
        f.write(trajectory_scores.json())
