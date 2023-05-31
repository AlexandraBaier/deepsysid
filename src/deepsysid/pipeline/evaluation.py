import os
from typing import Dict, List, Literal, Tuple

import h5py
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from .configuration import ExperimentConfiguration, GridSearchTrackingConfiguration
from .data_io import (
    build_result_file_name,
    build_score_file_name,
    build_tracker_config_file_name,
)
from .metrics import retrieve_metric_class
from ..tracker.base import (
    BaseEventTracker,
    retrieve_tracker_class,
    TrackerAggregator,
    EventData,
    EventType,
)


class ReadableEvaluationScores(BaseModel):
    state_names: List[str]
    scores_per_horizon: Dict[int, Dict[str, List[float]]]


def evaluate_model(
    config: ExperimentConfiguration,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
    result_directory: str,
    models_directory: str,
) -> None:
    # Load from the maximum horizon file.
    test_file_path = os.path.join(
        result_directory,
        model_name,
        build_result_file_name(
            mode=mode,
            window_size=config.window_size,
            horizon_size=config.horizon_size,
            extension='hdf5',
        ),
    )
    model_directory = os.path.expanduser(
        os.path.normpath(os.path.join(models_directory, model_name))
    )

    # set tracking
    def tracker_fcn(_):
        return None

    tracker = tracker_fcn
    if config.tracker is not None:
        trackers: List[BaseEventTracker] = list()
        for tracker_name, tracker_config in config.tracker.items():
            tracker_class = retrieve_tracker_class(tracker_config.tracking_class)
            tracker_config_file = os.path.join(
                model_directory,
                build_tracker_config_file_name(tracker_name, model_name),
            )
            if os.path.exists(tracker_config_file):
                raw_config = GridSearchTrackingConfiguration.parse_file(
                    tracker_config_file
                )
                tracker_config.parameters = tracker_class.CONFIG.parse_obj(
                    raw_config.parameters
                )
            trackers.append(tracker_class(tracker_config.parameters))
        tracker = TrackerAggregator(trackers)
        tracker(EventData(EventType.SET_TAG, {'tags': [('mode', mode)]}))

    # Load predicted and true states for each multi-step sequence.
    pred = []
    true = []

    with h5py.File(test_file_path, 'r') as f:
        file_names = [
            fn.decode('UTF-8') for fn in f['main']['metadata']['file_names'][:].tolist()
        ]
        for i in range(len(file_names)):
            pred.append(
                f['main'][str(i)]['outputs']['pred_state'][:].astype(np.float64)
            )
            true.append(
                f['main'][str(i)]['outputs']['true_state'][:].astype(np.float64)
            )

    # Load metrics.
    metrics = dict()
    for name, metric_config in config.metrics.items():
        metric_class = retrieve_metric_class(metric_config.metric_class)
        metrics[name] = metric_class(metric_config.parameters)

    results: Dict[
        int, Dict[str, Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]]
    ] = dict()
    for horizon_size in range(1, config.horizon_size + 1):
        results[horizon_size] = dict()
        for name, metric in metrics.items():
            score, meta = metric.measure(
                y_true=[t[:horizon_size] for t in true],
                y_pred=[p[:horizon_size] for p in pred],
            )
            results[horizon_size][name] = (score, meta)
    # print(results)
    tracker(
        EventData(
            EventType.TRACK_METRICS,
            {
                'metrics': [
                    (name, float(np.mean(value[0])), 0)
                    for name, value in results[horizon_size].items()
                ]
            },
        )
    )

    scores_file_path = os.path.join(
        result_directory,
        model_name,
        build_score_file_name(
            mode=mode,
            window_size=config.window_size,
            horizon_size=config.horizon_size,
            extension='hdf5',
        ),
    )
    with h5py.File(scores_file_path, 'w') as f:
        f.attrs['state_names'] = np.array(list(map(np.string_, config.state_names)))

        for horizon_size, result in results.items():
            horizon_grp = f.create_group(f'horizon-{horizon_size}')

            for name, (score, meta) in result.items():
                metric_grp = horizon_grp.create_group(name)
                metric_grp.create_dataset('score', data=score)
                meta_grp = metric_grp.create_group('meta')
                for meta_name, value in meta.items():
                    meta_grp.create_dataset(meta_name, data=value)

    readable_scores_file_path = os.path.join(
        result_directory,
        model_name,
        build_score_file_name(
            mode=mode,
            window_size=config.window_size,
            horizon_size=config.horizon_size,
            extension='json',
        ),
    )
    readable_evaluation_result = ReadableEvaluationScores(
        state_names=config.state_names,
        scores_per_horizon=dict(
            (
                horizon_size,
                dict(
                    (name, list(score.tolist())) for name, (score, _) in result.items()
                ),
            )
            for horizon_size, result in results.items()
        ),
    )

    with open(readable_scores_file_path, mode='w') as f:
        f.write(readable_evaluation_result.json())
    tracker(
        EventData(
            EventType.TRACK_ARTIFACTS,
            {'artifacts': [(readable_scores_file_path, 'scores')]},
        )
    )
    tracker(EventData(EventType.SET_TAG, {'tags': [('validated', 1)]}))
    tracker(EventData(EventType.STOP_RUN, {}))
