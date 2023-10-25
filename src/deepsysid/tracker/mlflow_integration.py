import os
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from scipy.io import savemat

from ..models.utils import TrainingPrediction, XYdata
from ..pipeline.configuration import GridSearchTrackingConfiguration
from ..pipeline.data_io import build_tracker_config_file_name
from .base import BaseEventTracker, BaseEventTrackerConfig
from .event_data import (
    EventData,
    LoadTrackingConfiguration,
    SaveTrackingConfiguration,
    SetExperiment,
    SetTags,
    StopRun,
    TrackArtifacts,
    TrackFigures,
    TrackMetrics,
    TrackParameters,
    TrackSequencesAsMatFile,
)

FIGURE_DIRECTORY_NAME = 'figures'


class MlflowConfig(BaseEventTrackerConfig):
    tracking_uri: Optional[str]


class MlFlowTracker(BaseEventTracker):
    CONFIG = MlflowConfig

    def __init__(self, config: MlflowConfig) -> None:
        super().__init__(config)
        if hasattr(config, 'tracking_uri'):
            mlflow.set_tracking_uri(config.tracking_uri)

    def __call__(self, event: EventData) -> None:
        # print(f'[TRACKER:] \t {event.msg}')
        if isinstance(event, TrackParameters):
            for key, value in event.parameters.items():
                mlflow.log_param(key, value)
        elif isinstance(event, SetExperiment):
            experiment_name = os.path.split(
                pathlib.Path(event.dataset_directory).parent.parent
            )[1]
            mlflow.set_experiment(experiment_name)
        elif isinstance(event, SaveTrackingConfiguration):
            self.save_tracking_configuration(event)
        elif isinstance(event, StopRun):
            mlflow.end_run()
        elif isinstance(event, SetTags):
            for key, value in event.tags.items():
                mlflow.set_tag(key, value)
        elif isinstance(event, LoadTrackingConfiguration):
            self.load_tracking_configuration(event)
        elif isinstance(event, TrackMetrics):
            for key, value in event.metrics.items():
                mlflow.log_metric(key, value, event.step)
        elif isinstance(event, TrackFigures):
            if isinstance(event.results, TrainingPrediction):
                fig = plot_outputs(event.results)
            elif isinstance(event.results, XYdata):
                fig = plot_xydata(event.results)
            mlflow.log_figure(fig, f'{FIGURE_DIRECTORY_NAME}/{event.name}')
            plt.close()
        elif isinstance(event, TrackSequencesAsMatFile):
            self.save_mat_file(event)
        elif isinstance(event, TrackArtifacts):
            for artifact_path, local_path in event.artifacts.items():
                mlflow.log_artifact(local_path, artifact_path)

        else:
            raise NotImplementedError(f'{type(event)} is not implemented.')

    def save_mat_file(self, event: TrackSequencesAsMatFile) -> None:
        savemat(
            event.file_name,
            {'y': np.array(event.sequences[0]), 'y_hat': np.array(event.sequences[1])},
        )
        mlflow.log_artifact(event.file_name, event.artifact_path)

    def save_tracking_configuration(self, event: SaveTrackingConfiguration) -> None:
        run = mlflow.active_run()
        tracker_config_file_name = os.path.join(
                            event.model_directory,
                            build_tracker_config_file_name(event.model_name),
                        )
        # if run is not None and not(os.path.exists(tracker_config_file_name)):
        for tracker_config in event.config.values():
            if (
                tracker_config.tracking_class
                == f'{self.__module__}.{self.__class__.__name__}'
            ):
                tracker_config.parameters.id = run.info.run_id
                with open(tracker_config_file_name,
                    mode='w',
                ) as f:
                    f.write(tracker_config.json())

    def load_tracking_configuration(self, event: LoadTrackingConfiguration) -> None:
        config_file = os.path.join(
            event.model_directory, build_tracker_config_file_name(event.model_name)
        )
        if os.path.exists(config_file):
            raw_config = GridSearchTrackingConfiguration.parse_file(config_file)
            tracker_config = self.CONFIG.parse_obj(raw_config.parameters)
            mlflow.start_run(run_id=tracker_config.id)


def plot_outputs(result: TrainingPrediction) -> plt.Figure:
    seq_len, ny = result.zp.shape
    fig, axs = plt.subplots(nrows=ny, ncols=1, tight_layout=True, squeeze=False)
    if result.y_lin is None:
        result.y_lin = np.zeros(shape=(seq_len, ny))
    fig.suptitle('Output plots')
    t = np.linspace(0, seq_len - 1, seq_len)
    for element, ax in zip(range(ny), axs[:, 0]):
        ax.plot(t, result.zp[:, element], '--', label=r'$z_p$')
        ax.plot(t, result.zp_hat[:, element], label=r'$\hat{z}_p$')
        ax.plot(t, result.y_lin[:, element], '--', label=r'$y_{lin}$')
        ax.set_title(f'$z_{element+1}$')
        ax.set_xlabel('time step')
        ax.grid()
        ax.legend()
    return fig


def plot_xydata(result: XYdata) -> plt.Figure:
    assert result.x.shape[0] == result.y.shape[0]
    fig, ax = plt.subplots()
    fig.suptitle(result.title)
    ax.plot(result.x, result.y)
    ax.grid()

    return fig
