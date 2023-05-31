import os
import pathlib
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np

from ..models.utils import TrainingPrediction
from ..pipeline.configuration import GridSearchTrackingConfiguration
from ..pipeline.data_io import build_tracker_config_file_name
from . import base

FIGURE_DIRECTORY_NAME = 'figures'


class MlflowConfig(base.BaseEventTrackerConfig):
    tracking_uri: Optional[str]


class MlFlowTracker(base.BaseEventTracker):
    CONFIG = MlflowConfig

    def __init__(self, config: MlflowConfig) -> None:
        super().__init__()
        if hasattr(config, 'tracking_uri'):
            mlflow.set_tracking_uri(config.tracking_uri)

    def __call__(self, event: base.EventData) -> None:
        if isinstance(event, base.TrackParameters):
            for key, value in event.parameters.items():
                mlflow.log_param(key, value)
        elif isinstance(event, base.SetExperiment):
            experiment_name = os.path.split(
                pathlib.Path(event.dataset_directory).parent.parent
            )[1]
            mlflow.set_experiment(experiment_name)
        elif isinstance(event, base.SaveTrackingConfiguration):
            self.save_tracking_configuration(event)
        elif isinstance(event, base.StopRun):
            mlflow.end_run()
        elif isinstance(event, base.SetTags):
            for key, value in event.tags.items():
                mlflow.set_tag(key, value)
        elif isinstance(event, base.LoadTrackingConfiguration):
            self.load_tracking_configuration(event)
        elif isinstance(event, base.TrackMetrics):
            for key, value in event.metrics.items():
                mlflow.log_metric(key, value)
        elif isinstance(event, base.TrackFigures):
            mlflow.log_figure(
                plot_outputs(event.results), f'{FIGURE_DIRECTORY_NAME}/{event.name}'
            )
        elif isinstance(event, base.TrackArtifacts):
            for artifact_path, local_path in event.artifacts.items():
                mlflow.log_artifact(local_path, artifact_path)

        else:
            raise NotImplementedError(f'{type(event)} is not implemented.')

    def save_tracking_configuration(
        self, event: base.SaveTrackingConfiguration
    ) -> None:
        run = mlflow.active_run()
        if run is not None:
            for tracker_config in event.config.values():
                if (
                    tracker_config.tracking_class
                    == f'{self.__module__}.{self.__class__.__name__}'
                ):
                    tracker_config.parameters.id = run.info.run_id
                    with open(
                        os.path.join(
                            event.model_directory,
                            build_tracker_config_file_name(event.model_name),
                        ),
                        mode='w',
                    ) as f:
                        f.write(tracker_config.json())

    def load_tracking_configuration(
        self, event: base.LoadTrackingConfiguration
    ) -> None:
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
        ax.grid()
        ax.legend()
    return fig
