from .base import BaseEventTracker, EventData, BaseEventTrackerConfig
import mlflow
from typing import Optional


class MlflowConfig(BaseEventTrackerConfig):
    tracking_uri: Optional[str]


class MlFlowTracker(BaseEventTracker):
    CONFIG = MlflowConfig

    def __init__(self, config: MlflowConfig) -> None:
        super().__init__(config)
        if config.tracking_uri:
            mlflow.set_tracking_uri(config.tracking_uri)

    def __call__(self, event: EventData) -> None:
        if 'metric' in event.data:
            for metric_list in event.data['metric']:
                mlflow.log_metric(metric_list[0], metric_list[1], step=metric_list[2])
        if 'figures' in event.data:
            for figures_list in event.data['figures']:
                mlflow.log_figure(figure=figures_list[0], artifact_file=figures_list[1])
