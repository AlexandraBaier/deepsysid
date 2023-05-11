from .base import BaseEventTracker, EventData
import mlflow

class MlFlowTracker(EventTracker):
    def __init__(self, tracking_uri: str) -> None:
        super().__init__()
        mlflow.set_tracking_uri(tracking_uri)


    def __call__(self, event: EventData) -> None:
        if 'metric' in event.data:
            for metric_list in event.data['metric']:
                mlflow.log_metric(metric_list[0], metric_list[1], step=metric_list[2])
        if 'figures' in event.data:
            for figures_list in event.data['figures']:
                mlflow.log_figure(figure=figures_list[0], artifact_file=figures_list[1])

