from typing import Optional

import mlflow

from .base import (
    BaseEventTracker,
    BaseEventTrackerConfig,
    EventData,
    EventReturn,
    EventType,
)


class MlflowConfig(BaseEventTrackerConfig):
    tracking_uri: Optional[str]


class MlFlowTracker(BaseEventTracker):
    CONFIG = MlflowConfig

    def __init__(self, config: MlflowConfig) -> None:
        super().__init__()
        if hasattr(config, 'tracking_uri'):
            mlflow.set_tracking_uri(config.tracking_uri)  # type: ignore
        if config.id:
            run = mlflow.active_run()  # type: ignore
            if hasattr(run, 'info') and not (run.info.run_id == config.id):
                mlflow.start_run(run_id=config.id)  # type: ignore

    def __call__(self, event: EventData) -> EventReturn:
        event_return = EventReturn(dict())
        if event.event_type == EventType.TRACK_PARAMETERS:
            if 'parameters' in event.data:
                for parameter_list in event.data['parameters']:
                    mlflow.log_param(  # type: ignore
                        parameter_list[0], parameter_list[1]
                    )
        elif event.event_type == EventType.TRACK_METRICS:
            if 'metrics' in event.data:
                for metric_list in event.data['metrics']:
                    mlflow.log_metric(  # type: ignore
                        metric_list[0], metric_list[1], step=metric_list[2]
                    )
        elif event.event_type == EventType.TRACK_FIGURES:
            if 'figures' in event.data:
                for figures_list in event.data['figures']:
                    mlflow.log_figure(  # type: ignore
                        figure=figures_list[0],
                        artifact_file=f'figures/{figures_list[1]}',
                    )
        elif event.event_type == EventType.TRACK_ARTIFACTS:
            if 'artifacts' in event.data:
                for artifact_list in event.data['artifacts']:
                    mlflow.log_artifact(  # type: ignore
                        local_path=artifact_list[0], artifact_path=artifact_list[1]
                    )
        elif event.event_type == EventType.GET_ID:
            run = mlflow.active_run()  # type: ignore
            if run is not None:
                event_return = EventReturn({'id': run.info.run_id})
        elif event.event_type == EventType.SET_TAG:
            if 'tags' in event.data:
                for tag_list in event.data['tags']:
                    mlflow.set_tag(tag_list[0], tag_list[1])  # type: ignore
        elif event.event_type == EventType.SET_EXPERIMENT_NAME:
            if 'experiment name' in event.data:
                mlflow.set_experiment(  # type: ignore
                    experiment_name=event.data['experiment name']
                )
        elif event.event_type == EventType.STOP_RUN:
            mlflow.end_run()  # type: ignore
        else:
            raise NotImplementedError(f'{event.event_type} is not implemented.')

        return event_return
