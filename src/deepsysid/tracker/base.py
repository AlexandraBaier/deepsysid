from __future__ import annotations

# https://peps.python.org/pep-0563/#abstract
import abc
import dataclasses
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union

# to avoid circular imports, only required for type checking
if TYPE_CHECKING:
    from ..pipeline.configuration import ExperimentTrackingConfiguration

from pydantic import BaseModel

from ..models.utils import TrainingPrediction


@dataclasses.dataclass
class EventData:
    msg: str


@dataclasses.dataclass
class StopRun(EventData):
    run_status: Optional[str]


@dataclasses.dataclass
class EventReturn:
    data: Dict[str, Any]


@dataclasses.dataclass
class TrackParameters(EventData):
    parameters: Dict[str, Union[str, float, int]]


@dataclasses.dataclass
class TrackFigures(EventData):
    results: TrainingPrediction
    name: str


@dataclasses.dataclass
class TrackArtifacts(EventData):
    artifacts: Dict[str, str]


@dataclasses.dataclass
class SetExperiment(EventData):
    dataset_directory: str


@dataclasses.dataclass
class SaveTrackingConfiguration(EventData):
    config: Dict[str, ExperimentTrackingConfiguration]
    model_name: str
    model_directory: str


@dataclasses.dataclass
class LoadTrackingConfiguration(EventData):
    model_directory: str
    model_name: str


@dataclasses.dataclass
class SetTags(EventData):
    tags: Dict[str, Union[str, bool]]


@dataclasses.dataclass
class TrackMetrics(EventData):
    metrics: Dict[str, float]


class BaseEventTrackerConfig(BaseModel):
    id: Optional[str]


class BaseEventTracker(metaclass=abc.ABCMeta):
    CONFIG = BaseEventTrackerConfig

    @abc.abstractmethod
    def __call__(self, Event: EventData) -> None:
        pass


class TrackerAggregator(BaseEventTracker):
    def __init__(self, trackers: List[BaseEventTracker]) -> None:
        super().__init__()
        self.trackers = trackers

    def __call__(self, event: EventData) -> None:
        for tracker in self.trackers:
            # print(f'[TRACKER] \t {event.msg}')
            tracker(event)


def retrieve_tracker_class(
    tracker_class_string: str,
) -> Type[BaseEventTracker]:
    # https://stackoverflow.com/a/452981
    parts = tracker_class_string.split('.')
    module_string = '.'.join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, BaseEventTracker):
        raise ValueError(f'{cls} is not a subclass of BaseEventTracker.')
    return cls  # type: ignore
