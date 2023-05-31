import dataclasses
from typing import Dict, Any, Type, List, Union, Optional
from enum import Enum
import abc
from pydantic import BaseModel


class EventType(Enum):
    TRACK_PARAMETERS = 1
    TRACK_METRICS = 2
    TRACK_FIGURES = 3
    TRACK_ARTIFACTS = 4
    LOG_TEXT = 5
    GET_ID = 6
    SET_TAG = 7
    SET_EXPERIMENT_NAME = 8
    STOP_RUN = 9


@dataclasses.dataclass
class EventData:
    event_type: EventType
    data: Dict[str, Any]


@dataclasses.dataclass
class EventReturn:
    data: Dict[str, Any]


class BaseEventTrackerConfig(BaseModel):
    id: Optional[str]


class BaseEventTracker(metaclass=abc.ABCMeta):
    CONFIG = BaseEventTrackerConfig

    @abc.abstractmethod
    def __call__(self, Event: EventData) -> Union[EventReturn, List[EventReturn]]:
        pass


class TrackerAggregator(BaseEventTracker):
    def __init__(self, trackers: List[BaseEventTracker]) -> None:
        super().__init__()
        self.trackers = trackers

    def __call__(self, Event: EventData) -> List[EventReturn]:
        event_returns: List[EventReturn] = list()
        for tracker in self.trackers:
            return_event = tracker(Event)
            if not isinstance(return_event, List):
                event_returns.append(return_event)
        return event_returns


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
