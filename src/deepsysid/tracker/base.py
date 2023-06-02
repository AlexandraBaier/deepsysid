from typing import List, Optional, Type

from .configuration import BaseEventTrackerConfig
from .event_data import EventData


class BaseEventTracker:
    CONFIG = BaseEventTrackerConfig

    def __init__(self, config: Optional[BaseEventTrackerConfig] = None) -> None:
        pass

    def __call__(self, event: EventData) -> None:
        pass


class TrackerAggregator(BaseEventTracker):
    def __init__(self, config: BaseEventTrackerConfig) -> None:
        super().__init__(config)
        self.trackers: List[BaseEventTracker] = []

    def __call__(self, event: EventData) -> None:
        for tracker in self.trackers:
            tracker(event)

    def register_tracker(self, tracker: BaseEventTracker) -> None:
        self.trackers.append(tracker)


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
