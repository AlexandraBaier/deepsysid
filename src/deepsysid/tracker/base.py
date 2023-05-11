import dataclasses
from typing import Literal, Dict, Any, Type
import abc

@dataclasses.dataclass
class EventData:
    event: Literal['logging']
    data: Dict[str, Any]


class BaseEventTracker(metaclass = abc.ABCMeta):
    @abc.abstractmethod
    def __call__(Event: EventData) -> None:
        pass


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
