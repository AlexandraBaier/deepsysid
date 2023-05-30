import abc
import dataclasses
from typing import Dict, List, Type, Union, Callable

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from ...models.base import DynamicIdentificationModel
from ...tracker.base import EventData

TestResultMetadata = Dict[str, Union[List[str], List[float], List[int]]]


@dataclasses.dataclass
class TestSimulation:
    control: NDArray[np.float64]
    state: NDArray[np.float64]
    initial_state: NDArray[np.float64]
    file_name: str


@dataclasses.dataclass
class TestSequenceResult:
    inputs: Dict[str, NDArray[np.float64]]
    outputs: Dict[str, NDArray[np.float64]]
    initial_states: Dict[str, NDArray[np.float64]]
    metadata: Dict[str, NDArray[np.float64]]


@dataclasses.dataclass
class TestResult:
    sequences: List[TestSequenceResult]
    metadata: TestResultMetadata


class BaseTestConfig(BaseModel):
    control_names: List[str]
    state_names: List[str]
    window_size: int
    horizon_size: int


class BaseTest(metaclass=abc.ABCMeta):
    CONFIG: Type[BaseTestConfig] = BaseTestConfig

    def __init__(self, config: BaseTestConfig, device_name: str):
        pass

    @abc.abstractmethod
    def test(
        self, model: DynamicIdentificationModel, simulations: List[TestSimulation], tracker: Callable[[EventData], None] = lambda _: None,
    ) -> TestResult:
        pass


def retrieve_test_class(test_class_string: str) -> Type[BaseTest]:
    # https://stackoverflow.com/a/452981
    parts = test_class_string.split('.')
    module_string = '.'.join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, BaseTest):
        raise ValueError(f'{cls} is not a subclass of BaseTest.')
    return cls  # type: ignore
