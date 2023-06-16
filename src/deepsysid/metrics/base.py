import abc
from typing import Callable, Dict, List, Tuple, Type, TypeVar

import numpy as np
from numpy._typing import NDArray
from pydantic import BaseModel


class BaseMetricConfig(BaseModel):
    state_names: List[str]
    sample_time: float


class BaseMetric(metaclass=abc.ABCMeta):
    CONFIG: Type[BaseMetricConfig] = BaseMetricConfig

    def __init__(self, config: BaseMetricConfig):
        pass

    @abc.abstractmethod
    def measure(
        self, y_true: List[NDArray[np.float64]], y_pred: List[NDArray[np.float64]]
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        """
        :param y_true: List of NDArrays of shape (time, state).
        :param y_pred: List of NDArrays of shape (time, state).
        :return: Tuple of the (primary) metric with shape (state,)
            or (1,) and supporting data. For example, mean error
            per state over all time steps as first element while
            the dictionary contains standard deviation  of the
            error per state variable and the mean error per time step.
        """
        pass


def retrieve_metric_class(metric_class_string: str) -> Type[BaseMetric]:
    # https://stackoverflow.com/a/452981
    parts = metric_class_string.split('.')
    module_string = '.'.join(parts[:-1])
    module = __import__(module_string)

    cls = getattr(module, parts[1])
    if len(parts) > 2:
        for component in parts[2:]:
            cls = getattr(cls, component)

    if not issubclass(cls, BaseMetric):
        raise ValueError(f'{cls} is not a subclass of BaseMetric.')
    return cls  # type: ignore


Self = TypeVar('Self', bound=BaseMetric)
MeasureMethod = Callable[
    [Self, List[NDArray[np.float64]], List[NDArray[np.float64]]],
    Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]],
]


def validate_measure_arguments(measure: MeasureMethod) -> MeasureMethod:
    def validated_measure(
        metric: BaseMetric,
        y_true: List[NDArray[np.float64]],
        y_pred: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        if len(y_true) != len(y_pred):
            raise ValueError(
                'Lists y_true and y_pred and steps need to have the same length.'
            )
        if not all(t.shape == p.shape for t, p in zip(y_true, y_pred)):
            raise ValueError(
                'Shapes of pairwise elements in y_true and y_pred have to match.'
            )
        if not all(
            len(t.shape) == 2 and len(p.shape) == 2 for t, p in zip(y_true, y_pred)
        ):
            raise ValueError('y_true and y_pred have to be 2-dimensional arrays.')

        return measure(metric, y_true, y_pred)

    return validated_measure
