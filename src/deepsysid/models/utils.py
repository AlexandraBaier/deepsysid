from typing import List, Tuple, TypeVar

import numpy as np
from torch import Tensor

TensorType = TypeVar('TensorType', Tensor, np.ndarray)


def mean_stddev(array_seq: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    mean = np.mean(np.vstack(array_seq), axis=0)
    stddev = np.std(np.vstack(array_seq), axis=0)
    return mean, stddev


def denormalize(x: TensorType, mean: TensorType, stddev: TensorType) -> TensorType:
    return x * stddev + mean


def normalize(x: TensorType, mean: TensorType, stddev: TensorType) -> TensorType:
    return (x - mean) / stddev
