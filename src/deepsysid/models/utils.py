from typing import List, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray
import torch

TensorType = TypeVar('TensorType', torch.Tensor, NDArray[np.float64])


def mean_stddev(
    array_seq: List[NDArray[np.float64]],
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    mean = np.mean(np.vstack(array_seq), axis=0)
    stddev = np.std(np.vstack(array_seq), axis=0)
    return mean, stddev


def denormalize(x: TensorType, mean: TensorType, stddev: TensorType) -> TensorType:
    return x * stddev + mean


def normalize(x: TensorType, mean: TensorType, stddev: TensorType) -> TensorType:
    return (x - mean) / stddev


def sequence_norm(x: torch.Tensor) -> torch.Tensor:
    norm = torch.tensor(0, device=x.device).float()

    for x_k in x:

        x_k = x_k.unsqueeze(0)
        norm += (x_k @ x_k.T).squeeze()
    return norm
