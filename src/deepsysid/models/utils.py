from typing import Any, List, Optional, Tuple, TypeVar


import dataclasses
from typing import List, Optional, Tuple, TypeVar

import numpy as np
import torch
from numpy.typing import NDArray

TensorType = TypeVar('TensorType', torch.Tensor, NDArray[np.float64])


@dataclasses.dataclass
class TrainingPrediction:
    zp: NDArray[np.float64]
    zp_hat: NDArray[np.float64]
    u: NDArray[np.float64]
    y_lin: Optional[NDArray[np.float64]] = None

@dataclasses.dataclass
class XYdata:
    x: NDArray[np.float64]
    y: NDArray[np.float64]
    title:str


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
