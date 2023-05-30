import dataclasses
import importlib
from types import ModuleType
from typing import Any, List, Optional, Tuple, TypeVar, Callable

import numpy as np
import torch
from . import base
from ..tracker.base import EventData, EventType
from numpy.typing import NDArray

matplotlib: Optional[ModuleType] = None
plt: Optional[ModuleType] = None
MATPLOTLIB_EXISTS = False
try:
    matplotlib = importlib.import_module('matplotlib')
    plt = importlib.import_module('matplotlib.pyplot')
    MATPLOTLIB_EXISTS = True
except ImportError:
    pass


TensorType = TypeVar('TensorType', torch.Tensor, NDArray[np.float64])


@dataclasses.dataclass
class TrainingPrediction:
    zp: NDArray[np.float64]
    zp_hat: NDArray[np.float64]
    u: NDArray[np.float64]
    y_lin: Optional[NDArray[np.float64]]


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


if MATPLOTLIB_EXISTS:
    # I do not know how to type hint this correctly in
    # case that matplotlib is not installed.
    # The actual return type is matplotlib.figure.Figure.
    def plot_outputs(result: TrainingPrediction) -> Any:
        if matplotlib is None or plt is None:
            raise ImportError(
                'Package matplotlib is required to run this function '
                'and has not been installed.'
            )

        seq_len, ny = result.zp.shape
        fig, axs = plt.subplots(nrows=ny, ncols=1, tight_layout=True, squeeze=False)
        if result.y_lin is None:
            result.y_lin = np.zeros(shape=(seq_len, ny))
        fig.suptitle('Output plots')
        t = np.linspace(0, seq_len - 1, seq_len)
        for element, ax in zip(range(ny), axs[:, 0]):
            ax.plot(t, result.zp[:, element], '--', label=r'$z_p$')
            ax.plot(t, result.zp_hat[:, element], label=r'$\hat{z}_p$')
            ax.plot(t, result.y_lin[:, element], '--', label=r'$y_{lin}$')
            ax.set_title(f'$z_{element+1}$')
            ax.grid()
            ax.legend()
        return fig
