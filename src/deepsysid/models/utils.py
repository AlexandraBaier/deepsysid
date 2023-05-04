from typing import List, Tuple, TypeVar, Optional

import numpy as np
import torch
from numpy.typing import NDArray
import dataclasses
import matplotlib.pyplot as plt
import matplotlib

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


def plot_input(result: TrainingPrediction) -> matplotlib.figure.Figure:
    seq_len, nu = result.u.shape
    fig, axs = plt.subplots(nrows=nu, ncols=1, tight_layout=True, squeeze=False)
    fig.suptitle('Input plots')
    t = np.linspace(0, seq_len - 1, seq_len)
    for element, ax in zip(range(nu), axs[:, 0]):
        ax.plot(t, result.u[:, element])
        ax.set_title(f'$u_{element+1}$')
        ax.grid()
    return fig


def plot_outputs(result: TrainingPrediction) -> matplotlib.figure.Figure:
    seq_len, ny = result.zp.shape
    fig, axs = plt.subplots(nrows=ny, ncols=1, tight_layout=True, squeeze=False)
    if result.y_lin is None:
        result.y_lin = np.zeros(shape=(seq_len, ny))
    fig.suptitle('Output plots')
    t = np.linspace(0, seq_len - 1, seq_len)
    for element, ax in zip(range(ny), axs[:, 0]):
        ax.plot(t, result.zp[:, element], '--', label=r'$z_p$')
        ax.plot(t, result.zp_hat[:, element], label=r'$\\hat{z}_p$')
        ax.plot(t, result.y_lin[:, element], '--', label=r'$y_{lin}$')
        ax.set_title(f'$z_{element+1}$')
        ax.grid()
        ax.legend()
    return fig
