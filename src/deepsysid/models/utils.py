import dataclasses
import importlib
from types import ModuleType
from typing import Any, List, Optional, Tuple, TypeVar

import numpy as np
import torch
import os
from numpy.typing import NDArray
from .base import DynamicIdentificationModel, NormalizedHiddenStateInitializerPredictorModel
from ..pipeline.testing import base
from ..pipeline.data_io import load_simulation_data
from ..pipeline.testing.io import split_simulations
from ..cli.interface import DATASET_DIR_ENV_VAR

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


def validate(model:NormalizedHiddenStateInitializerPredictorModel, loss: torch.nn.Module, state_names: List[str], control_names: List[str], initial_state_names: List[str], sequence_length: int, device: torch.device, horizon_size: Optional[int] = None) -> np.float64:
    # load validation data
    if horizon_size is None:
        horizon_size = sequence_length * 5
    # create dataset directory
    dataset_directory = os.path.expanduser(os.environ[DATASET_DIR_ENV_VAR])
    us, ys, x0s = load_simulation_data(
        directory=dataset_directory,
        control_names=control_names,
        state_names=state_names,
        initial_state_names=initial_state_names
    )
    simulations = [
        base.TestSimulation(u, y, x0, '-')
        for u, y, x0 in zip(us, ys, x0s)
    ]

    u_list: List[NDArray[np.float64]] = list()
    y_list: List[NDArray[np.float64]] = list()
    x0_list: List[NDArray[np.float64]] = list()
    u_init_list: List[NDArray[np.float64]] = list()
    y_init_list: List[NDArray[np.float64]] = list()
    for sample in split_simulations(sequence_length, horizon_size, simulations):
        u_init_list.append(sample.initial_control)
        y_init_list.append(sample.initial_state)

        u_list.append(sample.true_control)
        y_list.append(sample.true_state)
        x0_list.append(sample.x0)
    
    
    if model.state_mean and model.state_std and model.control_mean and model.control_std:
        us = normalize(np.vstack(u_list), model.control_mean, model.control_std)
        ys = normalize(np.vstack(y_list), model.state_mean, model.state_std)

        us_init = normalize(np.vstack(u_init_list), model.control_mean, model.control_std)
        ys_init = normalize(np.vstack(u_init_list), model.state_mean, model.state_std)

    uy_init = np.hstack((us_init[:,1:], ys_init[:,:-1]))

    us_torch = torch.from_numpy(us).to(device)
    ys_torch = torch.from_numpy(ys).to(device)
    uy_init_torch = torch.from_numpy(uy_init).to(device)
    x0 = torch.from_numpy(np.vstack(x0_list))

    # evaluate model on data use sequence length x 5 as prediction horizon    
    _, hx = model.initializer.forward(uy_init_torch)
    ys_hat_torch, _ = model.predictor.forward(us_torch, hx=hx)
    # return validation error normalized over all states
    return np.float64(loss.forward(ys_torch, ys_hat_torch).detach().numpy())
    


    

