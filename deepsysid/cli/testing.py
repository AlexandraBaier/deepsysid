import dataclasses
import os
from typing import Iterator, List, Literal, Optional, Tuple

import h5py
import numpy as np

from deepsysid import execution
from deepsysid.models.base import DynamicIdentificationModel
from deepsysid.models.hybrid.bounded_residual import HybridResidualLSTMModel


@dataclasses.dataclass
class ModelTestResult:
    control: List[np.ndarray]
    true_state: List[np.ndarray]
    pred_state: List[np.ndarray]
    file_names: List[str]
    whitebox: List[np.ndarray]
    blackbox: List[np.ndarray]


def load_test_simulations(
    configuration: execution.ExperimentConfiguration,
    model_name: str,
    device_name: str,
    mode: Literal['train', 'validation', 'test'],
    dataset_directory: str,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    # Initialize and load model
    model_directory = os.path.expanduser(
        os.path.normpath(configuration.models[model_name].location)
    )

    model = execution.initialize_model(configuration, model_name, device_name)
    execution.load_model(model, model_directory, model_name)

    # Prepare test data
    dataset_directory = os.path.join(dataset_directory, 'processed', mode)

    file_names = list(
        map(
            lambda fn: os.path.basename(fn).split('.')[0],
            execution.load_file_names(dataset_directory),
        )
    )
    controls, states = execution.load_simulation_data(
        directory=dataset_directory,
        control_names=configuration.control_names,
        state_names=configuration.state_names,
    )

    simulations = list(zip(controls, states, file_names))

    return simulations


def test_model(
    simulations: List[Tuple[np.ndarray, np.ndarray, str]],
    config: execution.ExperimentConfiguration,
    model: DynamicIdentificationModel,
    threshold: Optional[float] = None,
) -> ModelTestResult:
    # Execute predictions on test data
    control = []
    pred_states = []
    true_states = []
    file_names = []
    whiteboxes = []
    blackboxes = []
    for (
        initial_control,
        initial_state,
        true_control,
        true_state,
        file_name,
    ) in split_simulations(config.window_size, config.horizon_size, simulations):
        if isinstance(model, HybridResidualLSTMModel):
            # Hybrid residual models can return physical and LSTM output separately
            # and also support clipping.
            pred_target, whitebox, blackbox = model.simulate_hybrid(
                initial_control,
                initial_state,
                true_control,
                threshold=threshold if threshold is not None else np.inf,
            )
            whiteboxes.append(whitebox)
            blackboxes.append(blackbox)
        else:
            pred_target = model.simulate(initial_control, initial_state, true_control)

        control.append(true_control)
        pred_states.append(pred_target)
        true_states.append(true_state)
        file_names.append(file_name)

    return ModelTestResult(
        control=control,
        true_state=true_states,
        pred_state=pred_states,
        file_names=file_names,
        whitebox=whiteboxes,
        blackbox=blackboxes,
    )


def save_model_tests(
    test_result: ModelTestResult,
    config: execution.ExperimentConfiguration,
    result_directory: str,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
    threshold: Optional[float] = None,
):
    # Save true and predicted time series
    result_directory = os.path.join(result_directory, model_name)
    try:
        os.mkdir(result_directory)
    except FileExistsError:
        pass

    result_file_path = os.path.join(
        result_directory,
        build_result_file_name(
            mode, config.window_size, config.horizon_size, 'hdf5', threshold=threshold
        ),
    )
    with h5py.File(result_file_path, 'w') as f:
        write_test_results_to_hdf5(
            f,
            control_names=config.control_names,
            state_names=config.state_names,
            file_names=test_result.file_names,
            control=test_result.control,
            pred_states=test_result.pred_state,
            true_states=test_result.true_state,
            whiteboxes=test_result.whitebox,
            blackboxes=test_result.blackbox,
        )


def split_simulations(
    window_size: int,
    horizon_size: int,
    simulations: List[Tuple[np.ndarray, np.ndarray, str]],
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]]:
    total_length = window_size + horizon_size
    for control, state, file_name in simulations:
        for i in range(total_length, control.shape[0], total_length):
            initial_control = control[i - total_length : i - total_length + window_size]
            initial_state = state[i - total_length : i - total_length + window_size]
            true_control = control[i - total_length + window_size : i]
            true_state = state[i - total_length + window_size : i]

            yield initial_control, initial_state, true_control, true_state, file_name


def write_test_results_to_hdf5(
    f: h5py.File,
    control_names: List[str],
    state_names: List[str],
    file_names: List[str],
    control: List[np.ndarray],
    pred_states: List[np.ndarray],
    true_states: List[np.ndarray],
    whiteboxes: List[np.ndarray],
    blackboxes: List[np.ndarray],
):
    f.attrs['control_names'] = np.array([np.string_(name) for name in control_names])
    f.attrs['state_names'] = np.array([np.string_(name) for name in state_names])

    f.create_dataset('file_names', data=np.array(list(map(np.string_, file_names))))

    control_grp = f.create_group('control')
    pred_grp = f.create_group('predicted')
    true_grp = f.create_group('true')
    if (len(whiteboxes) > 0) and (len(blackboxes) > 0):
        whitebox_grp = f.create_group('whitebox')
        blackbox_grp = f.create_group('blackbox')

        for i, (true_control, pred_state, true_state, whitebox, blackbox) in enumerate(
            zip(control, pred_states, true_states, whiteboxes, blackboxes)
        ):
            control_grp.create_dataset(str(i), data=true_control)
            pred_grp.create_dataset(str(i), data=pred_state)
            true_grp.create_dataset(str(i), data=true_state)
            whitebox_grp.create_dataset(str(i), data=whitebox)
            blackbox_grp.create_dataset(str(i), data=blackbox)
    else:
        for i, (true_control, pred_state, true_state) in enumerate(
            zip(control, pred_states, true_states)
        ):
            control_grp.create_dataset(str(i), data=true_control)
            pred_grp.create_dataset(str(i), data=pred_state)
            true_grp.create_dataset(str(i), data=true_state)


def build_result_file_name(
    mode: Literal['train', 'validation', 'test'],
    window_size: int,
    horizon_size: int,
    extension: str,
    threshold: Optional[float] = None,
) -> str:
    if threshold is None:
        return f'{mode}-w_{window_size}-h_{horizon_size}.{extension}'

    threshold_str = f'{threshold:.f}'.replace('.', '')
    return (
        f'threshold_hybrid_{mode}-w_{window_size}'
        f'-h_{horizon_size}-t_{threshold_str}.{extension}'
    )
