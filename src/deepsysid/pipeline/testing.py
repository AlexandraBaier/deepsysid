import dataclasses
import os
from typing import Dict, Iterator, List, Literal, Optional, Tuple, Union

import h5py
import numpy as np

from ..models.base import DynamicIdentificationModel
from ..models.hybrid.bounded_residual import HybridResidualLSTMModel
from ..pipeline.configuration import ExperimentConfiguration, initialize_model
from .data_io import build_result_file_name, load_file_names, load_simulation_data
from .model_io import load_model


@dataclasses.dataclass
class ModelTestResult:
    control: List[np.ndarray]
    true_state: List[np.ndarray]
    pred_state: List[np.ndarray]
    file_names: List[str]
    metadata: List[Dict[str, np.ndarray]]


def test_model(
    configuration: ExperimentConfiguration,
    model_name: str,
    device_name: str,
    mode: Literal['train', 'validation', 'test'],
    dataset_directory: str,
    result_directory: str,
    models_directory: str,
):
    model_directory = os.path.expanduser(
        os.path.normpath(os.path.join(models_directory, model_name))
    )
    model = initialize_model(configuration, model_name, device_name)
    load_model(model, model_directory, model_name)

    simulations = load_test_simulations(
        configuration=configuration,
        mode=mode,
        dataset_directory=dataset_directory,
    )

    test_result = simulate_model(
        simulations=simulations, config=configuration, model=model
    )

    save_model_tests(
        test_result=test_result,
        config=configuration,
        result_directory=result_directory,
        model_name=model_name,
        mode=mode,
    )

    if configuration.thresholds and isinstance(model, HybridResidualLSTMModel):
        for threshold in configuration.thresholds:
            test_result = simulate_model(
                simulations=simulations,
                config=configuration,
                model=model,
                threshold=threshold,
            )
            save_model_tests(
                test_result=test_result,
                config=configuration,
                result_directory=result_directory,
                model_name=model_name,
                mode=mode,
                threshold=threshold,
            )


def load_test_simulations(
    configuration: ExperimentConfiguration,
    mode: Literal['train', 'validation', 'test'],
    dataset_directory: str,
) -> List[Tuple[np.ndarray, np.ndarray, str]]:
    # Prepare test data
    dataset_directory = os.path.join(dataset_directory, 'processed', mode)

    file_names = list(
        map(
            lambda fn: os.path.basename(fn).split('.')[0],
            load_file_names(dataset_directory),
        )
    )
    controls, states = load_simulation_data(
        directory=dataset_directory,
        control_names=configuration.control_names,
        state_names=configuration.state_names,
    )

    simulations = list(zip(controls, states, file_names))

    return simulations


def simulate_model(
    simulations: List[Tuple[np.ndarray, np.ndarray, str]],
    config: ExperimentConfiguration,
    model: DynamicIdentificationModel,
    threshold: Optional[float] = None,
) -> ModelTestResult:
    # Execute predictions on test data
    control = []
    pred_states = []
    true_states = []
    file_names = []
    metadata = []
    for (
        initial_control,
        initial_state,
        true_control,
        true_state,
        file_name,
    ) in split_simulations(config.window_size, config.horizon_size, simulations):
        if isinstance(model, HybridResidualLSTMModel):
            simulation_result = model.simulate(
                initial_control,
                initial_state,
                true_control,
                threshold=threshold if threshold is not None else np.infty,
            )  # type: Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]
        else:
            simulation_result = model.simulate(
                initial_control, initial_state, true_control
            )

        if isinstance(simulation_result, np.ndarray):
            pred_target = simulation_result
        else:
            pred_target = simulation_result[0]
            metadata.append(simulation_result[1])

        control.append(true_control)
        pred_states.append(pred_target)
        true_states.append(true_state)
        file_names.append(file_name)

    return ModelTestResult(
        control=control,
        true_state=true_states,
        pred_state=pred_states,
        file_names=file_names,
        metadata=metadata,
    )


def save_model_tests(
    test_result: ModelTestResult,
    config: ExperimentConfiguration,
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
            metadata=test_result.metadata,
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
    metadata: List[Dict[str, np.ndarray]],
):
    f.attrs['control_names'] = np.array([np.string_(name) for name in control_names])
    f.attrs['state_names'] = np.array([np.string_(name) for name in state_names])

    f.create_dataset('file_names', data=np.array(list(map(np.string_, file_names))))

    control_grp = f.create_group('control')
    pred_grp = f.create_group('predicted')
    true_grp = f.create_group('true')

    for i, (true_control, pred_state, true_state) in enumerate(
        zip(control, pred_states, true_states)
    ):
        control_grp.create_dataset(str(i), data=true_control)
        pred_grp.create_dataset(str(i), data=pred_state)
        true_grp.create_dataset(str(i), data=true_state)

    metadata_grp = f.create_group('metadata')
    if len(metadata) > 0:
        for name in metadata[0]:
            metadata_sub_grp = metadata_grp.create_group(name)
            for i, data in enumerate(md[name] for md in metadata):
                metadata_sub_grp.create_dataset(str(i), data=data)
