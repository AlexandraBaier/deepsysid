import dataclasses
import os
from typing import Dict, Iterator, List, Literal, Optional

import h5py
import numpy as np
from numpy.typing import NDArray

from ..configuration import ExperimentConfiguration
from ..data_io import build_result_file_name, load_file_names, load_simulation_data
from .base import TestResult, TestResultMetadata, TestSequenceResult, TestSimulation


def load_test_simulations(
    configuration: ExperimentConfiguration,
    mode: Literal['train', 'validation', 'test'],
    dataset_directory: str,
) -> List[TestSimulation]:
    # Prepare test data
    dataset_directory = os.path.join(dataset_directory, 'processed', mode)

    file_names = list(
        map(
            lambda fn: os.path.basename(fn).split('.')[0],
            load_file_names(dataset_directory),
        )
    )
    controls, states, initial_states = load_simulation_data(
        directory=dataset_directory,
        control_names=configuration.control_names,
        state_names=configuration.state_names,
        initial_state_names=configuration.initial_state_names,
    )

    simulations = [
        TestSimulation(control, state, initial_state, file_name)
        for control, state, initial_state, file_name in zip(
            controls, states, initial_states, file_names
        )
    ]

    return simulations


def save_model_tests(
    main_result: TestResult,
    additional_test_results: Dict[str, TestResult],
    config: ExperimentConfiguration,
    result_directory: str,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
) -> None:
    # Save true and predicted time series
    result_directory = os.path.join(result_directory, model_name)

    os.makedirs(result_directory, exist_ok=True)

    result_file_path = os.path.join(
        result_directory,
        build_result_file_name(mode, config.window_size, config.horizon_size, 'hdf5'),
    )

    with h5py.File(result_file_path, mode='w') as f:
        f.attrs['control_names'] = np.array(
            [np.string_(name) for name in config.control_names]
        )
        f.attrs['state_names'] = np.array(
            [np.string_(name) for name in config.state_names]
        )

        main_grp = f.create_group('main')
        main_metadata_grp = main_grp.create_group('metadata')
        _save_test_result_metadata_to_hdf_group(main_result.metadata, main_metadata_grp)

        for idx, sequence in enumerate(main_result.sequences):
            sequence_grp = main_grp.create_group(str(idx))
            _save_test_sequence_result_to_hdf_group(sequence, sequence_grp)

        additional_grp = f.create_group('additional')
        for test_name, test_result in additional_test_results.items():
            test_grp = additional_grp.create_group(test_name)
            additional_metadata_grp = test_grp.create_group('metadata')
            _save_test_result_metadata_to_hdf_group(
                test_result.metadata, additional_metadata_grp
            )

            for idx, sequence in enumerate(test_result.sequences):
                sequence_grp = test_grp.create_group(str(idx))
                _save_test_sequence_result_to_hdf_group(sequence, sequence_grp)


def _save_test_result_metadata_to_hdf_group(
    metadata: TestResultMetadata, group: h5py.Group
) -> None:
    for metadata_name, metadata_value in metadata.items():
        if len(metadata_value) == 0:
            continue
        if isinstance(metadata_value[0], (int, float)):
            group.create_dataset(metadata_name, data=np.array(metadata_value))
        else:
            group.create_dataset(
                metadata_name,
                data=np.array([np.string_(value) for value in metadata_value]),
            )


def _save_test_sequence_result_to_hdf_group(
    sequence: TestSequenceResult, group: h5py.Group
) -> None:
    inputs_grp = group.create_group('inputs')
    outputs_grp = group.create_group('outputs')
    additional_sequence_metadata_grp = group.create_group('metadata')

    for input_name, input_data in sequence.inputs.items():
        inputs_grp.create_dataset(input_name, data=input_data)
    for output_name, output_data in sequence.outputs.items():
        outputs_grp.create_dataset(output_name, data=output_data)
    for metadata_name, metadata_data in sequence.metadata.items():
        additional_sequence_metadata_grp.create_dataset(
            metadata_name, data=metadata_data
        )


@dataclasses.dataclass
class SimulateTestSample:
    initial_control: NDArray[np.float64]
    initial_state: NDArray[np.float64]
    true_control: NDArray[np.float64]
    true_state: NDArray[np.float64]
    x0: Optional[NDArray[np.float64]]
    file_name: str


def split_simulations(
    window_size: int,
    horizon_size: int,
    simulations: List[TestSimulation],
) -> Iterator[SimulateTestSample]:
    total_length = window_size + horizon_size
    for sim in simulations:
        for i in range(total_length, sim.control.shape[0], total_length):
            initial_control = sim.control[
                i - total_length : i - total_length + window_size
            ]
            initial_state = sim.state[i - total_length : i - total_length + window_size]
            true_control = sim.control[i - total_length + window_size : i]
            true_state = sim.state[i - total_length + window_size : i]
            x0 = sim.initial_state[i - total_length + window_size]

            yield SimulateTestSample(
                initial_control,
                initial_state,
                true_control,
                true_state,
                x0,
                sim.file_name,
            )
