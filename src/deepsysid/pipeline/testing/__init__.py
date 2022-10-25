import abc
import dataclasses
import logging
import os
from typing import Type, Dict, List, Literal, Optional, Iterator, Union

import h5py
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

from deepsysid.models.base import DynamicIdentificationModel
from deepsysid.models.hybrid.bounded_residual import HybridResidualLSTMModel
from deepsysid.pipeline.configuration import ExperimentConfiguration, initialize_model
from deepsysid.pipeline.data_io import load_file_names, load_simulation_data, build_result_file_name
from deepsysid.pipeline.model_io import load_model


logger = logging.getLogger(__name__)


TestResultValue = Union[List[str], NDArray[np.float64]]
TestResult = Dict[str, Union[TestResultValue, Dict[str, TestResultValue]]]


@dataclasses.dataclass
class ModelTestResult:
    control: List[NDArray[np.float64]]
    true_state: List[NDArray[np.float64]]
    pred_state: List[NDArray[np.float64]]
    file_names: List[str]
    metadata: List[Dict[str, NDArray[np.float64]]]


@dataclasses.dataclass
class TestSimulation:
    control: NDArray[np.float64]
    state: NDArray[np.float64]
    file_name: str


class BaseTestConfig(BaseModel):
    control_names: List[str]
    state_names: List[str]
    window_size: int
    horizon_size: int


class BaseTest(metaclass=abc.ABCMeta):
    CONFIG: Type[BaseTestConfig] = BaseTestConfig

    def __init__(self, config: BaseTestConfig):
        pass

    @abc.abstractmethod
    def test(
            self,
            model: DynamicIdentificationModel,
            simulations: List[TestSimulation]
    ) -> TestResult:
        pass


def run_tests_on_model(
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

    logger.addHandler(
        logging.FileHandler(
            filename=os.path.join(model_directory, 'testing.log'), mode='a'
        )
    )

    _ = load_test_simulations(
        configuration=configuration,
        mode=mode,
        dataset_directory=dataset_directory,
    )


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
    controls, states = load_simulation_data(
        directory=dataset_directory,
        control_names=configuration.control_names,
        state_names=configuration.state_names,
    )

    simulations = [
        TestSimulation(control, state, file_name)
        for control, state, file_name in zip(controls, states, file_names)
    ]

    return simulations


def save_model_tests(
    test_result: ModelTestResult,
    config: ExperimentConfiguration,
    result_directory: str,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
    threshold: Optional[float] = None,
) -> None:
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


@dataclasses.dataclass
class SimulateTestSample:
    initial_control: NDArray[np.float64]
    initial_state: NDArray[np.float64]
    true_control: NDArray[np.float64]
    true_state: NDArray[np.float64]
    file_name: str


def split_simulations(
    window_size: int,
    horizon_size: int,
    simulations: List[TestSimulation],
) -> Iterator[SimulateTestSample]:
    total_length = window_size + horizon_size
    for sim in simulations:
        for i in range(total_length, sim.control.shape[0], total_length):
            initial_control = sim.control[i - total_length : i - total_length + window_size]
            initial_state = sim.state[i - total_length : i - total_length + window_size]
            true_control = sim.control[i - total_length + window_size : i]
            true_state = sim.state[i - total_length + window_size : i]

            yield SimulateTestSample(initial_control, initial_state, true_control, true_state, sim.file_name)


def write_test_results_to_hdf5(
    f: h5py.File,
    control_names: List[str],
    state_names: List[str],
    file_names: List[str],
    control: List[NDArray[np.float64]],
    pred_states: List[NDArray[np.float64]],
    true_states: List[NDArray[np.float64]],
    metadata: List[Dict[str, NDArray[np.float64]]],
) -> None:
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
