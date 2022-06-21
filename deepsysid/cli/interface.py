import argparse
import dataclasses
import json
import logging
import os
import sys
from typing import Any, Dict, Iterator, List, Literal, Optional, Tuple

import h5py
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .. import execution, utils
from ..models.base import DynamicIdentificationModel
from ..models.hybrid.bounded_residual import HybridResidualLSTMModel


def run_deepsysid_cli():
    cli = DeepSysIdCommandLineInterface()
    cli.run()


class DeepSysIdCommandLineInterface:
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            'Command line interface for the deepsysid package.'
        )
        self.parser.set_defaults(func=lambda args: self.parser.print_help())

        self.subparsers = self.parser.add_subparsers()

        self.build_configuration_parser = self.subparsers.add_parser(
            name='build_configuration',
            help=(
                'Build configuration file given grid-search configuration template. '
                'Resulting configuration is written to CONFIGURATION.'
            ),
        )
        self.build_configuration_parser.add_argument(
            'template', help='path to grid-search template JSON'
        )
        self.build_configuration_parser.set_defaults(func=self.__build_configuration)

        self.train_parser = self.subparsers.add_parser(
            name='train', help='Train a model.'
        )
        add_parser_arguments(
            self.train_parser,
            model_argument=True,
            cuda_argument=True,
            stdout_argument=True,
        )
        self.train_parser.set_defaults(func=self.__train_model)

        self.test_parser = self.subparsers.add_parser(name='test', help='Test a model.')
        add_parser_arguments(
            self.test_parser,
            model_argument=True,
            cuda_argument=True,
            mode_argument=True,
        )
        self.test_parser.set_defaults(func=self.__test_model)

        self.evaluate_parser = self.subparsers.add_parser(
            name='evaluate', help='Evaluate a model.'
        )
        add_parser_arguments(
            self.evaluate_parser, model_argument=True, mode_argument=True
        )
        self.evaluate_parser.set_defaults(func=self.__evaluate_model)

        self.write_model_names_parser = self.subparsers.add_parser(
            name='write_model_names',
            help='Write all model names from the configuration to a text file.',
        )
        self.write_model_names_parser.add_argument(
            'output', help='Output path for generated text file.'
        )
        self.write_model_names_parser.set_defaults(func=self.__write_model_names)

    def run(self):
        args = self.parser.parse_args()
        args.func(args)

    def __build_configuration(self, args):
        with open(os.path.expanduser(args.template), mode='r') as f:
            template = json.load(f)

        grid_search_template = execution.ExperimentGridSearchTemplate.parse_obj(
            template
        )
        configuration = execution.ExperimentConfiguration.from_grid_search_template(
            grid_search_template, 'cpu'
        )

        with open(os.environ['CONFIGURATION'], mode='w') as f:
            json.dump(configuration.dict(), f)

    def __train_model(self, args):
        if 'device_idx' in args:
            device_name = build_device_name(args.enable_cuda, args.device_idx)
        else:
            device_name = build_device_name(args.enable_cuda, None)

        train_model(
            model_name=args.model,
            device_name=device_name,
            configuration_path=os.environ['CONFIGURATION'],
            dataset_directory=os.environ['DATASET_DIRECTORY'],
            disable_stdout=args.disable_stdout,
        )

    def __test_model(self, args):
        if 'device_idx' in args:
            device_name = build_device_name(args.enable_cuda, args.device_idx)
        else:
            device_name = build_device_name(args.enable_cuda, None)

        with open(os.environ['CONFIGURATION'], mode='r') as f:
            configuration = json.load(f)
        configuration = execution.ExperimentConfiguration.parse_obj(configuration)

        model_directory = os.path.expanduser(
            os.path.normpath(configuration.models[args.model].location)
        )
        model = execution.initialize_model(configuration, args.model, device_name)
        execution.load_model(model, model_directory, args.model)

        simulations = load_test_simulations(
            configuration=configuration,
            model_name=args.model,
            device_name=device_name,
            mode=args.mode,
            dataset_directory=os.environ['DATASET_DIRECTORY'],
        )

        test_result = run_model_tests(
            simulations=simulations, config=configuration, model=model
        )
        save_model_tests(
            test_result=test_result,
            config=configuration,
            result_directory=os.environ['RESULT_DIRECTORY'],
            model_name=args.model,
            mode=args.mode,
        )

        if configuration.thresholds and isinstance(model, HybridResidualLSTMModel):
            for threshold in configuration.thresholds:
                test_result = run_model_tests(
                    simulations=simulations,
                    config=configuration,
                    model=model,
                    threshold=threshold,
                )
                save_model_tests(
                    test_result=test_result,
                    config=configuration,
                    result_directory=os.environ['RESULT_DIRECTORY'],
                    model_name=args.model,
                    mode=args.mode,
                    threshold=threshold,
                )

    def __evaluate_model(self, args):
        with open(os.environ['CONFIGURATION'], mode='r') as f:
            config = json.load(f)
        config = execution.ExperimentConfiguration.parse_obj(config)

        evaluate_model(
            config=config,
            model_name=args.model,
            mode=args.mode,
            result_directory=os.environ['RESULT_DIRECTORY'],
        )

        model = execution.initialize_model(
            config, args.model, device_name=build_device_name(False)
        )
        if config.thresholds and isinstance(model, HybridResidualLSTMModel):
            for threshold in config.thresholds:
                evaluate_model(
                    config=config,
                    model_name=args.model,
                    mode=args.mode,
                    result_directory=os.environ['RESULT_DIRECTORY'],
                    threshold=threshold,
                )

    def __write_model_names(self, args):
        with open(os.environ['CONFIGURATION'], mode='r') as f:
            config = json.load(f)

        config = execution.ExperimentConfiguration.parse_obj(config)
        model_names = list(config.models.keys())

        with open(args.output, mode='w') as f:
            f.write('\n'.join(model_names) + '\n')


def add_parser_arguments(
    parser: argparse.ArgumentParser,
    model_argument: bool = False,
    cuda_argument: bool = False,
    stdout_argument: bool = False,
    mode_argument: bool = False,
):
    if model_argument:
        parser.add_argument('model', help='model name as defined in configuration')
    if cuda_argument:
        parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA.')
        parser.add_argument(
            '--device-idx', action='store', type=int, help='Index of GPU.'
        )
    if stdout_argument:
        parser.add_argument(
            '--disable-stdout',
            action='store_true',
            help='Prevent logging to reach STDOUT.',
        )
    if mode_argument:
        parser.add_argument(
            '--mode',
            action='store',
            help='Either "train" or "validation" or "test".',
            choices=('train', 'validation', 'test'),
        )


def build_device_name(enable_cuda: bool, device_idx: Optional[int] = None) -> str:
    if enable_cuda:
        if device_idx:
            device_name = f'cuda:{device_idx}'
        else:
            device_name = 'cuda'
    else:
        device_name = 'cpu'

    return device_name


def train_model(
    model_name: str,
    device_name: str,
    configuration_path: str,
    dataset_directory: str,
    disable_stdout: bool,
):
    # Load configuration
    with open(configuration_path, mode='r') as f:
        config = json.load(f)

    config = execution.ExperimentConfiguration.parse_obj(config)

    dataset_directory = os.path.join(dataset_directory, 'processed', 'train')
    model_directory = os.path.expanduser(
        os.path.normpath(config.models[model_name].location)
    )

    # Configure logging
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(
        logging.FileHandler(
            filename=os.path.join(model_directory, 'training.log'), mode='a'
        )
    )
    if not disable_stdout:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    # Load dataset
    controls, states = execution.load_simulation_data(
        directory=dataset_directory,
        control_names=config.control_names,
        state_names=config.state_names,
    )

    # Initialize model
    model = execution.initialize_model(config, model_name, device_name)
    # Train model
    logger.info(f'Training model on {device_name} if implemented.')
    model.train(control_seqs=controls, state_seqs=states)
    # Save model
    try:
        os.mkdir(model_directory)
    except FileExistsError:
        pass
    execution.save_model(model, model_directory, model_name)


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


@dataclasses.dataclass
class ModelTestResult:
    control: List[np.ndarray]
    true_state: List[np.ndarray]
    pred_state: List[np.ndarray]
    file_names: List[str]
    whitebox: List[np.ndarray]
    blackbox: List[np.ndarray]


def run_model_tests(
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


def evaluate_model(
    config: execution.ExperimentConfiguration,
    model_name: str,
    mode: Literal['train', 'validation', 'test'],
    result_directory: str,
    threshold: Optional[float] = None,
):
    test_file_path = os.path.join(
        result_directory,
        model_name,
        build_result_file_name(
            mode=mode,
            window_size=config.window_size,
            horizon_size=config.horizon_size,
            extension='hdf5',
            threshold=threshold,
        ),
    )

    scores_file_path = os.path.join(
        result_directory,
        model_name,
        build_score_file_name(
            mode=mode,
            window_size=config.window_size,
            horizon_size=config.horizon_size,
            extension='hdf5',
            threshold=threshold,
        ),
    )
    readable_scores_file_path = os.path.join(
        result_directory,
        model_name,
        build_score_file_name(
            mode=mode,
            window_size=config.window_size,
            horizon_size=config.horizon_size,
            extension='json',
            threshold=threshold,
        ),
    )

    pred = []
    true = []
    steps = []

    # Load predicted and true states for each multi-step sequence.
    with h5py.File(test_file_path, 'r') as f:
        file_names = [fn.decode('UTF-8') for fn in f['file_names'][:].tolist()]
        for i in range(len(file_names)):
            pred.append(f['predicted'][str(i)][:])
            true.append(f['true'][str(i)][:])
            steps.append(f['predicted'][str(i)][:].shape[0])

    def mse(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return mean_squared_error(t, p, multioutput='raw_values')

    def rmse(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.sqrt(mean_squared_error(t, p, multioutput='raw_values'))

    def rmse_std(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return np.std(
            np.sqrt(mean_squared_error(t, p, multioutput='raw_values')), axis=0
        )

    def mae(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return mean_absolute_error(t, p, multioutput='raw_values')

    def d1(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return utils.index_of_agreement(t, p, j=1)

    score_functions = (
        ('mse', mse),
        ('rmse', rmse),
        ('rmse-std', rmse_std),
        ('mae', mae),
        ('d1', d1),
    )

    scores = dict()
    for name, fct in score_functions:
        scores[name] = utils.score_on_sequence(true, pred, fct)

    with h5py.File(scores_file_path, 'w') as f:
        f.attrs['state_names'] = np.array(list(map(np.string_, config.state_names)))
        f.create_dataset('file_names', data=np.array(list(map(np.string_, file_names))))
        f.create_dataset('steps', data=np.array(steps))

        for name, _ in score_functions:
            f.create_dataset(name, data=scores[name])

    average_scores = dict()
    for name, _ in score_functions:
        # 1/60 * sum_1^60 RMSE
        average_scores[name] = np.average(scores[name], weights=steps, axis=0).tolist()

    with open(readable_scores_file_path, mode='w') as f:
        obj: Dict[str, Any] = dict()
        obj['scores'] = average_scores
        obj['state_names'] = config.state_names
        json.dump(obj, f)


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


def build_score_file_name(
    mode: Literal['train', 'validation', 'test'],
    window_size: int,
    horizon_size: int,
    extension: str,
    threshold: Optional[float] = None,
) -> str:
    if threshold is None:
        return f'scores-{mode}-w_{window_size}-h_{horizon_size}.{extension}'

    threshold_str = f'{threshold:.f}'.replace('.', '')
    return (
        f'scores-{mode}-w_{window_size}-h_{horizon_size}-t_{threshold_str}.{extension}'
    )
