import argparse
import json
import logging
import os
import sys
import time
from typing import Optional

from ..pipeline.configuration import (
    ExperimentConfiguration,
    ExperimentGridSearchTemplate,
)
from ..pipeline.evaluation import evaluate_model
from ..pipeline.explaining import explain_model
from ..pipeline.gridsearch import (
    ExperimentSessionManager,
    ExperimentSessionReport,
    SessionAction,
)
from ..pipeline.testing.runner import test_model
from ..pipeline.training import train_model
from .download import (
    download_dataset_4_dof_simulated_ship,
    download_dataset_industrial_robot,
    download_dataset_pelican_quadcopter,
    download_dataset_toy,
)

CONFIGURATION_ENV_VAR = 'CONFIGURATION'
DATASET_DIR_ENV_VAR = 'DATASET_DIRECTORY'
MODELS_DIR_ENV_VAR = 'MODELS_DIRECTORY'
RESULT_DIR_ENV_VAR = 'RESULT_DIRECTORY'


def run_deepsysid_cli() -> None:
    cli = DeepSysIdCommandLineInterface()
    cli.run()


class DeepSysIdCommandLineInterface:
    def __init__(self) -> None:
        self.parser = argparse.ArgumentParser(
            'Command line interface for the deepsysid package.'
        )
        self.parser.set_defaults(func=lambda args: self.parser.print_help())

        self.subparsers = self.parser.add_subparsers()

        self.validate_configuration_parser = self.subparsers.add_parser(
            name='validate_configuration',
            help='Validate configuration file defined in CONFIGURATION.',
        )
        self.validate_configuration_parser.set_defaults(func=validate_configuration)

        self.train_parser = self.subparsers.add_parser(
            name='train', help='Train a model.'
        )
        add_parser_arguments(
            self.train_parser,
            model_argument=True,
            cuda_argument=True,
            stdout_argument=True,
        )
        self.train_parser.set_defaults(func=train)

        self.test_parser = self.subparsers.add_parser(name='test', help='Test a model.')
        add_parser_arguments(
            self.test_parser,
            model_argument=True,
            cuda_argument=True,
            mode_argument=True,
        )
        self.test_parser.set_defaults(func=test)

        self.explain_parser = self.subparsers.add_parser(
            name='explain', help='Explain a model.'
        )
        add_parser_arguments(
            self.explain_parser,
            model_argument=True,
            cuda_argument=True,
            mode_argument=True,
        )
        self.explain_parser.set_defaults(func=explain)

        self.evaluate_parser = self.subparsers.add_parser(
            name='evaluate', help='Evaluate a model.'
        )
        add_parser_arguments(
            self.evaluate_parser, model_argument=True, mode_argument=True
        )
        self.evaluate_parser.set_defaults(func=evaluate)

        self.write_model_names_parser = self.subparsers.add_parser(
            name='write_model_names',
            help='Write all model names from the configuration to a text file.',
        )
        self.write_model_names_parser.add_argument(
            'output', help='Output path for generated text file.'
        )
        self.write_model_names_parser.set_defaults(func=write_model_names)

        self.session_parser = self.subparsers.add_parser(
            name='session',
            help=(
                'Run a full experiment given the configuration JSON. '
                'State of the session can be loaded from and is saved to disk. '
                'This allows stopping and continuing a session at any point.'
            ),
        )
        self.session_parser.add_argument(
            'reportout', help='Output path for session report JSON.'
        )
        self.session_parser.add_argument(
            'action',
            action='store',
            help=(
                'Select whether to start a "NEW" session, '
                '"CONTINUE" an old session, or finish the experiment with "TEST_BEST".'
            ),
            choices=('NEW', 'CONTINUE', 'TEST_BEST'),
        )
        self.session_parser.add_argument(
            '--reportin',
            help=(
                'For "CONTINUE" and "TEST_BEST" '
                'you need to provide an existing session report.'
            ),
        )
        add_parser_arguments(self.session_parser, cuda_argument=True)
        self.session_parser.set_defaults(func=session)

        self.download_parser = self.subparsers.add_parser(
            name='download', help='Download and prepare datasets.'
        )
        self.download_parser.set_defaults(
            func=lambda args: self.download_parser.print_help()
        )
        self.download_subparsers = self.download_parser.add_subparsers()

        self.download_4dof_sim_ship_parser = self.download_subparsers.add_parser(
            '4dof-sim-ship', help='Downloads https://doi.org/10.18419/darus-2905.'
        )
        self.download_4dof_sim_ship_parser.add_argument(
            'target_routine', help='Target directory for routine operation dataset.'
        )
        self.download_4dof_sim_ship_parser.add_argument(
            'target_ood', help='Target directory for OOD operation dataset.'
        )
        self.download_4dof_sim_ship_parser.set_defaults(func=download_4dof_sim_ship)

        self.download_toy_dataset_parser = self.download_subparsers.add_parser(
            'toy_dataset',
            help=(
                'Downloads Coupled MSD, Pole on a Cart, ',
                'Single pendulum with torque input, ',
                'see https://github.com/Dany-L/statesim for details.',
            ),
        )
        self.download_toy_dataset_parser.add_argument(
            'target', help='Target directory for dataset.'
        )
        self.download_toy_dataset_parser.set_defaults(func=download_toy_dataset)

        self.download_pelican_parser = self.download_subparsers.add_parser(
            'pelican', help='Downloads https://github.com/wavelab/pelican_dataset.'
        )

        self.download_pelican_parser.add_argument(
            'target', help='Target directory for dataset.'
        )
        self.download_pelican_parser.add_argument(
            '--train_fraction',
            required=True,
            action='store',
            type=float,
            help='Fraction of dataset used for training.',
        )
        self.download_pelican_parser.add_argument(
            '--validation_fraction',
            required=True,
            action='store',
            type=float,
            help='Fraction of dataset used for validation.',
        )
        self.download_pelican_parser.set_defaults(func=download_pelican)

        self.download_industrial_robot_parser = self.download_subparsers.add_parser(
            'industrial-robot',
            help='Downloads https://fdm-fallback.uni-kl.de/TUK/FB/MV/WSKL/0001/.',
        )
        self.download_industrial_robot_parser.add_argument(
            'target', help='Target directory for dataset.'
        )
        self.download_industrial_robot_parser.add_argument(
            '--validation_fraction',
            required=True,
            action='store',
            type=float,
            help='Fraction of dataset used for validation.',
        )
        self.download_industrial_robot_parser.set_defaults(
            func=download_industrial_robot
        )

    def run(self) -> None:
        args = self.parser.parse_args()
        args.func(args)


def validate_configuration(_: argparse.Namespace) -> None:
    with open(os.path.expanduser(os.environ[CONFIGURATION_ENV_VAR]), mode='r') as f:
        template = json.load(f)

    grid_search_template = ExperimentGridSearchTemplate.parse_obj(template)
    ExperimentConfiguration.from_grid_search_template(grid_search_template)


def train(args: argparse.Namespace) -> None:
    if 'device_idx' in args:
        device_name = build_device_name(args.enable_cuda, args.device_idx)
    else:
        device_name = build_device_name(args.enable_cuda, None)

    # Configure logging
    # This is the "root" logger, so we do not initializer it with a name.
    setup_root_logger(
        disable_stdout=args.disable_stdout,
        file_name=os.path.expanduser(
            os.path.normpath(
                os.path.join(os.environ[MODELS_DIR_ENV_VAR], args.model, 'training.log')
            )
        ),
    )

    with open(os.path.expanduser(os.environ[CONFIGURATION_ENV_VAR]), mode='r') as f:
        config_dict = json.load(f)
    config = ExperimentConfiguration.from_grid_search_template(
        ExperimentGridSearchTemplate.parse_obj(config_dict), device_name=device_name
    )

    train_model(
        model_name=args.model,
        device_name=device_name,
        configuration=config,
        dataset_directory=os.path.expanduser(os.environ[DATASET_DIR_ENV_VAR]),
        models_directory=os.path.expanduser(os.environ[MODELS_DIR_ENV_VAR]),
    )


def test(args: argparse.Namespace) -> None:
    if 'device_idx' in args:
        device_name = build_device_name(args.enable_cuda, args.device_idx)
    else:
        device_name = build_device_name(args.enable_cuda, None)

    with open(os.path.expanduser(os.environ[CONFIGURATION_ENV_VAR]), mode='r') as f:
        config_dict = json.load(f)
    config = ExperimentConfiguration.from_grid_search_template(
        ExperimentGridSearchTemplate.parse_obj(config_dict), device_name=device_name
    )
    logger = setup_root_logger(
        file_name=os.path.expanduser(
            os.path.join(os.environ[RESULT_DIR_ENV_VAR], args.model, 'testing.log')
        )
    )

    time_start = time.time()

    test_model(
        model_name=args.model,
        device_name=device_name,
        mode=args.mode,
        configuration=config,
        dataset_directory=os.path.expanduser(os.environ[DATASET_DIR_ENV_VAR]),
        result_directory=os.path.expanduser(os.environ[RESULT_DIR_ENV_VAR]),
        models_directory=os.path.expanduser(os.environ[MODELS_DIR_ENV_VAR]),
    )

    time_end = time.time()
    logger.info(f'Testing time: {time_end - time_start:1f}')


def explain(args: argparse.Namespace) -> None:
    if 'device_idx' in args:
        device_name = build_device_name(args.enable_cuda, args.device_idx)
    else:
        device_name = build_device_name(args.enable_cuda, None)

    with open(os.path.expanduser(os.environ[CONFIGURATION_ENV_VAR]), mode='r') as f:
        config_dict = json.load(f)
    config = ExperimentConfiguration.from_grid_search_template(
        ExperimentGridSearchTemplate.parse_obj(config_dict), device_name=device_name
    )
    setup_root_logger(
        file_name=os.path.expanduser(
            os.path.join(os.environ[RESULT_DIR_ENV_VAR], args.model, 'explain.log')
        )
    )

    explain_model(
        model_name=args.model,
        device_name=device_name,
        mode=args.mode,
        configuration=config,
        dataset_directory=os.path.expanduser(os.environ[DATASET_DIR_ENV_VAR]),
        result_directory=os.path.expanduser(os.environ[RESULT_DIR_ENV_VAR]),
        models_directory=os.path.expanduser(os.environ[MODELS_DIR_ENV_VAR]),
    )


def evaluate(args: argparse.Namespace) -> None:
    with open(os.path.expanduser(os.environ[CONFIGURATION_ENV_VAR]), mode='r') as f:
        config_dict = json.load(f)
    config = ExperimentConfiguration.from_grid_search_template(
        ExperimentGridSearchTemplate.parse_obj(config_dict)
    )

    evaluate_model(
        config=config,
        model_name=args.model,
        mode=args.mode,
        result_directory=os.path.expanduser(os.environ[RESULT_DIR_ENV_VAR]),
    )


def write_model_names(args: argparse.Namespace) -> None:
    with open(os.path.expanduser(os.environ[CONFIGURATION_ENV_VAR]), mode='r') as f:
        config_dict = json.load(f)

    config = ExperimentConfiguration.from_grid_search_template(
        ExperimentGridSearchTemplate.parse_obj(config_dict)
    )
    model_names = list(config.models.keys())

    with open(args.output, mode='w') as f:
        f.write('\n'.join(model_names) + '\n')


def session(args: argparse.Namespace) -> None:
    setup_root_logger(
        file_name=os.path.expanduser(
            os.path.join(os.environ[RESULT_DIR_ENV_VAR], 'session.log')
        )
    )

    if 'device_idx' in args:
        device_name = build_device_name(args.enable_cuda, args.device_idx)
    else:
        device_name = build_device_name(args.enable_cuda, None)

    with open(os.path.expanduser(os.environ[CONFIGURATION_ENV_VAR]), mode='r') as f:
        template_obj = json.load(f)
    config = ExperimentGridSearchTemplate.parse_obj(template_obj)

    report: Optional[ExperimentSessionReport] = None
    if args.reportin:
        with open(os.path.expanduser(args.reportin), mode='r') as f:
            report_obj = json.load(f)
        report = ExperimentSessionReport.parse_obj(report_obj)

    if args.action.lower() == 'new':
        session_action = SessionAction.NEW
    elif args.action.lower() == 'continue':
        session_action = SessionAction.CONTINUE
    elif args.action.lower() == 'test_best':
        session_action = SessionAction.TEST_BEST
    else:
        raise ValueError(
            f'{args.action} is not a valid session action. '
            f'Valid actions are '
            f'{",".join(str(action) for action in SessionAction)}.'
        )

    manager = ExperimentSessionManager(
        config=config,
        device_name=device_name,
        session_action=session_action,
        dataset_directory=os.path.expanduser(os.environ[DATASET_DIR_ENV_VAR]),
        models_directory=os.path.expanduser(os.environ[MODELS_DIR_ENV_VAR]),
        results_directory=os.path.expanduser(os.environ[RESULT_DIR_ENV_VAR]),
        session_report=report,
    )

    def save_report(r: ExperimentSessionReport) -> None:
        with open(args.reportout, mode='w') as f:
            f.write(r.json())

    manager.run_session(callback=save_report)
    save_report(manager.get_session_report())


def download_toy_dataset(args: argparse.Namespace) -> None:
    target_dir = args.target

    setup_root_logger()
    download_dataset_toy(target_directory=target_dir)


def download_4dof_sim_ship(args: argparse.Namespace) -> None:
    routine_dir = args.target_routine
    ood_dir = args.target_ood

    setup_root_logger()

    download_dataset_4_dof_simulated_ship(
        routine_directory=routine_dir, ood_directory=ood_dir
    )


def download_pelican(args: argparse.Namespace) -> None:
    directory = args.target
    train_fraction = args.train_fraction
    validation_fraction = args.validation_fraction

    setup_root_logger()

    download_dataset_pelican_quadcopter(
        target_directory=directory,
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
    )


def download_industrial_robot(args: argparse.Namespace) -> None:
    directory = args.target
    validation_fraction = args.validation_fraction

    setup_root_logger()

    download_dataset_industrial_robot(directory, validation_fraction)


def add_parser_arguments(
    parser: argparse.ArgumentParser,
    model_argument: bool = False,
    cuda_argument: bool = False,
    stdout_argument: bool = False,
    mode_argument: bool = False,
) -> None:
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
            required=True,
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


def setup_root_logger(
    disable_stdout: bool = False, file_name: Optional[str] = None
) -> logging.Logger:
    logger = logging.getLogger('deepsysid')
    logger.setLevel(logging.INFO)
    if not disable_stdout:
        logger.addHandler(logging.StreamHandler(sys.stdout))
    if file_name is not None:
        directory = os.path.dirname(file_name)
        os.makedirs(directory, exist_ok=True)
        logger.addHandler(logging.FileHandler(filename=file_name, mode='a'))
    return logger
