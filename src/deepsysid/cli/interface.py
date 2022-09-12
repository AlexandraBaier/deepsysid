import argparse
import json
import logging
import os
import sys
from typing import Optional

from ..models.hybrid.bounded_residual import HybridResidualLSTMModel
from ..pipeline.configuration import (
    ExperimentConfiguration,
    ExperimentGridSearchTemplate,
    initialize_model,
)
from ..pipeline.evaluation import (
    evaluate_4dof_ship_trajectory,
    evaluate_model,
    evaluate_quadcopter_trajectory,
)
from ..pipeline.gridsearch import (
    ExperimentSessionManager,
    ExperimentSessionReport,
    SessionAction,
)
from ..pipeline.testing import test_model
from ..pipeline.training import train_model
from .download import (
    download_dataset_4_dof_simulated_ship,
    download_dataset_pelican_quadcopter,
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
        self.build_configuration_parser.set_defaults(func=build_configuration)

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

        self.evaluate_parser = self.subparsers.add_parser(
            name='evaluate', help='Evaluate a model.'
        )
        add_parser_arguments(
            self.evaluate_parser, model_argument=True, mode_argument=True
        )
        self.evaluate_parser.set_defaults(func=evaluate)

        self.evaluate_4dof_ship_trajectory_parser = self.subparsers.add_parser(
            name='evaluate_4dof_ship_trajectory',
            help=(
                'Evaluate the trajectory of a 4-DOF ship model, '
                'such as the 4-DOF ship motion dataset found at '
                'https://darus.uni-stuttgart.de/dataset.xhtml'
                '?persistentId=doi:10.18419/darus-2905.'
            ),
        )
        add_parser_arguments(
            self.evaluate_4dof_ship_trajectory_parser,
            model_argument=True,
            mode_argument=True,
        )
        self.evaluate_4dof_ship_trajectory_parser.set_defaults(
            func=cli_evaluate_4dof_ship_trajectory
        )

        self.evaluate_quadcopter_parser = self.subparsers.add_parser(
            name='evaluate_quadcopter_trajectory',
            help=(
                'Evaluate the trajectory of a 6-DOF quadcopter model, '
                'such as the quadcopter motion dataset found at '
                'https://github.com/wavelab/pelican_dataset.'
            ),
        )
        add_parser_arguments(
            self.evaluate_quadcopter_parser, model_argument=True, mode_argument=True
        )
        self.evaluate_quadcopter_parser.set_defaults(
            func=cli_evaluate_quadcopter_trajectory
        )

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
                'Run a full experiment given a grid-search session template. '
                'State of the session can be loaded from and is saved to disk. '
                'This allows stopping and continuing a session at any point.'
            ),
        )
        self.session_parser.add_argument(
            'template', help='Path to grid-search template JSON.'
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

        self.download_pelican_parser = self.download_subparsers.add_parser(
            'pelican', help='Downloads https://github.com/wavelab/pelican_dataset.'
        )
        self.download_pelican_parser.add_argument(
            'target', help='Target directory for dataset.'
        )
        self.download_pelican_parser.set_defaults(func=download_pelican)

    def run(self) -> None:
        args = self.parser.parse_args()
        args.func(args)


def build_configuration(args: argparse.Namespace) -> None:
    with open(os.path.expanduser(args.template), mode='r') as f:
        template = json.load(f)

    grid_search_template = ExperimentGridSearchTemplate.parse_obj(template)
    configuration = ExperimentConfiguration.from_grid_search_template(
        grid_search_template, 'cpu'
    )

    with open(os.environ[CONFIGURATION_ENV_VAR], mode='w') as f:
        json.dump(configuration.dict(), f)


def train(args: argparse.Namespace) -> None:
    if 'device_idx' in args:
        device_name = build_device_name(args.enable_cuda, args.device_idx)
    else:
        device_name = build_device_name(args.enable_cuda, None)

    # Configure logging
    # This is the "root" logger, so we do not initializer it with a name.
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not args.disable_stdout:
        logger.addHandler(logging.StreamHandler(sys.stdout))

    with open(os.environ[CONFIGURATION_ENV_VAR], mode='r') as f:
        config = json.load(f)
    config = ExperimentConfiguration.parse_obj(config)

    train_model(
        model_name=args.model,
        device_name=device_name,
        configuration=config,
        dataset_directory=os.environ[DATASET_DIR_ENV_VAR],
        models_directory=os.environ[MODELS_DIR_ENV_VAR],
    )


def test(args: argparse.Namespace) -> None:
    if 'device_idx' in args:
        device_name = build_device_name(args.enable_cuda, args.device_idx)
    else:
        device_name = build_device_name(args.enable_cuda, None)

    with open(os.environ[CONFIGURATION_ENV_VAR], mode='r') as f:
        config = json.load(f)
    config = ExperimentConfiguration.parse_obj(config)

    test_model(
        model_name=args.model,
        device_name=device_name,
        mode=args.mode,
        configuration=config,
        dataset_directory=os.environ[DATASET_DIR_ENV_VAR],
        result_directory=os.environ[RESULT_DIR_ENV_VAR],
        models_directory=os.environ[MODELS_DIR_ENV_VAR],
    )


def evaluate(args: argparse.Namespace) -> None:
    with open(os.environ[CONFIGURATION_ENV_VAR], mode='r') as f:
        config = json.load(f)
    config = ExperimentConfiguration.parse_obj(config)

    evaluate_model(
        config=config,
        model_name=args.model,
        mode=args.mode,
        result_directory=os.environ[RESULT_DIR_ENV_VAR],
    )

    model = initialize_model(config, args.model, device_name=build_device_name(False))
    if config.thresholds and isinstance(model, HybridResidualLSTMModel):
        for threshold in config.thresholds:
            evaluate_model(
                config=config,
                model_name=args.model,
                mode=args.mode,
                result_directory=os.environ[RESULT_DIR_ENV_VAR],
                threshold=threshold,
            )


def cli_evaluate_4dof_ship_trajectory(args: argparse.Namespace) -> None:
    with open(os.environ[CONFIGURATION_ENV_VAR], mode='r') as f:
        config = json.load(f)
    config = ExperimentConfiguration.parse_obj(config)

    evaluate_4dof_ship_trajectory(
        configuration=config,
        result_directory=os.environ[RESULT_DIR_ENV_VAR],
        model_name=args.model,
        mode=args.mode,
    )


def cli_evaluate_quadcopter_trajectory(args: argparse.Namespace) -> None:
    with open(os.environ[CONFIGURATION_ENV_VAR], mode='r') as f:
        config = json.load(f)
    config = ExperimentConfiguration.parse_obj(config)

    evaluate_quadcopter_trajectory(
        configuration=config,
        result_directory=os.environ[RESULT_DIR_ENV_VAR],
        model_name=args.model,
        mode=args.mode,
    )


def write_model_names(args: argparse.Namespace) -> None:
    with open(os.environ[CONFIGURATION_ENV_VAR], mode='r') as f:
        config = json.load(f)

    config = ExperimentConfiguration.parse_obj(config)
    model_names = list(config.models.keys())

    with open(args.output, mode='w') as f:
        f.write('\n'.join(model_names) + '\n')


def session(args: argparse.Namespace) -> None:
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    if 'device_idx' in args:
        device_name = build_device_name(args.enable_cuda, args.device_idx)
    else:
        device_name = build_device_name(args.enable_cuda, None)

    with open(os.path.expanduser(args.template), mode='r') as f:
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
        dataset_directory=os.environ[DATASET_DIR_ENV_VAR],
        models_directory=os.environ[MODELS_DIR_ENV_VAR],
        results_directory=os.environ[RESULT_DIR_ENV_VAR],
        session_report=report,
    )

    def save_report(r: ExperimentSessionReport) -> None:
        with open(args.reportout, mode='w') as f:
            f.write(r.json())

    manager.run_session(callback=save_report)
    save_report(manager.get_session_report())


def download_4dof_sim_ship(args: argparse.Namespace) -> None:
    routine_dir = args.target_routine
    ood_dir = args.target_ood

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    download_dataset_4_dof_simulated_ship(
        routine_directory=routine_dir, ood_directory=ood_dir
    )


def download_pelican(args: argparse.Namespace) -> None:
    directory = args.target

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler(sys.stdout))

    download_dataset_pelican_quadcopter(target_directory=directory)


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
