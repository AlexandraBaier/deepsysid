import argparse
import json
import os
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
from ..pipeline.testing import test_model
from ..pipeline.training import train_model

CONFIGURATION_ENV_VAR = 'CONFIGURATION'
DATASET_DIR_ENV_VAR = 'DATASET_DIRECTORY'
MODELS_DIR_ENV_VAR = 'MODELS_DIRECTORY'
RESULT_DIR_ENV_VAR = 'RESULT_DIRECTORY'


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
            func=self.__evaluate_4dof_ship_trajectory
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
            func=self.__evaluate_quadcopter_trajectory
        )

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

        grid_search_template = ExperimentGridSearchTemplate.parse_obj(template)
        configuration = ExperimentConfiguration.from_grid_search_template(
            grid_search_template, 'cpu'
        )

        with open(os.environ[CONFIGURATION_ENV_VAR], mode='w') as f:
            json.dump(configuration.dict(), f)

    def __train_model(self, args):
        if 'device_idx' in args:
            device_name = build_device_name(args.enable_cuda, args.device_idx)
        else:
            device_name = build_device_name(args.enable_cuda, None)

        with open(os.environ[CONFIGURATION_ENV_VAR], mode='r') as f:
            config = json.load(f)
        config = ExperimentConfiguration.parse_obj(config)

        train_model(
            model_name=args.model,
            device_name=device_name,
            configuration=config,
            dataset_directory=os.environ[DATASET_DIR_ENV_VAR],
            disable_stdout=args.disable_stdout,
            models_directory=os.environ[MODELS_DIR_ENV_VAR],
        )

    def __test_model(self, args):
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

    def __evaluate_model(self, args):
        with open(os.environ[CONFIGURATION_ENV_VAR], mode='r') as f:
            config = json.load(f)
        config = ExperimentConfiguration.parse_obj(config)

        evaluate_model(
            config=config,
            model_name=args.model,
            mode=args.mode,
            result_directory=os.environ[RESULT_DIR_ENV_VAR],
        )

        model = initialize_model(
            config, args.model, device_name=build_device_name(False)
        )
        if config.thresholds and isinstance(model, HybridResidualLSTMModel):
            for threshold in config.thresholds:
                evaluate_model(
                    config=config,
                    model_name=args.model,
                    mode=args.mode,
                    result_directory=os.environ[RESULT_DIR_ENV_VAR],
                    threshold=threshold,
                )

    def __evaluate_4dof_ship_trajectory(self, args):
        with open(os.environ[CONFIGURATION_ENV_VAR], mode='r') as f:
            config = json.load(f)
        config = ExperimentConfiguration.parse_obj(config)

        evaluate_4dof_ship_trajectory(
            configuration=config,
            result_directory=os.environ[RESULT_DIR_ENV_VAR],
            model_name=args.model,
            mode=args.mode,
        )

    def __evaluate_quadcopter_trajectory(self, args):
        with open(os.environ[CONFIGURATION_ENV_VAR], mode='r') as f:
            config = json.load(f)
        config = ExperimentConfiguration.parse_obj(config)

        evaluate_quadcopter_trajectory(
            configuration=config,
            result_directory=os.environ[RESULT_DIR_ENV_VAR],
            model_name=args.model,
            mode=args.mode,
        )

    def __write_model_names(self, args):
        with open(os.environ[CONFIGURATION_ENV_VAR], mode='r') as f:
            config = json.load(f)

        config = ExperimentConfiguration.parse_obj(config)
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
