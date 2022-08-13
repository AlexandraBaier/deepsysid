import argparse
import json
import os
from typing import Optional

from .. import execution
from ..models.hybrid.bounded_residual import HybridResidualLSTMModel
from .evaluation import (
    evaluate_model,
    save_trajectory_results,
    test_4dof_ship_trajectory,
    test_quadcopter_trajectory,
)
from .testing import build_result_file_name, test_model
from .training import train_model


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

        test_model(
            model_name=args.model,
            device_name=device_name,
            mode=args.mode,
            configuration_path=os.environ['CONFIGURATION'],
            dataset_directory=os.environ['DATASET_DIRECTORY'],
            result_directory=os.environ['RESULT_DIRECTORY'],
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

    def __evaluate_4dof_ship_trajectory(self, args):
        with open(os.environ['CONFIGURATION'], mode='r') as f:
            config = json.load(f)
        config = execution.ExperimentConfiguration.parse_obj(config)

        result_directory = os.environ['RESULT_DIRECTORY']
        result_file_path = os.path.join(
            result_directory,
            args.model,
            build_result_file_name(
                mode=args.mode,
                window_size=config.window_size,
                horizon_size=config.horizon_size,
                extension='hdf5',
            ),
        )

        result = test_4dof_ship_trajectory(
            config=config, result_file_path=result_file_path
        )

        save_trajectory_results(
            result=result,
            config=config,
            result_directory=result_directory,
            model_name=args.model,
            mode=args.mode,
        )

    def __evaluate_quadcopter_trajectory(self, args):
        with open(os.environ['CONFIGURATION'], mode='r') as f:
            config = json.load(f)
        config = execution.ExperimentConfiguration.parse_obj(config)

        test_directory = os.environ['RESULT_DIRECTORY']
        test_file_path = os.path.join(
            test_directory,
            args.model,
            build_result_file_name(
                mode=args.mode,
                window_size=config.window_size,
                horizon_size=config.horizon_size,
                extension='hdf5',
            ),
        )

        result = test_quadcopter_trajectory(
            config=config, result_file_path=test_file_path
        )

        save_trajectory_results(
            result=result,
            config=config,
            result_directory=test_directory,
            model_name=args.model,
            mode=args.mode,
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
