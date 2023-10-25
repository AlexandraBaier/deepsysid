import logging
import os
from typing import Literal

from ...pipeline.model_io import load_model
from ..configuration import (
    ExperimentConfiguration,
    initialize_model,
    initialize_tracker,
)
from .base import BaseTestConfig, retrieve_test_class
from .inference import InferenceTest
from .io import load_test_simulations, save_model_tests

from ...tracker.event_data import LoadTrackingConfiguration, SetTags, StopRun, TrackParameters, SaveTrackingConfiguration

logger = logging.getLogger(__name__)


def test_model(
    configuration: ExperimentConfiguration,
    model_name: str,
    device_name: str,
    mode: Literal['train', 'validation', 'test'],
    dataset_directory: str,
    result_directory: str,
    models_directory: str,
) -> None:
    model_directory = os.path.expanduser(
        os.path.normpath(os.path.join(models_directory, model_name))
    )
    model = initialize_model(configuration, model_name, device_name)
    load_model(model, model_directory, model_name)

    tracker = initialize_tracker(experiment_config=configuration)
    tracker(
        LoadTrackingConfiguration(
            f'Load configuration from {model_directory}', model_directory, model_name
        )
    )
    if configuration.tracker is not None:
        tracker(
            SaveTrackingConfiguration(
                'Write tracking config to json file',
                configuration.tracker,
                model_name,
                model_directory,
            )
        )

    simulations = load_test_simulations(
        configuration=configuration,
        mode=mode,
        dataset_directory=dataset_directory,
    )

    main_test = InferenceTest(
        BaseTestConfig(
            control_names=configuration.control_names,
            state_names=configuration.state_names,
            window_size=configuration.window_size,
            horizon_size=configuration.horizon_size,
        ),
        device_name=device_name,
    )
    main_results = main_test.test(model, simulations)

    additional_results = dict()
    for test_name, test_config in configuration.additional_tests.items():
        test_cls = retrieve_test_class(test_config.test_class)
        test_instance = test_cls(test_config.parameters, device_name)

        additional_results[test_name] = test_instance.test(model, simulations, tracker)

    save_model_tests(
        main_result=main_results,
        additional_test_results=additional_results,
        config=configuration,
        result_directory=result_directory,
        model_name=model_name,
        mode=mode,
    )
    tracker(StopRun('Stop validation run', None))
