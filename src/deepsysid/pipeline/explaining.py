import logging
import os
from typing import Literal, Optional

import h5py

from deepsysid.explainability.base import (
    ExplainerNotImplementedForModel,
    ModelInput,
    retrieve_explainer_class,
    retrieve_explanation_metric_class,
)
from deepsysid.pipeline.configuration import ExperimentConfiguration, initialize_model
from deepsysid.pipeline.data_io import build_explanation_result_file_name
from deepsysid.pipeline.model_io import load_model
from deepsysid.pipeline.testing.io import load_test_simulations, split_simulations

logger = logging.getLogger(__name__)


def explain_model(
    configuration: ExperimentConfiguration,
    model_name: str,
    device_name: str,
    mode: Literal['train', 'validation', 'test'],
    dataset_directory: str,
    result_directory: str,
    models_directory: str,
    explainer_name: Optional[str] = None,
) -> None:
    model_directory = os.path.expanduser(
        os.path.normpath(os.path.join(models_directory, model_name))
    )
    model = initialize_model(configuration, model_name, device_name)
    load_model(model, model_directory, model_name)

    explained_simulations = load_test_simulations(
        configuration=configuration,
        mode=mode,
        dataset_directory=dataset_directory,
    )

    training_simulations = load_test_simulations(
        configuration=configuration, mode='train', dataset_directory=dataset_directory
    )
    training_inputs, training_outputs = zip(
        *[
            (
                ModelInput(
                    sim.initial_control,
                    sim.initial_state,
                    sim.true_control,
                    sim.x0,
                    sim.initial_x0,
                ),
                sim.true_state[-1, :],
            )
            for sim in split_simulations(
                configuration.window_size,
                configuration.horizon_size,
                training_simulations,
            )
        ]
    )

    model_inputs = [
        ModelInput(
            sim.initial_control,
            sim.initial_state,
            sim.true_control,
            sim.x0,
            sim.initial_x0,
        )
        for sim in split_simulations(
            configuration.window_size, configuration.horizon_size, explained_simulations
        )
    ]

    if configuration.explanation_metrics is None or configuration.explainers is None:
        return

    results = dict()
    explainers = configuration.explainers
    if explainer_name is not None:
        explainers = {explainer_name: explainers[explainer_name]}
    for metric_name, metric_config in configuration.explanation_metrics.items():
        metric_results = dict()
        metric_cls = retrieve_explanation_metric_class(metric_config.metric_class)
        metric = metric_cls(metric_config.parameters)

        for explainer_name, explainer_config in explainers.items():
            explainer_cls = retrieve_explainer_class(explainer_config.explainer_class)
            explainer = explainer_cls(explainer_config.parameters)
            explainer.initialize(list(training_inputs), list(training_outputs))
            logger.info(
                f'Running metric {metric.__class__} on explainer {explainer.__class__} '
                f'explaining {model.__class__}.'
            )
            try:
                result, metadata = metric.measure(model, explainer, model_inputs)
            except ExplainerNotImplementedForModel:
                logger.info(f'{model} cannot be explained with {explainer}.')
                continue
            metric_results[explainer_name] = (result, metadata)

        results[metric_name] = metric_results

    result_directory = os.path.expanduser(os.path.join(result_directory, model_name))
    os.makedirs(result_directory, exist_ok=True)

    result_file_name = os.path.join(
        result_directory,
        build_explanation_result_file_name(
            mode, configuration.window_size, configuration.horizon_size, 'hdf5'
        ),
    )
    with h5py.File(result_file_name, mode='w') as f:
        for metric_name, metric_result in results.items():
            metric_grp = f.create_group(metric_name)
            for explainer_name, (score, metadata) in metric_result.items():
                explainer_grp = metric_grp.create_group(explainer_name)
                explainer_grp.create_dataset('score', data=score)
                metadata_grp = explainer_grp.create_group('metadata')
                for meta_name, metadata_value in metadata.items():
                    metadata_grp.create_dataset(meta_name, data=metadata_value)
