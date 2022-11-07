import logging
import os
from typing import Literal

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
) -> None:
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

    model_inputs = [
        ModelInput(sim.initial_control, sim.initial_state, sim.true_control)
        for sim in split_simulations(
            configuration.window_size, configuration.horizon_size, simulations
        )
    ]

    if configuration.explanation_metrics is None or configuration.explainers is None:
        return

    results = dict()
    for metric_name, metric_config in configuration.explanation_metrics.items():
        metric_results = dict()
        metric_cls = retrieve_explanation_metric_class(metric_config.metric_class)
        metric = metric_cls(metric_config.parameters)

        for explainer_name, explainer_config in configuration.explainers.items():
            explainer_cls = retrieve_explainer_class(explainer_config.explainer_class)
            explainer = explainer_cls(explainer_config.parameters)
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
