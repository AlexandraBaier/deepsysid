import pytest

from deepsysid.models.base import DynamicIdentificationModel
from deepsysid.models.linear import LinearModel
from deepsysid.pipeline.configuration import (
    ExperimentConfiguration,
    ExperimentGridSearchSettings,
    ExperimentGridSearchTemplate,
    GridSearchMetricConfiguration,
    ModelGridSearchTemplate,
    retrieve_model_class,
)


def test_retrieve_model_class_module_does_not_exist():
    with pytest.raises(ModuleNotFoundError):
        retrieve_model_class('foo.bar.nothing.nada.NeverExistedEver')


def test_retrieve_model_class_not_dynamic_identification_model():
    with pytest.raises(ValueError):
        retrieve_model_class('datetime.timedelta')


def test_retrieve_model_class_successful():
    cls = retrieve_model_class('deepsysid.models.linear.LinearModel')

    assert issubclass(cls, DynamicIdentificationModel)
    assert cls == LinearModel


def test_experiment_configuration_from_grid_search_template_successful():
    template = ExperimentGridSearchTemplate(
        settings=ExperimentGridSearchSettings(
            train_fraction=0.6,
            validation_fraction=0.2,
            time_delta=0.5,
            window_size=10,
            horizon_size=20,
            control_names=['u1', 'u2'],
            state_names=['x1', 'x2', 'x3'],
            target_metric='d1',
            metrics=dict(
                d1=GridSearchMetricConfiguration(
                    metric_class='deepsysid.pipeline.metrics.IndexOfAgreementMetric',
                    parameters=dict(j=1),
                )
            ),
            additional_tests=dict(),
        ),
        models=[
            ModelGridSearchTemplate(
                model_base_name='LSTM',
                model_class='deepsysid.models.recurrent.LSTMInitModel',
                static_parameters=dict(
                    dropout=0.25,
                    sequence_length=50,
                    batch_size=32,
                    epochs_initializer=100,
                    epochs_predictor=200,
                ),
                flexible_parameters=dict(
                    recurrent_dim=[1, 64, 128],
                    num_recurrent_layers=[1, 2],
                    loss=['mse', 'msge'],
                    learning_rate=[0.01, 0.02],
                ),
            )
        ],
    )

    config = ExperimentConfiguration.from_grid_search_template(
        template, device_name='cpu'
    )
    assert len(config.models) == (3 * 2 * 2 * 2)
