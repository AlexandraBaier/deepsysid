import numpy as np
import pytest

from deepsysid.pipeline.metrics import (
    IndexOfAgreementMetric,
    IndexOfAgreementMetricConfig,
)


def test_index_of_agreement_empty_arrays():
    metric = IndexOfAgreementMetric(
        IndexOfAgreementMetricConfig(state_names=['1', '2'], sample_time=1.0, j=1)
    )

    a1 = np.array([])
    a2 = np.array([])

    with pytest.raises(ValueError):
        metric.measure([a1], [a2])


def test_index_of_agreement_mismatch_arrays():
    metric = IndexOfAgreementMetric(
        IndexOfAgreementMetricConfig(state_names=['1', '2'], sample_time=1.0, j=1)
    )

    a1 = np.array([[1, 2], [3, 4]])
    a2 = np.array([[1, 2], [3, 4], [5, 6]])

    with pytest.raises(ValueError):
        metric.measure([a1], [a2])


def test_index_of_agreement_incorrect_exponent():
    with pytest.raises(ValueError):
        IndexOfAgreementMetric(
            IndexOfAgreementMetricConfig(state_names=['1', '2'], sample_time=1.0, j=0)
        )


def test_index_of_agreement_perfect_agreement():
    metric = IndexOfAgreementMetric(
        IndexOfAgreementMetricConfig(state_names=['1', '2'], sample_time=1.0, j=1)
    )

    a1 = np.array([[1, 2], [3, 4]])
    a2 = np.array([[1, 2], [3, 4]])

    assert np.average(metric.measure([a1], [a2])[0]) == pytest.approx(1.0)
