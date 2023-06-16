from warnings import warn

from ..metrics.error import (
    MeanAbsoluteErrorMetric,
    MeanSquaredErrorMetric,
    NormalizedRootMeanSquaredErrorMetric,
    RootMeanSquaredErrorMetric,
)
from ..metrics.score import IndexOfAgreementMetric, IndexOfAgreementMetricConfig
from ..metrics.trajectory import (
    Trajectory4DOFRootMeanSquaredErrorMetric,
    TrajectoryNED6DOFRootMeanSquaredErrorMetric,
)

warn(
    message=(
        'deepsysid.pipeline.metrics is deprecated. '
        'Import metrics from deepsysid.metrics instead. '
        'deepsysid.pipeline.metrics will be removed in a future release.'
    ),
    category=DeprecationWarning,
)

__all__ = [
    'Trajectory4DOFRootMeanSquaredErrorMetric',
    'TrajectoryNED6DOFRootMeanSquaredErrorMetric',
    'MeanAbsoluteErrorMetric',
    'MeanSquaredErrorMetric',
    'RootMeanSquaredErrorMetric',
    'NormalizedRootMeanSquaredErrorMetric',
    'IndexOfAgreementMetricConfig',
    'IndexOfAgreementMetric',
]
