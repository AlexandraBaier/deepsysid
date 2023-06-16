from typing import Dict, List, Tuple, Type

import numpy as np
from numpy.typing import NDArray

from .base import BaseMetric, BaseMetricConfig, validate_measure_arguments


class IndexOfAgreementMetricConfig(BaseMetricConfig):
    j: int


class IndexOfAgreementMetric(BaseMetric):
    CONFIG: Type[BaseMetricConfig] = IndexOfAgreementMetricConfig

    def __init__(self, config: IndexOfAgreementMetricConfig):
        super().__init__(config)

        if config.j < 1:
            raise ValueError('Exponent j needs to be larger than 0.')

        self.j = config.j

    @validate_measure_arguments
    def measure(
        self,
        y_true: List[NDArray[np.float64]],
        y_pred: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        steps = [p.shape[0] for p in y_pred]
        scores = self.score_over_sequences(y_true, y_pred)
        return np.average(scores, weights=steps, axis=0), dict(per_step=scores)

    def score_over_sequences(
        self, true_seq: List[NDArray[np.float64]], pred_seq: List[NDArray[np.float64]]
    ) -> NDArray[np.float64]:
        score = np.zeros((len(pred_seq), pred_seq[0].shape[1]), dtype=np.float64)
        for i, (true, pred) in enumerate(zip(true_seq, pred_seq)):
            score[i, :] = self.score_per_sequence(true, pred)
        return score

    def score_per_sequence(
        self, y_true: NDArray[np.float64], y_pred: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        error_sum = np.sum(np.power(np.abs(y_true - y_pred), self.j), axis=0)
        partial_diff_true = np.abs(y_true - np.mean(y_true, axis=0))
        partial_diff_pred = np.abs(y_pred - np.mean(y_true, axis=0))
        partial_diff_sum = np.sum(
            np.power(partial_diff_true + partial_diff_pred, self.j), axis=0
        )
        score: NDArray[np.float64] = 1 - (error_sum / partial_diff_sum)
        return score
