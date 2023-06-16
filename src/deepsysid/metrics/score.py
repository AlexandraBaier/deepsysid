from typing import Dict, List, Tuple, Type

import numpy as np
from numpy.typing import NDArray

from .base import BaseMetric, BaseMetricConfig, validate_measure_arguments


class IndexOfAgreementMetricConfig(BaseMetricConfig):
    j: int


class IndexOfAgreementMetric(BaseMetric):
    """
    Willmott, C.J. (1981). On the validation of models.
    Physical Geography, 2(2), 184–194.

    Willmott, C.J., Ackleson, S.G., Davis, R.E., Feddema,
    J.J., Klink, K.M., Legates, D.R., O’Donnell, J., and
    Rowe, C.M. (1985). Statistics for the evaluation and
    comparison of models. Journal of Geophysical Research:
    Oceans, 90(C5), 8995–9005.
    """

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
        y_true_np = np.vstack(y_true)
        y_pred_np = np.vstack(y_pred)
        y_true_mean = np.mean(y_true_np, axis=0)

        numerator = np.sum(np.abs(y_true_np - y_pred_np) ** self.j, axis=0)
        denominator = np.sum(
            (np.abs(y_true_np - y_true_mean) + np.abs(y_pred_np - y_true_mean))
            ** self.j,
            axis=0,
        )
        dj_score: NDArray[np.float64] = 1.0 - numerator / denominator

        return dj_score, {f'd{self.j}': np.array([np.mean(dj_score)], dtype=np.float64)}


class RefinedIndexOfAgreementMetric(BaseMetric):
    """
    Willmott CJ, Robeson SM, Matsuura K (2012) A refined index of
    model performance. International Journal of Climatology 32: 2088–2094.

    Willmott CJ, Robeson SM, Matsuura K, Ficklin DL (2015)
    Assessment of three dimensionless measures of model performance.
    Environmental Modelling & Software 73: 167–174.

    Calculation from
    Li J (2017) Assessing the accuracy of predictive models for numerical
    data: Not r nor r2, why not? Then what? PLoS ONE 12(8): e0183250.
    https://doi.org/10.1371/journal.pone.0183250
    """

    @validate_measure_arguments
    def measure(
        self, y_true: List[NDArray[np.float64]], y_pred: List[NDArray[np.float64]]
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        y_true_np = np.vstack(y_true)
        y_pred_np = np.vstack(y_pred)
        y_true_mean = np.mean(y_true_np, axis=0)

        numerator = np.sum((y_true_np - y_pred_np) ** 2, axis=0)
        denominator = np.sum(y_true_np - y_true_mean**2, axis=0)
        option1: NDArray[np.float64] = 1.0 - numerator / denominator
        option2 = denominator / numerator - 1
        switch = (option1 >= 0).astype(np.float64)
        dr = switch * option1 + (1.0 - switch) * option2

        return dr, dict(dr=np.array([np.mean(dr)], dtype=np.float64))


class FitRatioMetric(BaseMetric):
    """
    Also called Variance Explained by
    Li J (2017) Assessing the accuracy of predictive models for
    numerical data: Not r nor r2, why not? Then what? PLoS ONE 12(8):
    e0183250. https://doi.org/10.1371/journal.pone.0183250
    """

    @validate_measure_arguments
    def measure(
        self, y_true: List[NDArray[np.float64]], y_pred: List[NDArray[np.float64]]
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        y_true_np = np.vstack(y_true)
        y_pred_np = np.vstack(y_pred)
        y_true_mean = np.mean(y_true_np, axis=0)

        numerator = np.sqrt(np.sum((y_true_np - y_pred_np) ** 2, axis=0))
        denominator = np.sqrt(np.sum(y_true_np - y_true_mean**2, axis=0))
        fit: NDArray[np.float64] = 1.0 - numerator / denominator
        return fit, dict(
            fit_ratio=np.array([np.mean(fit)], dtype=np.float64),
            fit_ratio_percentage=np.array([np.mean(fit) * 100.0], dtype=np.float64),
        )


class EfficiencyMetric(BaseMetric):
    """
    Nash, J. and Sutcliffe, J. (1970). River flow forecasting
    through conceptual models part I - a discussion of
    principles. Journal of Hydrology, 10(3), 282–290.

    Legates, D.R. and McCabe, G.J. (1999). Evaluating the use of
    “goodness-of-fit” measures in hydrologic and hydroclimatic
    model validation. Water Resources Research, 35(1), 233–241.
    """

    @validate_measure_arguments
    def measure(
        self, y_true: List[NDArray[np.float64]], y_pred: List[NDArray[np.float64]]
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        y_true_np = np.vstack(y_true)
        y_pred_np = np.vstack(y_pred)
        y_true_mean = np.mean(y_true_np, axis=0)

        numerator = np.sum((y_true_np - y_pred_np) ** 2, axis=0)
        denominator = np.sum(y_true_np - y_true_mean**2, axis=0)
        efficiency: NDArray[np.float64] = 1.0 - numerator / denominator
        return efficiency, dict(
            efficiency=np.array([np.mean(efficiency)], dtype=np.float64)
        )


class RefinedEfficiencyMetric(BaseMetric):
    """
    Legates DR, McCabe GJ (2013) A refined index of model performance: a rejoinder.
    International Journal of Climatology 33: 1053–1056.
    """

    @validate_measure_arguments
    def measure(
        self, y_true: List[NDArray[np.float64]], y_pred: List[NDArray[np.float64]]
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        y_true_np = np.vstack(y_true)
        y_pred_np = np.vstack(y_pred)
        y_true_mean = np.mean(y_true_np, axis=0)

        numerator = np.sum(np.abs(y_true_np - y_pred_np), axis=0)
        denominator = np.sum(np.abs(y_true_np - y_true_mean), axis=0)
        efficiency: NDArray[np.float64] = 1.0 - numerator / denominator
        return efficiency, dict(
            efficiency=np.array([np.mean(efficiency)], dtype=np.float64)
        )


class PearsonProductMomentCorrelationCoefficientMetric(BaseMetric):
    def measure(
        self, y_true: List[NDArray[np.float64]], y_pred: List[NDArray[np.float64]]
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        y_true_np = np.vstack(y_true)
        y_pred_np = np.vstack(y_pred)
        y_true_mean = np.mean(y_true_np, axis=0)
        y_pred_mean = np.mean(y_pred_np, axis=0)

        numerator = np.sum(
            (y_true_np - y_true_mean) * (y_pred_np - y_pred_mean), axis=0
        )
        denominator_1 = np.sum((y_true_np - y_true_mean) ** 2, axis=0)
        denominator_2 = np.sum((y_pred_np - y_pred_mean) ** 2, axis=0)
        denominator = np.sqrt(denominator_1 * denominator_2)
        correlation = numerator / denominator

        return correlation, dict(
            correlation=np.array([np.mean(correlation)], dtype=np.float64)
        )
