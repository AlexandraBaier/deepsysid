import itertools
import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from numpy.typing import NDArray
from sklearn.model_selection import cross_val_score

from ...tracker.base import BaseEventTracker
from ..base import DynamicIdentificationModelConfig
from .kernel import (
    DiagonalCorrelatedKernel,
    Kernel,
    RidgeKernel,
    StableSplineKernel,
    TunedCorrelationKernel,
)
from .regression import (
    BaseKernelRegressionModel,
    KernelRegression,
    construct_fit_input_arguments,
)


class SingleKernelRegressionCVModelConfig(DynamicIdentificationModelConfig):
    window_size: int
    folds: int
    repeats: int
    hyperparameter_grid: Dict[str, List[float]]


class SingleKernelRegressionCVModel(BaseKernelRegressionModel):
    CONFIG = SingleKernelRegressionCVModelConfig

    def __init__(
        self, config: SingleKernelRegressionCVModelConfig, kernel_class: Type[Kernel]
    ):
        super().__init__()

        self.input_size = len(config.control_names)
        self.output_size = len(config.state_names)
        self.window_size = config.window_size
        self.folds = config.folds
        self.repeats = config.repeats
        self.hyperparameter_grid = config.hyperparameter_grid
        self.kernel_class = kernel_class

        self._regressor: Optional[KernelRegression] = None
        self._best_hyperparameters: Optional[Dict[str, float]] = None

    @property
    def regressor(self) -> KernelRegression:
        if self._regressor is None:
            raise ValueError(
                'Regressor has not been trained. Call train_kernel_regressor first.'
            )
        return self._regressor

    def map_input(self, control: NDArray[np.float64]) -> NDArray[np.float64]:
        return control

    def train_kernel_regressor(
        self,
        normalized_control_seqs: List[NDArray[np.float64]],
        normalized_state_seqs: List[NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:
        x, y = construct_fit_input_arguments(
            control_seqs=normalized_control_seqs,
            state_seqs=normalized_state_seqs,
            input_window_size=self.window_size,
            output_window_size=self.window_size,
        )

        parameter_score: Dict[Tuple[Tuple[str, float], ...], float] = dict()
        parameter_scores: Dict[
            Tuple[Tuple[str, float], ...], List[float]
        ] = defaultdict(list)
        for hyperparameter_tuple in itertools.product(
            *[
                [(name, param) for param in params]
                for name, params in self.hyperparameter_grid.items()
            ]
        ):
            hyperparameter_dict = dict(hyperparameter_tuple)
            score = 0.0
            for repeat in range(self.repeats):
                regressor = self._construct_regressor(hyperparameter_dict)
                # Higher score is better, as it is the negative MSE.
                cv_scores = cross_val_score(
                    regressor, X=x, y=y, scoring='neg_mean_squared_error', cv=self.folds
                )
                cv_score = sum(cv_scores) / self.folds
                score += np.mean(cv_score)
                parameter_scores[hyperparameter_tuple].append(cv_score)

            parameter_score[hyperparameter_tuple] = score / self.repeats

        best_hyperparameter, best_score = max(
            parameter_score.items(), key=lambda t: t[1]
        )
        self._best_hyperparameters = dict(best_hyperparameter)
        self._regressor = self._construct_regressor(self._best_hyperparameters)
        self._regressor.fit(x, y)

        return dict(
            best_hyperparameter=np.array(
                [self._extract_sorted_hyperparameters(best_hyperparameter)],
                dtype=np.float64,
            ),
            scores=np.array(
                [
                    self._extract_sorted_hyperparameters(param) + scores
                    for param, scores in parameter_scores.items()
                ]
            ),
        )

    def save(
        self, file_path: Tuple[str, ...], tracker: BaseEventTracker = BaseEventTracker()
    ) -> None:
        offset = len(super().get_file_extension())
        super().save(file_path[:offset])

        if self._best_hyperparameters is None:
            raise ValueError('Model is not trained yet, call .train first.')

        with open(file_path[offset:][0], mode='w') as f:
            json.dump(self._best_hyperparameters, f)

    def load(self, file_path: Tuple[str, ...]) -> None:
        offset = len(super().get_file_extension())
        with open(file_path[offset:][0], mode='r') as f:
            json_dict = json.load(f)
        self._best_hyperparameters = json_dict
        self._regressor = self._construct_regressor(json_dict)

        super().load(file_path[:offset])

    def get_file_extension(self) -> Tuple[str, ...]:
        return super().get_file_extension() + ('json',)

    def hyperparameter_names(self) -> List[str]:
        return sorted(self.hyperparameter_grid.keys())

    def _construct_regressor(
        self, hyperparameters: Dict[str, float]
    ) -> KernelRegression:
        kernel_hyperparams = self.kernel_class.HYPERPARAMETER(**hyperparameters)
        return KernelRegression(
            input_dimension=self.input_size,
            output_dimension=self.output_size,
            input_window_size=self.window_size,
            output_window_size=self.window_size,
            input_kernels=[
                self.kernel_class(kernel_hyperparams) for _ in range(self.input_size)
            ],
            output_kernels=[
                self.kernel_class(kernel_hyperparams) for _ in range(self.output_size)
            ],
            ignore_kernel=False,
        )

    @staticmethod
    def _extract_sorted_hyperparameters(
        param_tuple: Tuple[Tuple[str, float], ...]
    ) -> List[float]:
        return [value for _, value in sorted(param_tuple, key=lambda t: t[0])]


class RidgeKernelRegressionCVModel(SingleKernelRegressionCVModel):
    def __init__(self, config: SingleKernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=RidgeKernel)


class DiagonalCorrelatedKernelRegressionCVModel(SingleKernelRegressionCVModel):
    def __init__(self, config: SingleKernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=DiagonalCorrelatedKernel)


class TunedCorrelationKernelRegressionCVModel(SingleKernelRegressionCVModel):
    def __init__(self, config: SingleKernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=TunedCorrelationKernel)


class StableSplineKernelRegressionCVModel(SingleKernelRegressionCVModel):
    def __init__(self, config: SingleKernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=StableSplineKernel)
