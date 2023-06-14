import json
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy._typing import NDArray
from sklearn.model_selection import cross_val_score

from ...tracker.base import BaseEventTracker
from ..base import DynamicIdentificationModelConfig
from .kernel import RidgeHyperparameter, RidgeKernel
from .regression import (
    BaseKernelRegressionModel,
    KernelRegression,
    construct_fit_input_arguments,
)


class RidgeRegressionCVModelConfig(DynamicIdentificationModelConfig):
    window_size: int
    folds: int
    repeats: int
    c_grid: List[float]


class RidgeRegressionCVModel(BaseKernelRegressionModel):
    CONFIG = RidgeRegressionCVModelConfig

    def __init__(self, config: RidgeRegressionCVModelConfig):
        super().__init__()

        self.input_size = len(config.control_names)
        self.output_size = len(config.state_names)
        self.window_size = config.window_size
        self.folds = config.folds
        self.repeats = config.repeats
        self.c_grid = config.c_grid
        self._regressor: Optional[KernelRegression] = None
        self._best_parameter: Optional[float] = None

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

        parameter_score = dict()
        parameter_scores: Dict[float, List[float]] = defaultdict(list)
        for c in self.c_grid:
            score = 0.0
            for repeat in range(self.repeats):
                regressor = self._construct_regressor(c)
                # Higher score is better, as it is the negative MSE.
                cv_scores = cross_val_score(
                    regressor, X=x, y=y, scoring='neg_mean_squared_error', cv=self.folds
                )
                cv_score = sum(cv_scores) / self.folds
                score += np.mean(cv_score)
                parameter_scores[c].append(cv_score)

            parameter_score[c] = score / self.repeats

        best_parameter, best_score = max(parameter_score.items(), key=lambda t: t[1])
        self._best_parameter = best_parameter
        self._regressor = self._construct_regressor(best_parameter)
        self._regressor.fit(x, y)

        return dict(
            best_hyperparameter=np.array([best_parameter], dtype=np.float64),
            scores=np.array(
                [[param] + scores for param, scores in parameter_scores.items()]
            ),
        )

    def save(
        self, file_path: Tuple[str, ...], tracker: BaseEventTracker = BaseEventTracker()
    ) -> None:
        offset = len(super().get_file_extension())
        super().save(file_path[:offset])

        with open(file_path[offset:][0], mode='w') as f:
            json.dump(dict(c=self._best_parameter), f)

    def load(self, file_path: Tuple[str, ...]) -> None:
        offset = len(super().get_file_extension())
        with open(file_path[offset:][0], mode='r') as f:
            json_dict = json.load(f)
        self._best_parameter = json_dict['c']
        self._regressor = self._construct_regressor(json_dict['c'])

        super().load(file_path[:offset])

    def get_file_extension(self) -> Tuple[str, ...]:
        return super().get_file_extension() + ('json',)

    def _construct_regressor(self, c: float) -> KernelRegression:
        return KernelRegression(
            input_dimension=self.input_size,
            output_dimension=self.output_size,
            input_window_size=self.window_size,
            output_window_size=self.window_size,
            input_kernels=[
                RidgeKernel(eta=RidgeHyperparameter(c=c))
                for _ in range(self.input_size)
            ],
            output_kernels=[
                RidgeKernel(eta=RidgeHyperparameter(c=c))
                for _ in range(self.output_size)
            ],
            ignore_kernel=False,
        )
