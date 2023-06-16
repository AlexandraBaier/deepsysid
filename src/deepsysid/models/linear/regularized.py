import abc
import dataclasses
import itertools
import json
from typing import Dict, Iterator, List, Optional, Tuple, Type

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


@dataclasses.dataclass
class KernelHyperparameterContainer:
    input_hyperparameters: List[Dict[str, float]]
    output_hyperparameters: List[Dict[str, float]]

    def construct_regressor(
        self,
        kernel_class: Type[Kernel],
        input_window_size: int,
        output_window_size: int,
    ) -> KernelRegression:
        input_kernels = [
            kernel_class(kernel_class.HYPERPARAMETER(**param))
            for param in self.input_hyperparameters
        ]
        output_kernels = [
            kernel_class(kernel_class.HYPERPARAMETER(**param))
            for param in self.output_hyperparameters
        ]
        return KernelRegression(
            input_dimension=len(input_kernels),
            output_dimension=len(output_kernels),
            input_window_size=input_window_size,
            output_window_size=output_window_size,
            input_kernels=input_kernels,
            output_kernels=output_kernels,
            ignore_kernel=False,
        )

    def to_json_serializable_dict(self) -> Dict[str, List[Dict[str, float]]]:
        return dict(
            input_hyperparameters=self.input_hyperparameters,
            output_hyperparameters=self.output_hyperparameters,
        )

    @classmethod
    def from_dict(
        cls, json_dict: Dict[str, List[Dict[str, float]]]
    ) -> 'KernelHyperparameterContainer':
        return cls(
            input_hyperparameters=json_dict['input_hyperparameters'],
            output_hyperparameters=json_dict['output_hyperparameters'],
        )


class KernelRegressionCVModelConfig(DynamicIdentificationModelConfig):
    window_size: int
    folds: int
    repeats: int
    hyperparameter_grid: Dict[str, List[float]]


class KernelRegressionCVModel(BaseKernelRegressionModel, metaclass=abc.ABCMeta):
    CONFIG = KernelRegressionCVModelConfig

    def __init__(
        self, config: KernelRegressionCVModelConfig, kernel_class: Type[Kernel]
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
        self._best_hyperparameters: Optional[KernelHyperparameterContainer] = None
        self._grid_search_results: Optional[
            List[Tuple[KernelHyperparameterContainer, List[float]]]
        ] = None

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

        parameter_score: Dict[int, float] = dict()
        self._grid_search_results = []
        containers = []
        for container_idx, container in enumerate(self._generate_hyperparameter_grid()):
            score = 0.0
            for repeat in range(self.repeats):
                regressor = container.construct_regressor(
                    kernel_class=self.kernel_class,
                    input_window_size=self.window_size,
                    output_window_size=self.window_size,
                )
                # Higher score is better, as it is the negative MSE.
                cv_scores = cross_val_score(
                    regressor, X=x, y=y, scoring='neg_mean_squared_error', cv=self.folds
                )
                cv_score = sum(cv_scores) / self.folds
                score += np.mean(cv_score)
                self._grid_search_results.append((container, cv_score))

            parameter_score[container_idx] = score / self.repeats
            containers.append(container)

        best_hyperparameter_idx, best_score = max(
            parameter_score.items(), key=lambda t: t[1]
        )
        best_hyperparameter = containers[best_hyperparameter_idx]
        self._best_hyperparameters = best_hyperparameter
        self._regressor = best_hyperparameter.construct_regressor(
            kernel_class=self.kernel_class,
            input_window_size=self.window_size,
            output_window_size=self.window_size,
        )
        self._regressor.fit(x, y)

        return dict()

    def save(
        self, file_path: Tuple[str, ...], tracker: BaseEventTracker = BaseEventTracker()
    ) -> None:
        offset = len(super().get_file_extension())
        super().save(file_path[:offset])

        if self._best_hyperparameters is None or self._grid_search_results is None:
            raise ValueError('Model is not trained yet, call .train first.')

        with open(file_path[offset:][0], mode='w') as f:
            json.dump(self._best_hyperparameters.to_json_serializable_dict(), f)

        with open(file_path[offset:][1], mode='w') as f:
            json.dump(
                [
                    [container.to_json_serializable_dict(), scores]
                    for container, scores in self._grid_search_results
                ],
                f,
            )

    def load(self, file_path: Tuple[str, ...]) -> None:
        offset = len(super().get_file_extension())
        with open(file_path[offset:][0], mode='r') as f:
            hyperparameters = KernelHyperparameterContainer.from_dict(json.load(f))
        self._best_hyperparameters = hyperparameters
        self._regressor = hyperparameters.construct_regressor(
            kernel_class=self.kernel_class,
            input_window_size=self.window_size,
            output_window_size=self.window_size,
        )

        with open(file_path[offset:][1], mode='r') as f:
            results_list = json.load(f)

        self._grid_search_results = [
            (KernelHyperparameterContainer.from_dict(container_dict), scores)
            for container_dict, scores in results_list
        ]

        super().load(file_path[:offset])

    def get_file_extension(self) -> Tuple[str, ...]:
        return super().get_file_extension() + ('hyperparameter.json', 'gridsearch.json')

    @abc.abstractmethod
    def _generate_hyperparameter_grid(self) -> Iterator[KernelHyperparameterContainer]:
        pass


class SingleKernelRegressionCVModel(KernelRegressionCVModel):
    def _generate_hyperparameter_grid(self) -> Iterator[KernelHyperparameterContainer]:
        for hyperparameter_tuple in itertools.product(
            *[
                [(name, param) for param in params]
                for name, params in self.hyperparameter_grid.items()
            ]
        ):
            input_parameters = [
                dict(hyperparameter_tuple) for _ in range(self.input_size)
            ]
            output_parameters = [
                dict(hyperparameter_tuple) for _ in range(self.output_size)
            ]
            yield KernelHyperparameterContainer(
                input_hyperparameters=input_parameters,
                output_hyperparameters=output_parameters,
            )


class RidgeKernelRegressionCVModel(SingleKernelRegressionCVModel):
    def __init__(self, config: KernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=RidgeKernel)


class DiagonalCorrelatedKernelRegressionCVModel(SingleKernelRegressionCVModel):
    def __init__(self, config: KernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=DiagonalCorrelatedKernel)


class TunedCorrelationKernelRegressionCVModel(SingleKernelRegressionCVModel):
    def __init__(self, config: KernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=TunedCorrelationKernel)


class StableSplineKernelRegressionCVModel(SingleKernelRegressionCVModel):
    def __init__(self, config: KernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=StableSplineKernel)


class InputOutputKernelRegressionCVModel(KernelRegressionCVModel):
    def _generate_hyperparameter_grid(self) -> Iterator[KernelHyperparameterContainer]:
        grid_line = [
            [(name, param) for param in params]
            for name, params in self.hyperparameter_grid.items()
        ]
        for input_hyperparameter_tuple in itertools.product(*grid_line):
            input_parameters = [
                dict(input_hyperparameter_tuple) for _ in range(self.input_size)
            ]
            for output_hyperparameter_tuple in itertools.product(*grid_line):
                output_parameters = [
                    dict(output_hyperparameter_tuple) for _ in range(self.output_size)
                ]
                yield KernelHyperparameterContainer(
                    input_hyperparameters=input_parameters,
                    output_hyperparameters=output_parameters,
                )


class InputOutputRidgeKernelRegressionCVModel(InputOutputKernelRegressionCVModel):
    def __init__(self, config: KernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=RidgeKernel)


class InputOutputDiagonalCorrelatedKernelRegressionCVModel(
    InputOutputKernelRegressionCVModel
):
    def __init__(self, config: KernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=DiagonalCorrelatedKernel)


class InputOutputTunedCorrelationKernelRegressionCVModel(
    InputOutputKernelRegressionCVModel
):
    def __init__(self, config: KernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=TunedCorrelationKernel)


class InputOutputStableSplineKernelRegressionCVModel(
    InputOutputKernelRegressionCVModel
):
    def __init__(self, config: KernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=StableSplineKernel)


class MultiKernelRegressionCVModel(KernelRegressionCVModel):
    """
    WARNING
    This model searches for individual kernel hyperparameters for
    each input and output variable. Accordingly, the hyperparameter search space
    using grid search explodes. Unless you have very few input and output
    variables (less than 5 in total), this will not be worth your time.
    """

    def _generate_hyperparameter_grid(self) -> Iterator[KernelHyperparameterContainer]:
        # A grid line is for example [(c, 0.1), (c, 0.2)],
        # which was constructed from the dictionary entry
        # 'c': [0.1, 0.2]
        # The length of grid lines is the number of hyperparameters.
        # The length of each grid line depends on the number of
        # choices configured for each hyperparameter.
        parameter_grid_lines = [
            [(name, param) for param in params]
            for name, params in self.hyperparameter_grid.items()
        ]
        # Each input and output variable can be individually configured.
        parameter_grid_lines_per_input = [
            grid_line
            for _ in range(self.input_size)
            for grid_line in parameter_grid_lines
        ]
        parameter_grid_lines_per_output = [
            grid_line
            for _ in range(self.output_size)
            for grid_line in parameter_grid_lines
        ]

        # All input/output parameters are stored in a single list
        # of length #hyperparameters * input_size or
        # #hyperparameters * output_size. The parameters are ordered
        # per variable, so the first #hyperparameters values are the
        # parameters for the first variable's kernel.
        n_hyperparameters = len(self.hyperparameter_grid)
        for input_kernel_parameters in itertools.product(
            *parameter_grid_lines_per_input
        ):
            input_parameters = [
                dict(input_kernel_parameters[idx : idx + n_hyperparameters])
                for idx in range(0, len(input_kernel_parameters), n_hyperparameters)
            ]
            for output_kernel_parameters in itertools.product(
                *parameter_grid_lines_per_output
            ):
                output_parameters = [
                    dict(output_kernel_parameters[idx : idx + n_hyperparameters])
                    for idx in range(
                        0, len(output_kernel_parameters), n_hyperparameters
                    )
                ]
                yield KernelHyperparameterContainer(
                    input_hyperparameters=input_parameters,
                    output_hyperparameters=output_parameters,
                )


class MultiRidgeKernelRegressionCVModel(MultiKernelRegressionCVModel):
    def __init__(self, config: KernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=RidgeKernel)


class MultiDiagonalCorrelatedKernelRegressionCVModel(MultiKernelRegressionCVModel):
    def __init__(self, config: KernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=DiagonalCorrelatedKernel)


class MultiTunedCorrelationKernelRegressionCVModel(MultiKernelRegressionCVModel):
    def __init__(self, config: KernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=TunedCorrelationKernel)


class MultiStableSplineKernelRegressionCVModel(MultiKernelRegressionCVModel):
    def __init__(self, config: KernelRegressionCVModelConfig) -> None:
        super().__init__(config, kernel_class=StableSplineKernel)
