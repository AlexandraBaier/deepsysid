from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from .base import BaseMetric, validate_measure_arguments


class MeanAbsoluteErrorMetric(BaseMetric):
    @validate_measure_arguments
    def measure(
        self,
        y_true: List[NDArray[np.float64]],
        y_pred: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        y_true_np = np.vstack(y_true)
        y_pred_np = np.vstack(y_pred)

        mae_per_var = np.mean(np.abs(y_true_np - y_pred_np), axis=0)
        mae = np.mean(mae_per_var)
        return mae_per_var, dict(
            mae=mae,
        )


class MeanSquaredErrorMetric(BaseMetric):
    @validate_measure_arguments
    def measure(
        self,
        y_true: List[NDArray[np.float64]],
        y_pred: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        y_true_np = np.vstack(y_true)
        y_pred_np = np.vstack(y_pred)

        mse_per_var = np.mean((y_true_np - y_pred_np) ** 2, axis=0)
        mse = np.mean(mse_per_var)
        return mse_per_var, dict(
            mse=mse,
        )


class RootMeanSquaredErrorMetric(BaseMetric):
    @validate_measure_arguments
    def measure(
        self,
        y_true: List[NDArray[np.float64]],
        y_pred: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        y_true_np = np.vstack(y_true)
        y_pred_np = np.vstack(y_pred)

        mse_per_var = np.mean((y_true_np - y_pred_np) ** 2, axis=0)
        rmse_per_var = np.sqrt(mse_per_var)
        rmse = np.mean(rmse_per_var)
        return rmse_per_var, dict(
            rmse=rmse,
            mse_per_var=mse_per_var,
        )


class NormalizedRootMeanSquaredErrorMetric(BaseMetric):
    @validate_measure_arguments
    def measure(
        self,
        y_true: List[NDArray[np.float64]],
        y_pred: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]:
        y_true_np = np.vstack(y_true)
        y_pred_np = np.vstack(y_pred)
        std = np.std(y_true_np, axis=0)

        mse_per_var = np.mean((y_true_np - y_pred_np) ** 2, axis=0)
        nrmse_per_var = 1.0 / std * np.sqrt(mse_per_var)
        nrmse = np.mean(nrmse_per_var)
        return nrmse_per_var, dict(nrmse=nrmse, mse_per_var=mse_per_var, std=std)
