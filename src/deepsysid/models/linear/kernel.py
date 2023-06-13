import abc

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel


class KernelHyperparameter(BaseModel):
    pass


class Kernel(metaclass=abc.ABCMeta):
    HYPERPARAMETER = KernelHyperparameter

    def __init__(self, eta: KernelHyperparameter) -> None:
        pass

    @abc.abstractmethod
    def construct(self, dimension: int) -> NDArray[np.float64]:
        pass


class ZeroKernel(Kernel):
    """
    WARNING: Only use ZeroKernel as a placeholder
        if you intend to set ignore_kernel=True in KernelRegression.
    ZeroKernel is a singular matrix, so solving the least-squares
        problem will fail, since the inverse of the kernel has to be computed.
    """

    def construct(self, dimension: int) -> NDArray[np.float64]:
        return np.zeros((dimension, dimension), dtype=np.float64)


class RidgeHyperparameter(KernelHyperparameter):
    c: float


class RidgeKernel(Kernel):
    HYPERPARAMETER = RidgeHyperparameter

    def __init__(self, eta: RidgeHyperparameter) -> None:
        super().__init__(eta)
        self.c = eta.c

    def construct(self, dimension: int) -> NDArray[np.float64]:
        kernel = self.c * np.eye(dimension)
        return kernel.astype(np.float64)


class DiagonalCorrelatedKernelHyperparameter(KernelHyperparameter):
    c: float
    lamb: float
    rho: float


class DiagonalCorrelatedKernel(Kernel):
    HYPERPARAMETER = DiagonalCorrelatedKernelHyperparameter

    def __init__(self, eta: DiagonalCorrelatedKernelHyperparameter) -> None:
        super().__init__(eta)
        self.c = eta.c
        self.lamb = eta.lamb
        self.rho = eta.lamb

    def construct(self, dimension: int) -> NDArray[np.float64]:
        col_idx = np.repeat([np.arange(dimension)], dimension, axis=0)
        row_idx = col_idx.T
        kernel: NDArray[np.float64] = self.c * (
            self.lamb ** ((row_idx + col_idx) / 2)
            * self.rho ** np.abs(row_idx - col_idx)
        )
        return kernel.astype(np.float64)


class TunedCorrelationKernelHyperparameter(KernelHyperparameter):
    c: float
    lamb: float


class TunedCorrelationKernel(Kernel):
    HYPERPARAMETER = TunedCorrelationKernelHyperparameter

    def __init__(self, eta: TunedCorrelationKernelHyperparameter) -> None:
        super().__init__(eta)
        self.kernel = DiagonalCorrelatedKernel(
            DiagonalCorrelatedKernelHyperparameter(
                c=eta.c, lamb=eta.lamb, rho=np.sqrt(eta.lamb)
            )
        )

    def construct(self, dimension: int) -> NDArray[np.float64]:
        return self.kernel.construct(dimension)


class StableSplineKernelHyperparameter(KernelHyperparameter):
    c: float
    lamb: float


class StableSplineKernel(Kernel):
    HYPERPARAMETER = StableSplineKernelHyperparameter

    def __init__(self, eta: StableSplineKernelHyperparameter) -> None:
        super().__init__(eta)
        self.c = eta.c
        self.lamb = eta.lamb

    def construct(self, dimension: int) -> NDArray[np.float64]:
        col_idx = np.repeat([np.arange(dimension)], dimension, axis=0)
        row_idx = col_idx.T
        kernel: NDArray[np.float64] = self.c * (
            self.lamb ** (row_idx + col_idx + np.maximum(row_idx, col_idx))
            - 1.0 / 3.0 * self.lamb ** (3 * np.maximum(row_idx, col_idx))
        )
        return kernel.astype(np.float64)


class FirstOrderStableSplineKernelHyperparameter(KernelHyperparameter):
    c: float
    lamb: float


class FirstOrderStableSplineKernel(Kernel):
    HYPERPARAMETER = FirstOrderStableSplineKernelHyperparameter

    def __init__(self, eta: FirstOrderStableSplineKernelHyperparameter) -> None:
        super().__init__(eta)
        self.c = eta.c
        self.lamb = eta.c

    def construct(self, dimension: int) -> NDArray[np.float64]:
        col_idx = np.repeat([np.arange(dimension)], dimension, axis=0)
        row_idx = col_idx.T
        kernel = self.c * self.lamb ** (np.maximum(row_idx, col_idx))
        return kernel.astype(np.float64)
