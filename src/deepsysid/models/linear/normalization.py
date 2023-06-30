import dataclasses
from typing import List

import numpy as np
from numpy._typing import NDArray

from deepsysid.models import utils


@dataclasses.dataclass
class StandardNormalizer:
    control_mean: NDArray[np.float64]
    control_std: NDArray[np.float64]
    state_mean: NDArray[np.float64]
    state_std: NDArray[np.float64]

    @classmethod
    def from_training_data(
        cls,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> 'StandardNormalizer':
        control_mean, control_std = utils.mean_stddev(control_seqs)
        state_mean, state_std = utils.mean_stddev(state_seqs)
        return cls(
            control_mean=control_mean,
            control_std=control_std,
            state_mean=state_mean,
            state_std=state_std,
        )

    def normalize_control(self, control: NDArray[np.float64]) -> NDArray[np.float64]:
        return utils.normalize(control, self.control_mean, self.control_std)

    def normalize_state(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return utils.normalize(state, self.state_mean, self.state_std)

    def denormalize_state(self, state: NDArray[np.float64]) -> NDArray[np.float64]:
        return utils.denormalize(state, self.state_mean, self.state_std)
