import abc
import logging
from typing import Dict, Generic, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel
from sklearn.base import BaseEstimator
from sklearn.metrics import r2_score

from ..networks.rnn import HiddenStateForwardModule
from . import utils

logger = logging.getLogger(__name__)


class DynamicIdentificationModelConfig(BaseModel):
    device_name: str = 'cpu'
    control_names: List[str]
    state_names: List[str]
    initial_state_names: Optional[List[str]]
    time_delta: float


class DynamicIdentificationModel(metaclass=abc.ABCMeta):
    CONFIG: Type[DynamicIdentificationModelConfig] = DynamicIdentificationModelConfig

    @abc.abstractmethod
    def __init__(self, config: Optional[DynamicIdentificationModelConfig]) -> None:
        pass

    @abc.abstractmethod
    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]],
    ) -> Optional[Dict[str, NDArray[np.float64]]]:
        pass

    @abc.abstractmethod
    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        x0: Optional[NDArray[np.float64]],
    ) -> Union[
        NDArray[np.float64], Tuple[NDArray[np.float64], Dict[str, NDArray[np.float64]]]
    ]:
        pass

    @abc.abstractmethod
    def save(self, file_path: Tuple[str, ...]) -> None:
        pass

    @abc.abstractmethod
    def load(self, file_path: Tuple[str, ...]) -> None:
        pass

    @abc.abstractmethod
    def get_file_extension(self) -> Tuple[str, ...]:
        pass

    @abc.abstractmethod
    def get_parameter_count(self) -> int:
        pass


BaseRegressor = TypeVar('BaseRegressor', bound=BaseEstimator)


class FixedWindowModel(
    DynamicIdentificationModel, Generic[BaseRegressor], metaclass=abc.ABCMeta
):
    def __init__(self, window_size: int, regressor: BaseRegressor) -> None:
        assert window_size >= 1
        super().__init__(None)

        self.window_size = window_size
        self.regressor = regressor

        self.control_mean: Optional[NDArray[np.float64]] = None
        self.control_stddev: Optional[NDArray[np.float64]] = None
        self.state_mean: Optional[NDArray[np.float64]] = None
        self.state_stddev: Optional[NDArray[np.float64]] = None

    def train(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_seqs: Optional[List[NDArray[np.float64]]],
    ) -> None:
        assert len(control_seqs) == len(state_seqs)
        assert control_seqs[0].shape[0] == state_seqs[0].shape[0]

        self.control_mean, self.control_stddev = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_stddev = utils.mean_stddev(state_seqs)

        # Prepare training data
        train_x_list, train_y_list = [], []
        for i in range(len(control_seqs)):
            control = utils.normalize(
                control_seqs[i], self.control_mean, self.control_stddev
            )
            state = utils.normalize(state_seqs[i], self.state_mean, self.state_stddev)

            x, y = self.transform_to_single_step_training_data(control, state)
            x = self._map_regressor_input(x)

            train_x_list.append(x)
            train_y_list.append(y)

        train_x = np.vstack(train_x_list)
        train_y = np.vstack(train_y_list)

        self.regressor.fit(train_x, train_y)
        r2_fit = r2_score(
            self.regressor.predict(train_x), train_y, multioutput='uniform_average'
        )
        logger.info(f'R2 Score: {r2_fit}')

    def simulate(
        self,
        initial_control: NDArray[np.float64],
        initial_state: NDArray[np.float64],
        control: NDArray[np.float64],
        x0: Optional[NDArray[np.float64]],
    ) -> NDArray[np.float64]:
        """
        Multi-step prediction of system states given control inputs and initial window.

        Arguments:
            initial_control - Control inputs for first window_size steps
            initial_state - System states for first window_size steps
            control - Control inputs of arbitrary length

        Returns:
            System states for each given control input
        """
        assert initial_control.shape[0] >= self.window_size
        assert initial_state.shape[0] >= self.window_size

        if (
            self.state_mean is None
            or self.state_stddev is None
            or self.control_mean is None
            or self.control_stddev is None
        ):
            raise ValueError('Model has not been trained and cannot simulate.')

        full_dim = initial_control.shape[1] + initial_state.shape[1]

        initial_control = utils.normalize(
            initial_control, self.control_mean, self.control_stddev
        )
        initial_state = utils.normalize(
            initial_state, self.state_mean, self.state_stddev
        )
        control = utils.normalize(control, self.control_mean, self.control_stddev)

        pred_states_list = []
        window = np.hstack(
            (initial_control[-self.window_size :], initial_state[-self.window_size :])
        ).flatten()
        for i in range(control.shape[0]):
            x = np.concatenate((window, control[i])).reshape(1, -1)
            x = self._map_regressor_input(x)

            state = np.squeeze(self.regressor.predict(x))
            pred_states_list.append(state)
            window = np.concatenate((window[full_dim:], control[i], state))

        pred_states = utils.denormalize(
            np.array(pred_states_list, np.float64), self.state_mean, self.state_stddev
        )

        return pred_states

    def _map_regressor_input(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """
        Apply any transformations, e.g. nonlinearities, to input of regressor.
        Arguments:
            x - each row has shape [u(t-W) x(t-W) ... u(t-1) x(t-1) u(t)]
            with window size W
        """
        return x

    def transform_to_single_step_training_data(
        self, control: NDArray[np.float64], state: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        full_dim = control.shape[1] + state.shape[1]
        control_dim = control.shape[1]

        rows = np.hstack((control, state))
        windows = self.sliding_window(rows)
        # x = [u(t-W) x(t-W) ... u(t-1) x(t-1) u(t)]
        x = windows[:, : full_dim * self.window_size + control_dim]
        # y = x(t)
        y = windows[:, full_dim * self.window_size + control_dim :]

        return x, y

    def sliding_window(self, arr: NDArray[np.float64]) -> NDArray[np.float64]:
        # non-overlapping windows
        width = self.window_size + 1
        return np.hstack([arr[i : 1 + i - width or None : width] for i in range(width)])


class NormalizedControlStateModel(DynamicIdentificationModel, metaclass=abc.ABCMeta):
    def __init__(self, config: DynamicIdentificationModelConfig):
        super().__init__(config)

        self._state_mean: Optional[NDArray[np.float64]] = None
        self._state_std: Optional[NDArray[np.float64]] = None
        self._control_mean: Optional[NDArray[np.float64]] = None
        self._control_std: Optional[NDArray[np.float64]] = None

    @property
    def state_mean(self) -> NDArray[np.float64]:
        if self._state_mean is None:
            raise ValueError('Model is not trained and has no computed state_mean.')
        return self._state_mean

    @property
    def state_std(self) -> NDArray[np.float64]:
        if self._state_std is None:
            raise ValueError('Model is not trained and has no computed state_std.')
        return self._state_std

    @property
    def control_mean(self) -> NDArray[np.float64]:
        if self._control_mean is None:
            raise ValueError('Model is not trained and has no computed control_mean')
        return self._control_mean

    @property
    def control_std(self) -> NDArray[np.float64]:
        if self._control_std is None:
            raise ValueError('Model is not trained and has no computed control_std.')
        return self._control_std


class NormalizedHiddenStateInitializerPredictorModel(
    NormalizedControlStateModel, metaclass=abc.ABCMeta
):
    @property
    @abc.abstractmethod
    def initializer(self) -> HiddenStateForwardModule:
        pass

    @property
    @abc.abstractmethod
    def predictor(self) -> HiddenStateForwardModule:
        pass
