import abc
import logging
from typing import List, Optional, Tuple, Type, Union, Dict

import numpy as np
from pydantic import BaseModel
from sklearn.metrics import r2_score

from .. import utils

logger = logging.getLogger()


class DynamicIdentificationModelConfig(BaseModel):
    device_name: str = 'cpu'
    control_names: List[str]
    state_names: List[str]
    time_delta: float


class DynamicIdentificationModel(metaclass=abc.ABCMeta):
    CONFIG: Type[DynamicIdentificationModelConfig] = DynamicIdentificationModelConfig

    @abc.abstractmethod
    def __init__(self, config: Optional[DynamicIdentificationModelConfig]):
        pass

    @abc.abstractmethod
    def train(self, control_seqs: List[np.ndarray], state_seqs: List[np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        pass

    @abc.abstractmethod
    def simulate(
        self,
        initial_control: np.ndarray,
        initial_state: np.ndarray,
        control: np.ndarray,
    ) -> Union[np.ndarray, Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        pass

    @abc.abstractmethod
    def save(self, file_path: Tuple[str, ...]):
        pass

    @abc.abstractmethod
    def load(self, file_path: Tuple[str, ...]):
        pass

    @abc.abstractmethod
    def get_file_extension(self) -> Tuple[str, ...]:
        pass

    @abc.abstractmethod
    def get_parameter_count(self) -> int:
        pass


class FixedWindowModel(DynamicIdentificationModel, metaclass=abc.ABCMeta):
    def __init__(self, window_size: int, regressor):
        assert window_size >= 1
        super().__init__(None)

        self.window_size = window_size
        self.regressor = regressor

        self.control_mean: Optional[np.ndarray] = None
        self.control_stddev: Optional[np.ndarray] = None
        self.state_mean: Optional[np.ndarray] = None
        self.state_stddev: Optional[np.ndarray] = None

    def train(self, control_seqs: List[np.ndarray], state_seqs: List[np.ndarray]):
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

            x, y = utils.transform_to_single_step_training_data(
                control, state, self.window_size
            )
            x = self._map_regressor_input(x)

            train_x_list.append(x)
            train_y_list.append(y)

        train_x = np.vstack(train_x_list)
        train_y = np.vstack(train_y_list)

        self.regressor = self.regressor.fit(train_x, train_y)
        r2_fit = r2_score(
            self.regressor.predict(train_x), train_y, multioutput='uniform_average'
        )
        logger.info(f'R2 Score: {r2_fit}')

    def simulate(
        self,
        initial_control: np.ndarray,
        initial_state: np.ndarray,
        control: np.ndarray,
    ) -> np.ndarray:
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
            np.array(pred_states_list), self.state_mean, self.state_stddev
        )

        return pred_states

    def _map_regressor_input(self, x: np.ndarray) -> np.ndarray:
        """
        Apply any transformations, e.g. nonlinearities, to input of regressor.
        Arguments:
            x - each row has shape [u(t-W) x(t-W) ... u(t-1) x(t-1) u(t)]
            with window size W
        """
        return x
