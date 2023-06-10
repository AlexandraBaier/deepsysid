from typing import Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray
from torch.utils import data


class FixedWindowDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
        self,
        window_size: int,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> None:
        self.window_size = window_size
        self.window_input, self.state_true = self.__load_dataset(
            control_seqs, state_seqs
        )

    def __load_dataset(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        window_input = []
        state_true = []
        for control, state in zip(control_seqs, state_seqs):
            for time in range(
                self.window_size, control.shape[0] - 1, int(self.window_size / 4) + 1
            ):
                window_input.append(
                    np.concatenate(
                        (
                            control[
                                time - self.window_size + 1 : time + 1, :
                            ].flatten(),
                            state[time - self.window_size : time, :].flatten(),
                        )
                    )
                )
                state_true.append(state[time + 1, :])

        return np.vstack(window_input), np.vstack(state_true)

    def __len__(self) -> int:
        return self.window_input.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return dict(
            window_input=self.window_input[idx], state_true=self.state_true[idx]
        )


class RecurrentInitializerDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        sequence_length: int,
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.x, self.y = self.__load_data(control_seqs, state_seqs)

    def __load_data(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        x_seq = list()
        y_seq = list()
        for control, state in zip(control_seqs, state_seqs):
            n_samples = int(
                (control.shape[0] - self.sequence_length - 1) / self.sequence_length
            )

            x = np.zeros(
                (n_samples, self.sequence_length, self.control_dim + self.state_dim),
                dtype=np.float64,
            )
            y = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )

            for idx in range(n_samples):
                time = idx * self.sequence_length

                x[idx, :, :] = np.hstack(
                    (
                        control[time + 1 : time + 1 + self.sequence_length, :],
                        state[time : time + self.sequence_length, :],
                    )
                )
                y[idx, :, :] = state[time + 1 : time + 1 + self.sequence_length, :]

            x_seq.append(x)
            y_seq.append(y)

        return np.vstack(x_seq), np.vstack(y_seq)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {'x': self.x[idx], 'y': self.y[idx]}


class RecurrentPredictorDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        sequence_length: int,
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.x0, self.y0, self.x, self.y = self.__load_data(control_seqs, state_seqs)

    def __load_data(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        x0_seq = list()
        y0_seq = list()
        x_seq = list()
        y_seq = list()

        for control, state in zip(control_seqs, state_seqs):
            n_samples = int(
                (control.shape[0] - 2 * self.sequence_length) / self.sequence_length
            )

            x0 = np.zeros(
                (n_samples, self.sequence_length, self.control_dim + self.state_dim),
                dtype=np.float64,
            )
            y0 = np.zeros((n_samples, self.state_dim), dtype=np.float64)
            x = np.zeros(
                (n_samples, self.sequence_length, self.control_dim), dtype=np.float64
            )
            y = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )

            for idx in range(n_samples):
                time = idx * self.sequence_length

                x0[idx, :, :] = np.hstack(
                    (
                        control[time + 1 : time + self.sequence_length + 1, :],
                        state[time : time + self.sequence_length, :],
                    )
                )
                y0[idx, :] = state[time + self.sequence_length - 1, :]
                x[idx, :, :] = control[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                y[idx, :, :] = state[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]

            x0_seq.append(x0)
            y0_seq.append(y0)
            x_seq.append(x)
            y_seq.append(y)

        return np.vstack(x0_seq), np.vstack(y0_seq), np.vstack(x_seq), np.vstack(y_seq)

    def __len__(self) -> int:
        return self.x0.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {
            'x0': self.x0[idx],
            'y0': self.y0[idx],
            'x': self.x[idx],
            'y': self.y[idx],
        }


class RecurrentPredictorInitialDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_state_seqs: List[NDArray[np.float64]],
        sequence_length: int,
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.initial_state_dim = initial_state_seqs[0].shape[1]
        self.wp_init, self.zp_init, self.wp, self.zp, self.x0 = self.__load_data(
            control_seqs, state_seqs, initial_state_seqs
        )

    def __load_data(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        wp_init_seq = list()
        zp_init_seq = list()
        wp_seq = list()
        zp_seq = list()
        x0_seq = list()

        for control, state, initial_state in zip(
            control_seqs, state_seqs, initial_state_seqs
        ):
            n_samples = int(
                (control.shape[0] - 2 * self.sequence_length) / self.sequence_length
            )

            wp_init = np.zeros(
                (n_samples, self.sequence_length, self.control_dim + self.state_dim),
                dtype=np.float64,
            )
            zp_init = np.zeros((n_samples, self.state_dim), dtype=np.float64)
            wp = np.zeros(
                (n_samples, self.sequence_length, self.control_dim), dtype=np.float64
            )
            zp = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )

            x0 = np.zeros(shape=(n_samples, self.initial_state_dim), dtype=np.float64)

            for idx in range(n_samples):
                time = idx * self.sequence_length

                wp_init[idx, :, :] = np.hstack(
                    (
                        control[time : time + self.sequence_length],
                        state[time : time + self.sequence_length, :],
                    )
                )
                zp_init[idx, :] = state[time + self.sequence_length - 1, :]
                wp[idx, :, :] = control[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                zp[idx, :, :] = state[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                x0[idx, :] = initial_state[time + self.sequence_length, :]

            wp_init_seq.append(wp_init)
            zp_init_seq.append(zp_init)
            wp_seq.append(wp)
            zp_seq.append(zp)
            x0_seq.append(x0)

        return (
            np.vstack(wp_init_seq),
            np.vstack(zp_init_seq),
            np.vstack(wp_seq),
            np.vstack(zp_seq),
            np.vstack(x0_seq),
        )

    def __len__(self) -> int:
        return self.wp_init.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {
            'wp_init': self.wp_init[idx],
            'zp_init': self.zp_init[idx],
            'wp': self.wp[idx],
            'zp': self.zp[idx],
            'x0': self.x0[idx],
        }


class RecurrentInitializerPredictorDataset(data.Dataset):
    def __init__(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        sequence_length: int,
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        (
            self.control_window,
            self.state_window,
            self.control_horizon,
            self.state_horizon,
        ) = self.__load_data(control_seqs, state_seqs)

    def __load_data(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        control_window_seq = list()
        state_window_seq = list()
        control_horizon_seq = list()
        state_horizon_seq = list()

        for control, state in zip(control_seqs, state_seqs):
            n_samples = int(
                (control.shape[0] - 2 * self.sequence_length) / self.sequence_length
            )

            control_window = np.zeros(
                (n_samples, self.sequence_length, self.control_dim),
                dtype=np.float64,
            )
            state_window = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )
            control_horizon = np.zeros(
                (n_samples, self.sequence_length, self.control_dim), dtype=np.float64
            )
            state_horizon = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )

            for idx in range(n_samples):
                time = idx * self.sequence_length

                control_window[idx, :, :] = control[
                    time : time + self.sequence_length, :
                ]

                state_window[idx, :, :] = state[time : time + self.sequence_length, :]

                control_horizon[idx, :, :] = control[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                state_horizon[idx, :, :] = state[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]

            control_window_seq.append(control_window)
            state_window_seq.append(state_window)
            control_horizon_seq.append(control_horizon)
            state_horizon_seq.append(state_horizon)

        return (
            np.vstack(control_window_seq),
            np.vstack(state_window_seq),
            np.vstack(control_horizon_seq),
            np.vstack(state_horizon_seq),
        )

    def __len__(self) -> int:
        return self.control_window.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {
            'control_window': self.control_window[idx],
            'state_window': self.state_window[idx],
            'control_horizon': self.control_horizon[idx],
            'state_horizon': self.state_horizon[idx],
        }


class RecurrentHybridPredictorDataset(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        sequence_length: int,
        un_control_seqs: List[NDArray[np.float64]],
        un_state_seqs: List[NDArray[np.float64]],
    ) -> None:
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]

        dataset = self.__load_data(
            control_seqs, state_seqs, un_control_seqs, un_state_seqs
        )
        self.x_init = dataset['x_init']
        self.x_pred = dataset['x_pred']
        self.ydot = dataset['ydot']
        self.y = dataset['y']
        self.initial_state = dataset['initial_state']
        self.x_control_unnormed = dataset['x_control_unnormed']
        self.x_state_unnormed = dataset['x_state_unnormed']

    def __load_data(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        un_control_seqs: List[NDArray[np.float64]],
        un_state_seqs: List[NDArray[np.float64]],
    ) -> Dict[str, NDArray[np.float64]]:
        x_init_seq = list()
        x_pred_seq = list()
        ydot_seq = list()
        y_seq = list()
        initial_state_seq = list()
        x_control_unnormed_seq = list()
        x_state_unnormed_seq = list()

        for control, state, un_control, un_state in zip(
            control_seqs, state_seqs, un_control_seqs, un_state_seqs
        ):
            n_samples = int(
                (control.shape[0] - self.sequence_length - 1)
                / (2 * self.sequence_length)
            )

            x_init = np.zeros(
                (n_samples, self.sequence_length, self.control_dim + self.state_dim),
                dtype=np.float64,
            )
            x_pred = np.zeros(
                (n_samples, self.sequence_length, self.control_dim), dtype=np.float64
            )
            ydot = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )
            y = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )
            initial_state = np.zeros((n_samples, self.state_dim), dtype=np.float64)
            x_control_unnormed = np.zeros(
                (n_samples, self.sequence_length, self.control_dim), dtype=np.float64
            )
            x_state_unnormed = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )

            for idx in range(n_samples):
                time = idx * self.sequence_length

                x_init[idx, :, :] = np.hstack(
                    (
                        control[time : time + self.sequence_length, :],
                        state[time : time + self.sequence_length, :],
                    )
                )
                x_pred[idx, :, :] = control[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                ydot[idx, :, :] = (
                    state[
                        time + self.sequence_length : time + 2 * self.sequence_length, :
                    ]
                    - state[
                        time
                        + self.sequence_length
                        - 1 : time
                        + 2 * self.sequence_length
                        - 1,
                        :,
                    ]
                )
                y[idx, :, :] = state[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                initial_state[idx, :] = un_state[time + self.sequence_length - 1, :]
                x_control_unnormed[idx, :, :] = un_control[
                    time + self.sequence_length : time + 2 * self.sequence_length, :
                ]
                x_state_unnormed[idx, :, :] = un_state[
                    time
                    + self.sequence_length
                    - 1 : time
                    + 2 * self.sequence_length
                    - 1,
                    :,
                ]

            x_init_seq.append(x_init)
            x_pred_seq.append(x_pred)
            ydot_seq.append(ydot)
            y_seq.append(y)
            initial_state_seq.append(initial_state)
            x_control_unnormed_seq.append(x_control_unnormed)
            x_state_unnormed_seq.append(x_state_unnormed)

        return dict(
            x_init=np.vstack(x_init_seq),
            x_pred=np.vstack(x_pred_seq),
            ydot=np.vstack(ydot_seq),
            y=np.vstack(y_seq),
            initial_state=np.vstack(initial_state_seq),
            x_control_unnormed=np.vstack(x_control_unnormed_seq),
            x_state_unnormed=np.vstack(x_state_unnormed_seq),
        )

    def __len__(self) -> int:
        return self.x_init.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {
            'x_init': self.x_init[idx],
            'x_pred': self.x_pred[idx],
            'ydot': self.ydot[idx],
            'y': self.y[idx],
            'initial_state': self.initial_state[idx],
            'x_control_unnormed': self.x_control_unnormed[idx],
            'x_state_unnormed': self.x_state_unnormed[idx],
        }
