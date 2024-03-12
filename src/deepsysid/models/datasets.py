from typing import Dict, List, Optional, Tuple

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
        window_size: Optional[int] = None,
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        if window_size is None:
            self.window_size = self.sequence_length
        else:
            self.window_size = window_size
        self.x, self.y = self.__load_data(control_seqs, state_seqs)

    def __load_data(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        x_seq = list()
        y_seq = list()
        for control, state in zip(control_seqs, state_seqs):
            n_samples = int((control.shape[0] - self.window_size) / self.window_size)

            x = np.zeros(
                (n_samples, self.window_size, self.control_dim + self.state_dim),
                dtype=np.float64,
            )
            y = np.zeros(
                (n_samples, self.window_size, self.state_dim), dtype=np.float64
            )

            for idx in range(n_samples):
                time = idx * self.window_size

                x[idx, :, :] = np.hstack(
                    (
                        control[time + 1 : time + 1 + self.window_size, :],
                        state[time : time + self.window_size, :],
                    )
                )
                y[idx, :, :] = state[time + 1 : time + 1 + self.window_size, :]

            x_seq.append(x)
            y_seq.append(y)

        return np.vstack(x_seq), np.vstack(y_seq)

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {'x': self.x[idx], 'y': self.y[idx]}
    
class RecurrentInitializerDataset2(data.Dataset[Dict[str, NDArray[np.float64]]]):
    def __init__(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        sequence_length: int,
        window_size: Optional[int] = None,
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        if window_size is None:
            self.window_size = self.sequence_length
        else:
            self.window_size = window_size
        self.x, self.y = self.__load_data(control_seqs, state_seqs)

    def __load_data(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        x_seq = list()
        y_seq = list()
        for control, state in zip(control_seqs, state_seqs):
            n_samples = int((control.shape[0] - self.window_size) / self.window_size)

            x = np.zeros(
                (n_samples, self.window_size, self.control_dim),
                dtype=np.float64,
            )
            y = np.zeros(
                (n_samples, self.window_size, self.state_dim), dtype=np.float64
            )

            for idx in range(n_samples):
                time = idx * self.window_size

                x[idx, :, :] = control[time + 1 : time + 1 + self.window_size, :]
                y[idx, :, :] = state[time + 1 : time + 1 + self.window_size, :]

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
        window_size: Optional[int] = None,
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        if window_size is None:
            self.window_size = self.sequence_length
        else:
            self.window_size = window_size
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
            # n_samples = int(
            #     (control.shape[0] - (self.window_size + self.sequence_length + 1))
            #     / self.sequence_length
            # )
            n_samples = int(control.shape[0]/(self.window_size+self.sequence_length+1))

            x0 = np.zeros(
                (n_samples, self.window_size, self.control_dim + self.state_dim),
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
                        control[time + 1 : time + self.window_size + 1, :],
                        state[time : time + self.window_size, :],
                    )
                )
                y0[idx, :] = state[time + self.window_size + 1, :]
                x[idx, :, :] = control[
                    time
                    + self.window_size
                    + 1 : time
                    + self.sequence_length
                    + self.window_size
                    + 1,
                    :,
                ]
                y[idx, :, :] = state[
                    time
                    + self.window_size
                    + 1 : time
                    + self.sequence_length
                    + self.window_size
                    + 1,
                    :,
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
            'idx': np.array(idx),
        }


class RecurrentPredictorInitializerInitialDataset(
    data.Dataset[Dict[str, NDArray[np.float64]]]
):
    def __init__(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_state_seqs: List[NDArray[np.float64]],
        sequence_length: int,
        x_mean: NDArray[np.float64],
        wp_mean: NDArray[np.float64],
        window_size: int = None,
        normalize_rnn: Optional[bool] = False,
    ):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        if window_size is None:
            self.window_size = sequence_length
        else:
            self.window_size = window_size
        self.normalize_rnn = normalize_rnn
        if self.normalize_rnn:
            self.control_dim_pred = (
                control_seqs[0].shape[1] + x_mean.shape[0] + wp_mean.shape[0]
            )
        else:
            self.control_dim_pred = control_seqs[0].shape[1]
        self.wp_mean = wp_mean
        self.x_mean = x_mean
        self.state_dim = state_seqs[0].shape[1]
        self.initial_state_dim = initial_state_seqs[0].shape[1]
        (
            self.wp_init,
            self.zp_init,
            self.x0_init,
            self.wp,
            self.zp,
            self.x0,
        ) = self.__load_data(control_seqs, state_seqs, initial_state_seqs)

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
        NDArray[np.float64],
    ]:
        wp_init_seq = list()
        zp_init_seq = list()
        x0_init_seq = list()
        wp_seq = list()
        zp_seq = list()
        x0_seq = list()

        for control, state, initial_state in zip(
            control_seqs, state_seqs, initial_state_seqs
        ):
            # n_samples = int(
            #     (control.shape[0] - (self.window_size + self.sequence_length + 1))
            #     / self.sequence_length
            # )
            n_samples = int(control.shape[0]/(self.window_size+self.sequence_length+1))

            wp_init = np.zeros(
                (n_samples, self.window_size, self.control_dim_pred),
                dtype=np.float64,
            )
            x0_init = np.zeros(
                (n_samples, self.window_size, self.initial_state_dim),
                dtype=np.float64,
            )
            zp_init = np.zeros((n_samples, self.window_size, self.state_dim), dtype=np.float64)
            wp = np.zeros(
                (n_samples, self.sequence_length, self.control_dim_pred),
                dtype=np.float64,
            )
            zp = np.zeros(
                (n_samples, self.sequence_length, self.state_dim), dtype=np.float64
            )

            x0 = np.zeros(shape=(n_samples, self.initial_state_dim), dtype=np.float64)

            for idx in range(n_samples):
                time = idx * self.sequence_length

                # inputs
                if self.normalize_rnn:
                    wp_init[idx, :, :] = np.hstack(
                        (
                            control[time + 1 : time + self.window_size + 1, :],
                            np.broadcast_to(
                                np.hstack((self.wp_mean, self.x_mean)),
                                (
                                    self.window_size,
                                    self.wp_mean.shape[0] + self.x_mean.shape[0],
                                ),
                            ),
                        )
                    )
                    wp[idx, :, :] = np.hstack(
                        (
                            control[
                                time
                                + self.window_size
                                + 1 : time
                                + self.sequence_length
                                + self.window_size
                                + 1,
                                :,
                            ],
                            np.broadcast_to(
                                np.hstack((self.wp_mean, self.x_mean)),
                                (
                                    self.sequence_length,
                                    self.wp_mean.shape[0] + self.x_mean.shape[0],
                                ),
                            ),
                        )
                    )
                else:
                    wp_init[idx, :, :] = control[
                        time + 1 : time + self.window_size + 1, :
                    ]
                    wp[idx, :, :] = control[
                        time
                        + self.window_size
                        + 1 : time
                        + self.sequence_length
                        + self.window_size
                        + 1,
                        :,
                    ]
                x0_init[idx, :, :] = initial_state[time : time + self.window_size, :]
                zp_init[idx, :] = state[time + self.window_size + 1, :]

                zp[idx, :, :] = state[
                    time
                    + self.window_size
                    + 1 : time
                    + self.sequence_length
                    + self.window_size
                    + 1,
                    :,
                ]
                x0[idx, :] = initial_state[time + self.window_size + 1, :]

            wp_init_seq.append(wp_init)
            zp_init_seq.append(zp_init)
            x0_init_seq.append(x0_init)
            wp_seq.append(wp)
            zp_seq.append(zp)
            x0_seq.append(x0)

        return (
            np.vstack(wp_init_seq),
            np.vstack(zp_init_seq),
            np.vstack(x0_init_seq),
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
            'x0_init': self.x0_init[idx],
            'wp': self.wp[idx],
            'zp': self.zp[idx],
            'x0': self.x0[idx],
        }


class RecurrentPredictorInitializerInitialDataset2(
    data.Dataset[Dict[str, NDArray[np.float64]]]
):
    def __init__(
        self,
        control_seqs: List[NDArray[np.float64]],
        state_seqs: List[NDArray[np.float64]],
        initial_state_seqs: List[NDArray[np.float64]],
        horizon: int,
        window: int,
    ):
        self.h = horizon
        self.w = window
        self.nd = control_seqs[0].shape[1]
        self.ne = state_seqs[0].shape[1]
        self.nx = initial_state_seqs[0].shape[1]
        
        (
            self.d_init,
            self.e_init,
            self.x0_init,
            self.d,
            self.e,
            self.x0,
        ) = self.__load_data(control_seqs, state_seqs, initial_state_seqs)

    def __load_data(
        self,
        d_seqs: List[NDArray[np.float64]],
        e_seqs: List[NDArray[np.float64]],
        x0_seqs: List[NDArray[np.float64]],
    ) -> Tuple[
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
        NDArray[np.float64],
    ]:
        d_init_seq = list()
        e_init_seq = list()
        x0_init_seq = list()
        d_seq = list()
        e_seq = list()
        x0_seq = list()

        for ds, es, x0s in zip(
            d_seqs, e_seqs, x0_seqs
        ):
            n_samples = int(ds.shape[0]/(self.w+self.h+1))

            d_init = np.zeros((n_samples, self.w, self.nd), dtype=np.float64)
            e_init = np.zeros((n_samples, self.w, self.ne), dtype=np.float64)
            x0_init = np.zeros((n_samples, self.nx), dtype=np.float64)

            d = np.zeros((n_samples, self.h, self.nd), dtype=np.float64)
            e = np.zeros((n_samples, self.h, self.ne), dtype=np.float64)
            x0 = np.zeros((n_samples, self.nx), dtype=np.float64)

            for idx in range(n_samples):
                time = idx * self.h

                # inputs
                d_init[idx, :, :self.nd] = ds[time + 1 : time + self.w + 1, :]
                # d_init[idx, :, self.nd:] = es[time : time + self.w, :]
                d[idx, :, :] = ds[time + self.w + 1 : time + self.h + self.w + 1, :]
                # initial state
                x0_init[idx, :] = x0s[time, :]
                x0[idx, :] = x0s[time + self.w + 1, :]
                # outputs
                e_init[idx, :] = es[time +1 : time + self.w +1, :]
                e[idx, :, :] = es[time + self.w + 1 : time + self.h + self.w + 1, :]
                

            d_init_seq.append(d_init)
            e_init_seq.append(e_init)
            x0_init_seq.append(x0_init)
            d_seq.append(d)
            e_seq.append(e)
            x0_seq.append(x0)

        return (
            np.vstack(d_init_seq),
            np.vstack(e_init_seq),
            np.vstack(x0_init_seq),
            np.vstack(d_seq),
            np.vstack(e_seq),
            np.vstack(x0_seq),
        )

    def __len__(self) -> int:
        return self.d_init.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, NDArray[np.float64]]:
        return {
            'd_init': self.d_init[idx],
            'e_init': self.e_init[idx],
            'x0_init': self.x0_init[idx],
            'd': self.d[idx],
            'e': self.e[idx],
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
                        control[time + 1 : time + 1 + self.sequence_length, :],
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
