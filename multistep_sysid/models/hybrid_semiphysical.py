import abc
import json
import logging

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from . import base
from ..networks import rnn, loss
from .. import utils

logger = logging.getLogger()


class HybridSerialSemiphysicalLSTMModel(base.DynamicIdentificationModel, abc.ABC):
    def __init__(self, device_name='cpu', **kwargs):
        super().__init__(**kwargs)

        self.device_name = device_name
        self.device = torch.device(device_name)

        self.control_dim = len(kwargs['control_names'])
        self.state_dim = len(kwargs['state_names'])

        self.recurrent_dim = kwargs['recurrent_dim']
        self.num_recurrent_layers = kwargs['num_recurrent_layers']
        self.feedforward_dim = kwargs['feedforward_dim']
        self.dropout = kwargs['dropout']

        self.sequence_length = kwargs['sequence_length']
        self.learning_rate = kwargs['learning_rate']
        self.batch_size = kwargs['batch_size']
        self.epochs_initializer = kwargs['epochs_initializer']
        self.epochs_parallel = kwargs['epochs_parallel']
        self.epochs_feedback = kwargs['epochs_feedback']

        if kwargs['loss'] == 'mse':
            self.loss = nn.MSELoss().to(self.device)
        elif kwargs['loss'] == 'msge':
            self.loss = loss.MSGELoss().to(self.device)

        self.whitebox = nn.Linear(
            in_features=self.get_semiphysical_in_features(),
            out_features=self.state_dim
        ).to(self.device)
        self.whitebox.weight.requires_grad = False
        self.whitebox.bias.requires_grad = False

        self.blackbox = rnn.LinearOutputLSTM(
            input_dim=self.control_dim + self.state_dim,  # control input and whitebox estimate
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=self.state_dim,
            dropout=self.dropout
        ).to(self.device)

        self.initializer = rnn.BasicLSTM(
            input_dim=self.control_dim + self.state_dim,
            recurrent_dim=self.recurrent_dim,
            output_dim=self.feedforward_dim + [self.state_dim],
            num_recurrent_layers=self.num_recurrent_layers,
            dropout=self.dropout
        ).to(self.device)

        self.optimizer_initializer = optim.Adam(
            params=self.initializer.parameters(),
            lr=self.learning_rate
        )
        self.optimizer_end2end = optim.Adam(
            params=self.blackbox.parameters(),
            lr=self.learning_rate
        )

        self.control_mean = None
        self.control_stddev = None
        self.state_mean = None
        self.state_stddev = None
        self.semiphysical_in_mean = None
        self.semiphysical_in_stddev = None

    def train(self, control_seqs, state_seqs, validator=None):
        self.blackbox.train()
        self.whitebox.train()
        self.initializer.train()

        self.control_mean, self.control_stddev = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_stddev = utils.mean_stddev(state_seqs)
        # don't forget to match control and state to correct time for input: u(t), x(t-1) predict x(t)
        semiphysical_in_seqs = [self.expand_semiphysical_input(torch.from_numpy(control[1:]),
                                                               torch.from_numpy(state[:-1])).numpy()
                                for control, state in zip(control_seqs, state_seqs)]
        self.semiphysical_in_mean, self.semiphysical_in_stddev = utils.mean_stddev(semiphysical_in_seqs)

        un_control_seqs = control_seqs
        un_state_seqs = state_seqs
        control_seqs = [utils.normalize(control, self.control_mean, self.control_stddev) for control in control_seqs]
        state_seqs = [utils.normalize(state, self.state_mean, self.state_stddev) for state in state_seqs]
        semiphysical_in_seqs = [
            utils.normalize(sp_in, self.semiphysical_in_mean, self.semiphysical_in_stddev)
            for sp_in in semiphysical_in_seqs
        ]

        sp_mean = torch.from_numpy(self.semiphysical_in_mean).float().to(self.device)
        sp_stddev = torch.from_numpy(self.semiphysical_in_stddev).float().to(self.device)
        normalize_semiphysical = lambda x: (x - sp_mean) / sp_stddev
        state_mean = torch.from_numpy(self.state_mean).float().to(self.device)
        state_stddev = torch.from_numpy(self.state_stddev).float().to(self.device)
        denormalize_state = lambda x: (x * state_stddev) + state_mean

        # Train linear model
        self.train_whitebox(semiphysical_in_seqs, state_seqs)

        initializer_dataset = _InitializerDataset(control_seqs, state_seqs, self.sequence_length)
        for i in range(self.epochs_initializer):
            data_loader = data.DataLoader(initializer_dataset, self.batch_size, shuffle=True, drop_last=True)
            total_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                self.initializer.zero_grad()
                y = self.initializer.forward(batch['x'].float().to(self.device))
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_initializer.step()

            logger.info(f'Epoch {i + 1}/{self.epochs_initializer} - Epoch Loss (Initializer): {total_loss}')

        dataset = _End2EndDataset(
            control_seqs=control_seqs, state_seqs=state_seqs, semiphysical_in_seqs=semiphysical_in_seqs,
            sequence_length=self.sequence_length, un_control_seqs=un_control_seqs, un_state_seqs=un_state_seqs
        )
        for i in range(self.epochs_parallel):
            data_loader = data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            total_epoch_loss = 0.0
            for batch in data_loader:
                self.blackbox.zero_grad()

                x_semiphysical = batch['x_semiphysical_in'].float().to(self.device)
                y_semiphysical = torch.zeros((self.batch_size, self.sequence_length, self.state_dim))\
                    .float().to(self.device)

                for time in range(self.sequence_length):
                    y_semiphysical[:, time, :] = self.whitebox.forward(x_semiphysical[:, time, :])

                x_init = batch['x_init'].float().to(self.device)
                x_pred = torch.cat((batch['x_pred'].float().to(self.device), y_semiphysical), dim=2)

                _, hx_init = self.initializer.forward(x_init, return_state=True)

                y_blackbox = self.blackbox.forward(x_pred, hx=hx_init)
                y = y_blackbox + y_semiphysical

                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_end2end.step()

            logger.info(f'Epoch {i + 1}/{self.epochs_parallel} - Epoch Loss (Parallel): {total_epoch_loss}')

            if validator:
                validation_error = validator(model=self, epoch=i)
                self.blackbox.train(True)
                self.whitebox.train(True)
                self.initializer.train(True)

                if validation_error:
                    validation_error_str = ','.join(
                        f'{err:.3E}' for err in validation_error
                    )
                    logger.info(f'Epoch {i + 1}/{self.epochs_parallel} - Validation Error: [{validation_error_str}]')

        for i in range(self.epochs_feedback):
            data_loader = data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            total_epoch_loss = 0.0
            for batch in data_loader:
                self.blackbox.zero_grad()

                current_state = batch['initial_state'].float().to(self.device)
                x_control_unnormed = batch['x_control_unnormed'].float().to(self.device)
                x_pred = batch['x_pred'].float().to(self.device)
                y_est = torch.zeros((self.batch_size, self.sequence_length, self.state_dim)).to(self.device)

                x_init = batch['x_init'].float().to(self.device)
                _, hx_init = self.initializer.forward(x_init, return_state=True)

                for time in range(self.sequence_length):
                    y_semiphysical = self.whitebox.forward(
                        normalize_semiphysical(
                            self.expand_semiphysical_input(x_control_unnormed[:, time, :], current_state)
                        )
                    )

                    y_blackbox, hx_init = self.blackbox.forward(
                        torch.cat((x_pred[:, time, :].unsqueeze(1), y_semiphysical.unsqueeze(1)), dim=2),
                        hx=hx_init, return_state=True)
                    current_state = y_blackbox.squeeze(1) + y_semiphysical
                    y_est[:, time, :] = current_state
                    current_state = denormalize_state(current_state)

                batch_loss = self.loss.forward(y_est, batch['y'].float().to(self.device))
                total_epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_end2end.step()

            logger.info(f'Epoch {i + 1}/{self.epochs_feedback} - Epoch Loss (Feedback): {total_epoch_loss}')

            if validator:
                validation_error = validator(model=self, epoch=i)
                self.blackbox.train(True)
                self.whitebox.train(True)
                self.initializer.train(True)

                if validation_error:
                    validation_error_str = ','.join(
                        f'{err:.3E}' for err in validation_error
                    )
                    logger.info(f'Epoch {i + 1}/{self.epochs_feedback} - Validation Error: [{validation_error_str}]')

    def simulate(self, initial_control, initial_state, control, return_whitebox_blackbox=False, threshold=np.infty):
        self.blackbox.eval()
        self.whitebox.eval()
        self.initializer.eval()

        sp_mean = torch.from_numpy(self.semiphysical_in_mean).float().to(self.device)
        sp_stddev = torch.from_numpy(self.semiphysical_in_stddev).float().to(self.device)
        normalize_semiphysical = lambda x: (x - sp_mean) / sp_stddev
        state_mean = torch.from_numpy(self.state_mean).float().to(self.device)
        state_stddev = torch.from_numpy(self.state_stddev).float().to(self.device)
        denormalize_state = lambda x: (x * state_stddev) + state_mean

        un_control = control
        current_state = initial_state[-1, :]
        initial_control = utils.normalize(initial_control, self.control_mean, self.control_stddev)
        initial_state = utils.normalize(initial_state, self.state_mean, self.state_stddev)
        control = utils.normalize(control, self.control_mean, self.control_stddev)

        y = np.zeros((control.shape[0], self.state_dim))
        whitebox = np.zeros((control.shape[0], self.state_dim))
        blackbox = np.zeros((control.shape[0], self.state_dim))

        with torch.no_grad():
            x_init = torch.from_numpy(np.hstack((initial_control, initial_state))).unsqueeze(0).float().to(self.device)
            _, hx = self.initializer.forward(x_init, return_state=True)  # hx is hidden state of predictor LSTM

            x_control_un = torch.from_numpy(un_control).unsqueeze(0).float().to(self.device)
            current_state = torch.from_numpy(current_state).unsqueeze(0).float().to(self.device)
            x_pred = torch.from_numpy(control).unsqueeze(0).float().to(self.device)
            for time in range(control.shape[0]):
                y_semiphysical = self.whitebox.forward(normalize_semiphysical(
                    self.expand_semiphysical_input(x_control_un[:, time, :], current_state)))
                x_blackbox = torch.cat((x_pred[:, time, :], y_semiphysical), dim=1).unsqueeze(1)
                y_blackbox, hx = self.blackbox.forward(x_blackbox, hx=hx, return_state=True)
                y_blackbox = torch.clamp(y_blackbox, -threshold, threshold)
                y_est = y_semiphysical + y_blackbox.squeeze(1)
                current_state = denormalize_state(y_est)

                y[time, :] = current_state.cpu().detach().numpy()
                whitebox[time, :] = y_semiphysical.cpu().detach().numpy()
                blackbox[time, :] = y_blackbox.squeeze(1).cpu().detach().numpy()

        if return_whitebox_blackbox:
            return y, whitebox, blackbox
        else:
            return y

    def save(self, file_path):
        torch.save(self.whitebox.state_dict(), file_path[0])
        torch.save(self.blackbox.state_dict(), file_path[1])
        torch.save(self.initializer.state_dict(), file_path[2])
        with open(file_path[3], mode='w') as f:
            json.dump({
                'state_mean': self.state_mean.tolist(),
                'state_stddev': self.state_stddev.tolist(),
                'control_mean': self.control_mean.tolist(),
                'control_stddev': self.control_stddev.tolist(),
                'semiphysical_in_mean': self.semiphysical_in_mean.tolist(),
                'semiphysical_in_stddev': self.semiphysical_in_stddev.tolist()
            }, f)

    def load(self, file_path):
        self.whitebox.load_state_dict(torch.load(file_path[0], map_location=self.device_name))
        self.whitebox.weight.requires_grad = False
        self.whitebox.bias.requires_grad = False
        self.blackbox.load_state_dict(torch.load(file_path[1], map_location=self.device_name))
        self.initializer.load_state_dict(torch.load(file_path[2], map_location=self.device_name))
        with open(file_path[3], mode='r') as f:
            norm = json.load(f)
        self.state_mean = np.array(norm['state_mean'])
        self.state_stddev = np.array(norm['state_stddev'])
        self.control_mean = np.array(norm['control_mean'])
        self.control_stddev = np.array(norm['control_stddev'])
        self.semiphysical_in_mean = np.array(norm['semiphysical_in_mean'])
        self.semiphysical_in_stddev = np.array(norm['semiphysical_in_stddev'])

    def get_file_extension(self):
        return 'whitebox.pth', 'blackbox.pth', 'initializer.pth', 'json'

    def get_parameter_count(self):
        whitebox_count = sum(p.numel() for p in self.whitebox.parameters())
        blackbox_count = sum(p.numel() for p in self.blackbox.parameters() if p.requires_grad)
        initializer_count = sum(p.numel() for p in self.initializer.parameters() if p.requires_grad)
        return whitebox_count + blackbox_count + initializer_count

    def train_whitebox(self, semiphysical_in_seqs, state_seqs):
        train_x, train_y = [], []
        for sp_in, state in zip(semiphysical_in_seqs, state_seqs):
            train_x.append(sp_in)
            train_y.append(state[1:])

        train_x = np.vstack(train_x)
        train_y = np.vstack(train_y)

        regressor = LinearRegression(fit_intercept=True)
        regressor.fit(train_x, train_y)
        linear_fit = r2_score(regressor.predict(train_x), train_y, multioutput='uniform_average')
        logger.info(f'Whitebox R2 Score: {linear_fit}')

        self.whitebox.weight = nn.Parameter(torch.from_numpy(regressor.coef_).float().to(self.device),
                                            requires_grad=False)
        self.whitebox.bias = nn.Parameter(torch.from_numpy(regressor.intercept_).float().to(self.device),
                                          requires_grad=False)

    @abc.abstractmethod
    def get_semiphysical_in_features(self):
        pass

    @abc.abstractmethod
    def expand_semiphysical_input(self, control, state):
        pass


class HybridSerialLinearLSTMModel(HybridSerialSemiphysicalLSTMModel):
    def get_semiphysical_in_features(self):
        return self.control_dim + self.state_dim

    def expand_semiphysical_input(self, control, state):
        return torch.cat((control, state), dim=1)


class HybridSerialQuadraticLSTMModel(HybridSerialSemiphysicalLSTMModel):
    def get_semiphysical_in_features(self):
        return 2 * (self.control_dim + self.state_dim)

    def expand_semiphysical_input(self, control, state):
        return torch.cat((
            control, state, control * control, state * state
        ), dim=1)


class HybridSerialBlankeLSTMModel(HybridSerialSemiphysicalLSTMModel):
    """
    does not model control inputs
    does not model roll rate p
    employs drag and damping features from maneuvering model by Blanke and Christensen
    phi is only dependent on phi and p (accordingly similar to Euler method)
    """
    def train_whitebox(self, semiphysical_in_seqs, state_seqs):
        train_x, train_y = [], []
        for sp_in, state in zip(semiphysical_in_seqs, state_seqs):
            train_x.append(sp_in)
            train_y.append(state[1:])

        train_x = np.vstack(train_x)
        train_y = np.vstack(train_y)

        # Train each dimension as separate equation
        def train_dimension(dim_mask, dim_name, dim_idx):
            regressor = LinearRegression(fit_intercept=False)
            regressor.fit(train_x[:, dim_mask], train_y[:, dim_idx])
            linear_fit = r2_score(
                regressor.predict(train_x[:, dim_mask]), train_y[:, dim_idx], multioutput='uniform_average'
            )
            logger.info(f'Whitebox R2 Score ({dim_name}): {linear_fit}')
            return regressor

        mask_u = (0, 5, 6)
        mask_v = (1, 7, 8, 9, 10, 11, 12, 13, 14)
        mask_r = (3, 7, 11, 15, 16, 17, 18, 19)
        mask_phi = (2, 4)
        reg_u = train_dimension(mask_u, 'u', 0)
        reg_v = train_dimension(mask_v, 'v', 1)
        reg_r = train_dimension(mask_r, 'r', 3)
        reg_phi = train_dimension(mask_phi, 'phi', 4)

        weight = np.zeros((self.state_dim, self.get_semiphysical_in_features()))
        weight[0, mask_u] = reg_u.coef_
        weight[1, mask_v] = reg_v.coef_
        weight[3, mask_r] = reg_r.coef_
        weight[4, mask_phi] = reg_phi.coef_

        bias = np.array((
            0.0, 0.0, 0.0, 0.0, 0.0
        ))

        self.whitebox.weight = nn.Parameter(torch.from_numpy(weight).float().to(self.device), requires_grad=False)
        self.whitebox.bias = nn.Parameter(torch.from_numpy(bias).float().to(self.device), requires_grad=False)

    def get_semiphysical_in_features(self):
        return 20# + self.control_dim

    def expand_semiphysical_input(self, control, state):
        u = state[:, 0].unsqueeze(1)
        v = state[:, 1].unsqueeze(1)
        p = state[:, 2].unsqueeze(1)
        r = state[:, 3].unsqueeze(1)
        phi = state[:, 4].unsqueeze(1)
        au = torch.abs(u)
        av = torch.abs(v)
        ar = torch.abs(r)
        auv = torch.abs(u * v)
        aur = torch.abs(u * r)
        auphi = torch.abs(u * phi)

        state = torch.cat((
            u,  # 0: X
            v,  # 1: Y
            p,  # 2: phi
            r,  # 3: N
            phi,  # 4: phi
            au * u,  # 5: X
            v * r,  # 6: X
            au * v,  # 7: Y, N
            u * r,  # 8: Y
            av * v,  # 9: Y
            ar * v,  # 10: Y
            av * r,  # 11: Y, N
            auv * phi,  # 12: Y
            aur * phi,  # 13: Y
            u * u * phi,  # 14: Y
            au * r,  # 15: N
            ar * r,  # 16: N
            auphi * phi,  # 17: N
            ar * u * phi,  # 18: N
            au * u * phi  # 19: N
        ), dim=1)
        return state
        #return torch.cat((control, state), dim=1)


class _InitializerDataset(data.Dataset):
    # x=[control state], y=[state]
    def __init__(self, control_seqs, state_seqs, sequence_length):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.x = None
        self.y = None
        self.__load_data(control_seqs, state_seqs)

    def __load_data(self, control_seqs, state_seqs):
        x_seq = list()
        y_seq = list()
        for control, state in zip(control_seqs, state_seqs):
            n_samples = int((control.shape[0] - self.sequence_length - 1) / self.sequence_length)

            x = np.zeros((n_samples, self.sequence_length, self.control_dim + self.state_dim))
            y = np.zeros((n_samples, self.sequence_length, self.state_dim))

            for idx in range(n_samples):
                time = idx * self.sequence_length

                x[idx, :, :] = np.hstack((control[time + 1:time + 1 + self.sequence_length, :],
                                          state[time:time + self.sequence_length, :]))
                y[idx, :, :] = state[time + 1:time + 1 + self.sequence_length, :]

            x_seq.append(x)
            y_seq.append(y)

        self.x = np.vstack(x_seq)
        self.y = np.vstack(y_seq)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return {'x': self.x[idx], 'y': self.y[idx]}


class _End2EndDataset(data.Dataset):
    def __init__(self, control_seqs, state_seqs, semiphysical_in_seqs, sequence_length, un_control_seqs,
                 un_state_seqs):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.sp_in_dim = semiphysical_in_seqs[0].shape[1]

        self.x_semiphysical_in = None
        self.x_init = None
        self.x_pred = None
        self.ydot = None
        self.y = None
        self.initial_state = None
        self.x_control_unnormed = None

        self.__load_data(control_seqs, state_seqs, semiphysical_in_seqs, un_control_seqs, un_state_seqs)

    def __load_data(self, control_seqs, state_seqs, semiphysical_in_seqs, un_control_seqs, un_state_seqs):
        x_semiphysical_in_seq = list()
        x_init_seq = list()
        x_pred_seq = list()
        ydot_seq = list()
        y_seq = list()
        initial_state_seq = list()
        x_control_unnormed_seq = list()

        for control, state, sp_in, un_control, un_state \
                in zip(control_seqs, state_seqs, semiphysical_in_seqs, un_control_seqs, un_state_seqs):
            n_samples = int((control.shape[0] - self.sequence_length - 1) / (2 * self.sequence_length))

            x_semiphysical_in = np.zeros((n_samples, self.sequence_length, self.sp_in_dim))
            x_init = np.zeros((n_samples, self.sequence_length, self.control_dim + self.state_dim))
            x_pred = np.zeros((n_samples, self.sequence_length, self.control_dim))
            ydot = np.zeros((n_samples, self.sequence_length, self.state_dim))
            y = np.zeros((n_samples, self.sequence_length, self.state_dim))
            initial_state = np.zeros((n_samples, self.state_dim))
            x_control_unnormed = np.zeros((n_samples, self.sequence_length, self.control_dim))

            for idx in range(n_samples):
                time = idx * self.sequence_length

                x_semiphysical_in[idx, :, :] = sp_in[time + self.sequence_length: time + 2 * self.sequence_length, :]
                x_init[idx, :, :] = np.hstack((control[time:time + self.sequence_length, :],
                                               state[time:time + self.sequence_length, :]))
                x_pred[idx, :, :] = control[time + self.sequence_length:time + 2 * self.sequence_length, :]
                ydot[idx, :, :] = (state[time + self.sequence_length:time + 2 * self.sequence_length, :]
                                   - state[time + self.sequence_length - 1:time + 2 * self.sequence_length - 1, :])
                y[idx, :, :] = state[time + self.sequence_length:time + 2 * self.sequence_length, :]
                initial_state[idx, :] = un_state[time + self.sequence_length - 1, :]
                x_control_unnormed[idx, :, :] = un_control[time + self.sequence_length:time + 2 * self.sequence_length,
                                                :]

            x_semiphysical_in_seq.append(x_semiphysical_in)
            x_init_seq.append(x_init)
            x_pred_seq.append(x_pred)
            ydot_seq.append(ydot)
            y_seq.append(y)
            initial_state_seq.append(initial_state)
            x_control_unnormed_seq.append(x_control_unnormed)

        self.x_semiphysical_in = np.vstack(x_semiphysical_in_seq)
        self.x_init = np.vstack(x_init_seq)
        self.x_pred = np.vstack(x_pred_seq)
        self.ydot = np.vstack(ydot_seq)
        self.y = np.vstack(y_seq)
        self.initial_state = np.vstack(initial_state_seq)
        self.x_control_unnormed = np.vstack(x_control_unnormed_seq)

    def __len__(self):
        return self.x_init.shape[0]

    def __getitem__(self, idx):
        return {'x_semiphysical_in': self.x_semiphysical_in[idx], 'x_init': self.x_init[idx],
                'x_pred': self.x_pred[idx], 'ydot': self.ydot[idx], 'y': self.y[idx],
                'initial_state': self.initial_state[idx], 'x_control_unnormed': self.x_control_unnormed[idx]}
