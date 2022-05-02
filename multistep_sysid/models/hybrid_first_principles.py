import json
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from . import base
from ..networks import first_principles, loss, rnn
from .. import utils

logger = logging.getLogger()


class HybridSerialFirstPrinciplesLSTMModel(base.DynamicIdentificationModel):
    def __init__(self, whitebox, device_name='cpu', **kwargs):
        super().__init__(**kwargs)

        self.device_name = device_name
        self.device = torch.device(device_name)

        self.control_dim = len(kwargs['control_names'])
        self.state_dim = len(kwargs['state_names'])
        self.time_delta = float(kwargs['dt'])

        self.recurrent_dim = kwargs['recurrent_dim']
        self.num_recurrent_layers = kwargs['num_recurrent_layers']
        self.feedforward_dim = kwargs['feedforward_dim']  # only used by initializer
        self.dropout = kwargs['dropout']

        self.sequence_length = kwargs['sequence_length']
        self.learning_rate = kwargs['learning_rate']
        self.batch_size = kwargs['batch_size']
        self.epochs_initializer = kwargs['epochs_initializer']
        self.epochs_parallel = kwargs['epochs_parallel']

        if kwargs['loss'] == 'mse':
            self.loss = nn.MSELoss().to(self.device)
        elif kwargs['loss'] == 'msge':
            self.loss = loss.MSGELoss().to(self.device)

        self.whitebox = whitebox.to(self.device).float()

        self.initializer = rnn.BasicLSTM(
            input_dim=self.control_dim+self.state_dim,
            recurrent_dim=self.recurrent_dim,
            output_dim=self.feedforward_dim + [self.state_dim],
            num_recurrent_layers=self.num_recurrent_layers,
            dropout=self.dropout
        ).to(self.device)

        # output layer is purely linear without bias, this allows separation of previous state and rate of change,
        # for improved interpretability since contribution to change of state can be attributed clearly
        # to either whitebox or blackbox
        self.blackbox = rnn.LinearOutputLSTM(  # serial-connection with whitebox
            input_dim=self.control_dim+self.state_dim,
            recurrent_dim=self.recurrent_dim,
            output_dim=self.state_dim,
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

    def train(self, control_seqs, state_seqs, validator=None):
        self.blackbox.train()
        self.initializer.train()

        self.control_mean, self.control_stddev = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_stddev = utils.mean_stddev(state_seqs)
        whitebox_scaling = torch.from_numpy(1.0/self.state_stddev).float().to(self.device)

        unnormed_control_seqs = control_seqs
        unnormed_state_seqs = state_seqs
        control_seqs = [utils.normalize(control, self.control_mean, self.control_stddev) for control in control_seqs]
        state_seqs = [utils.normalize(state, self.state_mean, self.state_stddev) for state in state_seqs]

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
            control_seqs, state_seqs, unnormed_control_seqs, unnormed_state_seqs, self.sequence_length
        )
        for i in range(self.epochs_parallel):
            data_loader = data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            total_epoch_loss = 0.0
            for batch in data_loader:
                self.blackbox.zero_grad()

                x_whitebox_control = batch['x_whitebox_control'].float().to(self.device)
                x_whitebox_state = batch['x_whitebox_state'].float().to(self.device)
                ydot_whitebox = torch.zeros((self.batch_size, self.sequence_length, self.state_dim))\
                    .float().to(self.device)

                for time in range(self.sequence_length):
                    ydot_whitebox[:, time, :] = (
                            self.time_delta * whitebox_scaling * self.whitebox.forward(x_whitebox_control[:, time, :],
                                                                                       x_whitebox_state[:, time, :])
                    )

                x_init = batch['x_init'].float().to(self.device)
                x_pred = torch.cat((batch['x_pred'].float().to(self.device), ydot_whitebox), dim=2)

                _, hx_init = self.initializer.forward(x_init, return_state=True)

                y_blackbox = self.blackbox.forward(x_pred, hx=hx_init)
                y = y_blackbox + ydot_whitebox

                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))

                total_epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_end2end.step()

            logger.info(f'Epoch {i + 1}/{self.epochs_parallel} - Epoch Loss (End2End): {total_epoch_loss}')
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

    def simulate(self, initial_control, initial_state, control, return_whitebox_blackbox=False, threshold=np.infty):
        self.blackbox.eval()
        self.initializer.eval()

        initial_control = utils.normalize(initial_control, self.control_mean, self.control_stddev)
        un_initial_state = initial_state
        initial_state = utils.normalize(initial_state, self.state_mean, self.state_stddev)
        control = utils.normalize(control, self.control_mean, self.control_stddev)

        whitebox_scaling = torch.from_numpy(1.0 / self.state_stddev).float().to(self.device)
        state_mean = torch.from_numpy(self.state_mean).float().to(self.device)
        state_stddev = torch.from_numpy(self.state_stddev).float().to(self.device)
        denormalize_state = lambda x: (x*state_stddev)+state_mean
        normalize_state = lambda x: (x - state_mean)/state_stddev
        control_mean = torch.from_numpy(self.control_mean).float().to(self.device)
        control_stddev = torch.from_numpy(self.control_stddev).float().to(self.device)
        denormalize_control = lambda x: (x*control_stddev)+control_mean

        states = np.zeros((control.shape[0], self.state_dim))
        whitebox = np.zeros((control.shape[0], self.state_dim))
        blackbox = np.zeros((control.shape[0], self.state_dim))

        with torch.no_grad():
            x_init = torch.from_numpy(np.hstack((initial_control, initial_state))).unsqueeze(0).float().to(self.device)
            x_blackbox = torch.from_numpy(control).unsqueeze(0).float().to(self.device)
            current_state = torch.from_numpy(un_initial_state[-1, :]).unsqueeze(0).float().to(self.device)

            # initialize blackbox
            _, hx_blackbox = self.initializer.forward(x_init, return_state=True)

            for time in range(control.shape[0]):
                # predict acceleration
                ydot_whitebox = self.time_delta * whitebox_scaling * self.whitebox.forward(
                    denormalize_control(x_blackbox[:, time, :]),
                    current_state
                ).unsqueeze(1)
                y_blackbox, hx_blackbox = self.blackbox.forward(
                    x_pred=torch.cat((x_blackbox[:, time, :].unsqueeze(1), ydot_whitebox), dim=2),
                    hx=hx_blackbox,
                    return_state=True
                )

                y_blackbox = torch.clamp(y_blackbox, -threshold, threshold)
                y = y_blackbox + ydot_whitebox

                # denormalize state
                current_state = denormalize_state(y.squeeze(1))
                states[time, :] = current_state.squeeze().cpu().detach().numpy()
                whitebox[time, :] = ydot_whitebox.cpu().detach().numpy()
                blackbox[time, :] = y_blackbox.squeeze(1).cpu().detach().numpy()

        if return_whitebox_blackbox:
            return states, whitebox, blackbox
        else:
            return states

    def save(self, file_path):
        torch.save(self.blackbox.state_dict(), file_path[0])
        torch.save(self.initializer.state_dict(), file_path[1])
        with open(file_path[2], mode='w') as f:
            json.dump({
                'state_mean': self.state_mean.tolist(),
                'state_stddev': self.state_stddev.tolist(),
                'control_mean': self.control_mean.tolist(),
                'control_stddev': self.control_stddev.tolist()
            }, f)

    def load(self, file_path):
        self.blackbox.load_state_dict(torch.load(file_path[0], map_location=self.device_name))
        self.initializer.load_state_dict(torch.load(file_path[1], map_location=self.device_name))
        with open(file_path[2], mode='r') as f:
            norm = json.load(f)
        self.state_mean = np.array(norm['state_mean'])
        self.state_stddev = np.array(norm['state_stddev'])
        self.control_mean = np.array(norm['control_mean'])
        self.control_stddev = np.array(norm['control_stddev'])

    def get_file_extension(self):
        return 'blackbox.pth', 'initializer.pth', 'json'

    def get_parameter_count(self):
        return (sum(p.numel() for p in self.blackbox.parameters() if p.requires_grad)
                + sum(p.numel() for p in self.initializer.parameters() if p.requires_grad))


class HybridSerialMinimalLSTMModel(HybridSerialFirstPrinciplesLSTMModel):
    def __init__(self, **kwargs):
        super().__init__(whitebox=first_principles.MinimalManeuveringEquations(**kwargs), **kwargs)


class HybridSerialPropulsionLSTMModel(HybridSerialFirstPrinciplesLSTMModel):
    def __init__(self, **kwargs):
        super().__init__(whitebox=first_principles.PropulsionManeuveringEquations(**kwargs), **kwargs)


class HybridSerialAutoRegressiveLSTMModel(HybridSerialFirstPrinciplesLSTMModel):
    def __init__(self, whitebox, **kwargs):
        super().__init__(whitebox=whitebox, **kwargs)

        self.epochs_feedback = kwargs['epochs_feedback']

    def train(self, control_seqs, state_seqs, validator=None):
        self.blackbox.train()
        self.initializer.train()

        self.control_mean, self.control_stddev = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_stddev = utils.mean_stddev(state_seqs)

        whitebox_scaling = torch.from_numpy(1.0 / self.state_stddev).float().to(self.device)
        state_mean = torch.from_numpy(self.state_mean).float().to(self.device)
        state_stddev = torch.from_numpy(self.state_stddev).float().to(self.device)
        denormalize_state = lambda x: (x * state_stddev) + state_mean

        unnormed_control_seqs = control_seqs
        unnormed_state_seqs = state_seqs
        control_seqs = [utils.normalize(control, self.control_mean, self.control_stddev) for control in control_seqs]
        state_seqs = [utils.normalize(state, self.state_mean, self.state_stddev) for state in state_seqs]

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
            control_seqs, state_seqs, unnormed_control_seqs, unnormed_state_seqs, self.sequence_length
        )
        for i in range(self.epochs_parallel):
            data_loader = data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            total_epoch_loss = 0.0
            for batch in data_loader:
                self.blackbox.zero_grad()

                x_whitebox_control = batch['x_whitebox_control'].float().to(self.device)
                x_whitebox_state = batch['x_whitebox_state'].float().to(self.device)
                ydot_whitebox = torch.zeros((self.batch_size, self.sequence_length, self.state_dim))\
                    .float().to(self.device)

                for time in range(self.sequence_length):
                    ydot_whitebox[:, time, :] = (
                            self.time_delta * whitebox_scaling *
                            self.whitebox.forward(x_whitebox_control[:, time, :], x_whitebox_state[:, time, :])
                    )

                x_init = batch['x_init'].float().to(self.device)
                x_pred = torch.cat((batch['x_pred'].float().to(self.device), ydot_whitebox), dim=2)

                _, hx_init = self.initializer.forward(x_init, return_state=True)

                y_blackbox = self.blackbox.forward(x_pred, hx=hx_init)
                y = y_blackbox + ydot_whitebox

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

        # Repeat end-to-end training but include feedback loop from output to whitebox input
        # In the previous end-to-end training, the auto-regression was inactive and whitebox received
        # true system state each time step.
        dataset = _End2EndDataset(
            control_seqs, state_seqs, unnormed_control_seqs, unnormed_state_seqs, self.sequence_length
        )
        for i in range(self.epochs_feedback):
            data_loader = data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)
            total_epoch_loss = 0.0
            for batch in data_loader:
                self.blackbox.zero_grad()

                # prepare whitebox input
                x_whitebox_control = batch['x_whitebox_control'].float().to(self.device)
                current_state = batch['x_whitebox_state'][:, 0, :].float().to(self.device)
                # prepare blackbox input
                x_init = batch['x_init'].float().to(self.device)
                _, hx_blackbox = self.initializer.forward(x_init, return_state=True)
                x_control = batch['x_pred'].float().to(self.device)
                # prepare model output
                y_est = torch.zeros((self.batch_size, self.sequence_length, self.state_dim)).to(self.device)

                for time in range(self.sequence_length):
                    ydot_whitebox = self.time_delta * whitebox_scaling * self.whitebox.forward(
                        x_whitebox_control[:, time, :],
                        current_state
                    )

                    x_blackbox = torch.cat((
                        x_control[:, time, :].unsqueeze(1),
                        ydot_whitebox.unsqueeze(1)
                    ), dim=2)
                    y_blackbox, hx_blackbox = self.blackbox.forward(x_blackbox, hx=hx_blackbox, return_state=True)

                    y_est[:, time, :] = y_blackbox.squeeze(1) + ydot_whitebox
                    current_state = denormalize_state(y_est[:, time, :])

                batch_loss = self.loss.forward(y_est, batch['y'].float().to(self.device))
                # reuse optimizer from previous training step, as it operates on same weights, and training may
                # benefit from already learned learning rates and momentum.
                total_epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_end2end.step()

            logger.info(f'Epoch {i + 1}/{self.epochs_feedback} - '
                        f'Epoch Loss (Auto-Regression): {total_epoch_loss}')
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


class HybridSerialMinimalAutoRegressiveLSTMModel(HybridSerialAutoRegressiveLSTMModel):
    def __init__(self, **kwargs):
        super().__init__(whitebox=first_principles.MinimalManeuveringEquations(**kwargs), **kwargs)


class HybridSerialPropulsionAutoRegressiveLSTMModel(HybridSerialAutoRegressiveLSTMModel):
    def __init__(self, **kwargs):
        super().__init__(whitebox=first_principles.PropulsionManeuveringEquations(**kwargs), **kwargs)


class HybridSerialBasicPelicanAutoRegressiveLSTMModel(HybridSerialAutoRegressiveLSTMModel):
    def __init__(self, **kwargs):
        super().__init__(whitebox=first_principles.BasicPelicanMotionEquations(**kwargs), **kwargs)


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
    def __init__(self, control_seqs, state_seqs, unnormed_control_seqs, unnormed_state_seqs, sequence_length):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]

        self.x_whitebox_control = None
        self.x_whitebox_state = None
        self.x_init = None
        self.x_pred = None
        self.ydot = None
        self.y = None

        self.__load_data(control_seqs, state_seqs, unnormed_control_seqs, unnormed_state_seqs)

    def __load_data(self, control_seqs, state_seqs, unnormed_control_seqs, unnormed_state_seqs):
        x_whitebox_control_seq = list()
        x_whitebox_state_seq = list()
        x_init_seq = list()
        x_pred_seq = list()
        ydot_seq = list()
        y_seq = list()

        for control, state, un_control, un_state \
                in zip(control_seqs, state_seqs, unnormed_control_seqs, unnormed_state_seqs):
            n_samples = int((control.shape[0] - self.sequence_length - 1) / (2 * self.sequence_length))

            x_whitebox_control = np.zeros((n_samples, self.sequence_length, self.control_dim))
            x_whitebox_state = np.zeros((n_samples, self.sequence_length, self.state_dim))
            x_init = np.zeros((n_samples, self.sequence_length, self.control_dim + self.state_dim))
            x_pred = np.zeros((n_samples, self.sequence_length, self.control_dim))
            ydot = np.zeros((n_samples, self.sequence_length, self.state_dim))
            y = np.zeros((n_samples, self.sequence_length, self.state_dim))

            for idx in range(n_samples):
                time = idx*self.sequence_length

                x_whitebox_control[idx, :, :] = un_control[time+self.sequence_length:time+2*self.sequence_length]
                x_whitebox_state[idx, :, :] = un_state[time+self.sequence_length-1:time+2*self.sequence_length-1]
                x_init[idx, :, :] = np.hstack((control[time:time+self.sequence_length, :],
                                               state[time:time+self.sequence_length, :]))
                x_pred[idx, :, :] = control[time+self.sequence_length:time+2*self.sequence_length, :]
                ydot[idx, :, :] = (state[time+self.sequence_length:time+2*self.sequence_length, :]
                                   - state[time+self.sequence_length-1:time+2*self.sequence_length-1, :])
                y[idx, :, :] = state[time+self.sequence_length:time+2*self.sequence_length, :]

            x_whitebox_control_seq.append(x_whitebox_control)
            x_whitebox_state_seq.append(x_whitebox_state)
            x_init_seq.append(x_init)
            x_pred_seq.append(x_pred)
            ydot_seq.append(ydot)
            y_seq.append(y)

        self.x_whitebox_control = np.vstack(x_whitebox_control_seq)
        self.x_whitebox_state = np.vstack(x_whitebox_state_seq)
        self.x_init = np.vstack(x_init_seq)
        self.x_pred = np.vstack(x_pred_seq)
        self.ydot = np.vstack(ydot_seq)
        self.y = np.vstack(y_seq)

    def __len__(self):
        return self.x_init.shape[0]

    def __getitem__(self, idx):
        return {'x_whitebox_control': self.x_whitebox_control[idx], 'x_whitebox_state': self.x_whitebox_state[idx],
                'x_init': self.x_init[idx], 'x_pred': self.x_pred[idx], 'ydot': self.ydot[idx], 'y': self.y[idx]}
