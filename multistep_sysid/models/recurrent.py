import json
import logging
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from . import base
from ..networks import loss, rnn
from .. import utils

logger = logging.getLogger()


class LSTMInitModel(base.DynamicIdentificationModel):
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
        self.epochs_predictor = kwargs['epochs_predictor']
        self.train_fraction_per_epoch = kwargs.get('train_fraction_per_epoch', None)

        if kwargs.get('loss') == 'mse':
            self.loss = nn.MSELoss().to(self.device)
        elif kwargs.get('loss') == 'msge':
            self.loss = loss.MSGELoss().to(self.device)
        else:
            raise ValueError(f'loss can only be "mse" or "msge"')

        self.predictor = rnn.BasicLSTM(
            input_dim=self.control_dim,
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=self.feedforward_dim + [self.state_dim],
            dropout=self.dropout
        ).to(self.device)

        self.initializer = rnn.BasicLSTM(
            input_dim=self.control_dim+self.state_dim,
            recurrent_dim=self.recurrent_dim,
            num_recurrent_layers=self.num_recurrent_layers,
            output_dim=self.feedforward_dim + [self.state_dim],
            dropout=self.dropout
        ).to(self.device)

        self.optimizer_pred = optim.Adam(self.predictor.parameters(), lr=self.learning_rate)
        self.optimizer_init = optim.Adam(self.initializer.parameters(), lr=self.learning_rate)

        self.control_mean = None
        self.control_stddev = None
        self.state_mean = None
        self.state_stddev = None

    def train(self, control_seqs, state_seqs, validator=None):
        self.predictor.train()
        self.initializer.train()

        self.control_mean, self.control_stddev = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_stddev = utils.mean_stddev(state_seqs)

        control_seqs = [utils.normalize(control, self.control_mean, self.control_stddev) for control in control_seqs]
        state_seqs = [utils.normalize(state, self.state_mean, self.state_stddev) for state in state_seqs]

        initializer_dataset = _InitializerDataset(control_seqs, state_seqs, self.sequence_length)
        # only use a fraction of the dataset in each epoch to train faster on large datasets
        if self.train_fraction_per_epoch:
            batches_per_epoch = self.train_fraction_per_epoch * int(len(initializer_dataset) / self.batch_size)
        else:
            batches_per_epoch = None

        time_start_init = time.time()
        for i in range(self.epochs_initializer):
            data_loader = data.DataLoader(initializer_dataset, self.batch_size, shuffle=True, drop_last=True)
            total_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                if batches_per_epoch and batch_idx >= batches_per_epoch:
                    break

                self.initializer.zero_grad()
                y = self.initializer.forward(batch['x'].float().to(self.device))
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_init.step()

            logger.info(f'Epoch {i + 1}/{self.epochs_initializer} - Epoch Loss (Initializer): {total_loss}')
        time_end_init = time.time()
        predictor_dataset = _PredictorDataset(control_seqs, state_seqs, self.sequence_length)
        # only use a fraction of the dataset in each epoch to train faster on large datasets
        if self.train_fraction_per_epoch:
            batches_per_epoch = self.train_fraction_per_epoch * int(len(predictor_dataset) / self.batch_size)
        else:
            batches_per_epoch = None
        time_start_pred = time.time()
        for i in range(self.epochs_predictor):
            data_loader = data.DataLoader(predictor_dataset, self.batch_size, shuffle=True, drop_last=True)
            total_loss = 0
            for batch_idx, batch in enumerate(data_loader):
                if batches_per_epoch and batch_idx >= batches_per_epoch:
                    break

                self.predictor.zero_grad()
                # Initialize predictor with state of initializer network
                _, hx = self.initializer.forward(batch['x0'].float().to(self.device), return_state=True)
                # Predict and optimize
                y = self.predictor.forward(batch['x'].float().to(self.device), hx=hx)
                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_pred.step()

            logger.info(f'Epoch {i + 1}/{self.epochs_predictor} - Epoch Loss (Predictor): {total_loss}')
            if validator:
                validation_error = validator(model=self, epoch=i)
                self.predictor.train(True)
                self.initializer.train(True)

                if validation_error:
                    validation_error_str = ','.join(
                        f'{err:.3E}' for err in validation_error
                    )
                    logger.info(f'Epoch {i + 1}/{self.epochs_predictor} - Validation Error: [{validation_error_str}]')
        time_end_pred = time.time()
        time_total_init = time_end_init - time_start_init
        time_total_pred = time_end_pred - time_start_pred
        logger.info(f'Training time for initializer {time_total_init}s and for predictor {time_total_pred}s')

    def simulate(self, initial_control, initial_state, control):
        self.initializer.eval()
        self.predictor.eval()

        initial_control = utils.normalize(initial_control, self.control_mean, self.control_stddev)
        initial_state = utils.normalize(initial_state, self.state_mean, self.state_stddev)
        control = utils.normalize(control, self.control_mean, self.control_stddev)

        with torch.no_grad():
            init_x = torch.from_numpy(np.hstack((initial_control[1:], initial_state[:-1])))\
                .unsqueeze(0).float().to(self.device)
            pred_x = torch.from_numpy(control).unsqueeze(0).float().to(self.device)

            _, hx = self.initializer.forward(init_x, return_state=True)
            y = self.predictor.forward(pred_x, hx=hx)
            y = y.cpu().detach().squeeze().numpy()

        y = utils.denormalize(y, self.state_mean, self.state_stddev)
        return y

    def save(self, file_path):
        torch.save(self.initializer.state_dict(), file_path[0])
        torch.save(self.predictor.state_dict(), file_path[1])
        with open(file_path[2], mode='w') as f:
            json.dump({
                'state_mean': self.state_mean.tolist(),
                'state_stddev': self.state_stddev.tolist(),
                'control_mean': self.control_mean.tolist(),
                'control_stddev': self.control_stddev.tolist()
            }, f)

    def load(self, file_path):
        self.initializer.load_state_dict(torch.load(file_path[0], map_location=self.device_name))
        self.predictor.load_state_dict(torch.load(file_path[1], map_location=self.device_name))
        with open(file_path[2], mode='r') as f:
            norm = json.load(f)
        self.state_mean = np.array(norm['state_mean'])
        self.state_stddev = np.array(norm['state_stddev'])
        self.control_mean = np.array(norm['control_mean'])
        self.control_stddev = np.array(norm['control_stddev'])

    def get_file_extension(self):
        return 'initializer.pth', 'predictor.pth', 'json'

    def get_parameter_count(self):
        # technically parameter counts of both networks are equal
        init_count = sum(p.numel() for p in self.initializer.parameters() if p.requires_grad)
        predictor_count = sum(p.numel() for p in self.predictor.parameters() if p.requires_grad)
        return init_count + predictor_count


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


class _PredictorDataset(data.Dataset):
    def __init__(self, control_seqs, state_seqs, sequence_length):
        self.sequence_length = sequence_length
        self.control_dim = control_seqs[0].shape[1]
        self.state_dim = state_seqs[0].shape[1]
        self.x0 = None
        self.x = None
        self.y = None
        self.__load_data(control_seqs, state_seqs)

    def __load_data(self, control_seqs, state_seqs):
        x0_seq = list()
        x_seq = list()
        y_seq = list()
        for control, state in zip(control_seqs, state_seqs):
            n_samples = int((control.shape[0] - 2 * self.sequence_length) / self.sequence_length)

            x0 = np.zeros((n_samples, self.sequence_length, self.control_dim + self.state_dim))
            x = np.zeros((n_samples, self.sequence_length, self.control_dim))
            y = np.zeros((n_samples, self.sequence_length, self.state_dim))

            for idx in range(n_samples):
                time = idx * self.sequence_length

                x0[idx, :, :] = np.hstack((control[time:time + self.sequence_length],
                                           state[time:time + self.sequence_length, :]))
                x[idx, :, :] = control[time + self.sequence_length:time + 2 * self.sequence_length, :]
                y[idx, :, :] = state[time + self.sequence_length:time + 2 * self.sequence_length, :]

            x0_seq.append(x0)
            x_seq.append(x)
            y_seq.append(y)

        self.x0 = np.vstack(x0_seq)
        self.x = np.vstack(x_seq)
        self.y = np.vstack(y_seq)

    def __len__(self):
        return self.x0.shape[0]

    def __getitem__(self, idx):
        return {'x0': self.x0[idx], 'x': self.x[idx], 'y': self.y[idx]}
