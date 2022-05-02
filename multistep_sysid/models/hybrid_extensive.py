import abc
import json
import logging

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from ..networks import first_principles
from . import base
from ..networks import rnn, loss
from .. import utils
from ..networks.first_principles import NoOpFirstPrinciples

logger = logging.getLogger()


class HybridSerialCombinedLSTMModel(base.DynamicIdentificationModel, abc.ABC):
    def __init__(self, physical, device_name='cpu', semiphysical_bias=True, **kwargs):
        super().__init__(**kwargs)

        self.device_name = device_name
        self.device = torch.device(device_name)

        self.control_dim = len(kwargs['control_names'])
        self.state_dim = len(kwargs['state_names'])
        self.time_delta = float(kwargs['dt'])

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

        self.physical = physical.to(self.device).float()

        self.semiphysical = nn.Linear(
            in_features=self.get_semiphysical_in_features(),
            out_features=self.state_dim,
            bias=semiphysical_bias
        ).to(self.device)
        self.semiphysical.weight.requires_grad = False
        if self.semiphysical.bias is None:
            self.semiphysical.bias = nn.Parameter(
                torch.from_numpy(np.array(0.0)).float().to(self.device)
            )
        self.semiphysical.bias.requires_grad = False

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
        self.initializer.train()
        self.physical.train()
        self.semiphysical.train()

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
        scale_acc = lambda x: x / state_stddev

        # Train linear model
        self.train_semiphysical(semiphysical_in_seqs, un_control_seqs, un_state_seqs, state_seqs)

        initializer_dataset = _InitializerDataset(control_seqs, state_seqs, self.sequence_length)
        for i in range(self.epochs_initializer):
            data_loader = data.DataLoader(initializer_dataset, self.batch_size, shuffle=True, drop_last=True)
            total_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                self.initializer.zero_grad()
                y = self.initializer.forward(batch['x'].float().to(self.device))
                batch_loss = F.mse_loss(y, batch['y'].float().to(self.device))
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
                x_physical_control = batch['x_control_unnormed'].float().to(self.device)
                x_physical_state = batch['x_state_unnormed'].float().to(self.device)
                y_whitebox = torch.zeros((self.batch_size, self.sequence_length, self.state_dim))\
                    .float().to(self.device)

                for time in range(self.sequence_length):
                    y_semiphysical = self.semiphysical.forward(x_semiphysical[:, time, :])
                    ydot_physical = scale_acc(
                        self.physical.forward(x_physical_control[:, time, :], x_physical_state[:, time, :])
                    )
                    y_whitebox[:, time, :] = y_semiphysical + self.time_delta * ydot_physical

                x_init = batch['x_init'].float().to(self.device)
                x_pred = torch.cat((batch['x_pred'].float().to(self.device), y_whitebox), dim=2)  # serial connection

                _, hx_init = self.initializer.forward(x_init, return_state=True)

                y_blackbox = self.blackbox_forward(x_pred, y_whitebox, hx=hx_init)
                y = y_blackbox + y_whitebox

                batch_loss = self.loss.forward(y, batch['y'].float().to(self.device))
                total_epoch_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer_end2end.step()

            logger.info(f'Epoch {i + 1}/{self.epochs_parallel} - Epoch Loss (Parallel): {total_epoch_loss}')

            if validator:
                validation_error = validator(model=self, epoch=i)
                self.blackbox.train(True)
                self.initializer.train(True)
                self.semiphysical.train(True)
                self.physical.train(True)

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
                y_est = torch.zeros((self.batch_size, self.sequence_length, self.state_dim)).float().to(self.device)

                x_init = batch['x_init'].float().to(self.device)
                _, hx_init = self.initializer.forward(x_init, return_state=True)

                for time in range(self.sequence_length):
                    y_semiphysical = self.semiphysical.forward(
                        normalize_semiphysical(
                            self.expand_semiphysical_input(x_control_unnormed[:, time, :], current_state)
                        )
                    )
                    ydot_physical = scale_acc(
                        self.physical.forward(x_control_unnormed[:, time, :], current_state)
                    )
                    y_whitebox = y_semiphysical + self.time_delta * ydot_physical

                    y_blackbox, hx_init = self.blackbox_forward(
                        torch.cat((x_pred[:, time, :].unsqueeze(1), y_whitebox.unsqueeze(1)), dim=2),
                        y_whitebox.unsqueeze(1),
                        hx=hx_init, return_state=True)
                    current_state = y_blackbox.squeeze(1) + y_whitebox
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
                self.initializer.train(True)
                self.semiphysical.train(True)
                self.physical.train(True)

                if validation_error:
                    validation_error_str = ','.join(
                        f'{err:.3E}' for err in validation_error
                    )
                    logger.info(f'Epoch {i + 1}/{self.epochs_feedback} - Validation Error: [{validation_error_str}]')

    def simulate(self, initial_control, initial_state, control, return_whitebox_blackbox=False, threshold=np.infty):
        self.blackbox.eval()
        self.initializer.eval()
        self.semiphysical.eval()
        self.physical.eval()

        sp_mean = torch.from_numpy(self.semiphysical_in_mean).float().to(self.device)
        sp_stddev = torch.from_numpy(self.semiphysical_in_stddev).float().to(self.device)
        normalize_semiphysical = lambda x: (x - sp_mean) / sp_stddev
        state_mean = torch.from_numpy(self.state_mean).float().to(self.device)
        state_stddev = torch.from_numpy(self.state_stddev).float().to(self.device)
        denormalize_state = lambda x: (x * state_stddev) + state_mean
        scale_acc = lambda x: x / state_stddev

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
                y_semiphysical = self.semiphysical.forward(normalize_semiphysical(
                    self.expand_semiphysical_input(x_control_un[:, time, :], current_state)))
                ydot_physical = scale_acc(
                    self.physical.forward(x_control_un[:, time, :], current_state)
                )
                y_whitebox = y_semiphysical + self.time_delta * ydot_physical

                x_blackbox = torch.cat((x_pred[:, time, :], y_whitebox), dim=1).unsqueeze(1)
                y_blackbox, hx = self.blackbox_forward(x_blackbox, None, hx=hx, return_state=True)
                y_blackbox = torch.clamp(y_blackbox, -threshold, threshold)
                y_est = y_blackbox.squeeze(1) + y_whitebox
                current_state = denormalize_state(y_est)
                y[time, :] = current_state.cpu().detach().numpy()
                whitebox[time, :] = y_whitebox.cpu().detach().numpy()
                blackbox[time, :] = y_blackbox.squeeze(1).cpu().detach().numpy()

        if return_whitebox_blackbox:
            return y, whitebox, blackbox
        else:
            return y

    def save(self, file_path):
        torch.save(self.semiphysical.state_dict(), file_path[0])
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
        self.semiphysical.load_state_dict(torch.load(file_path[0], map_location=self.device_name))
        self.semiphysical.weight.requires_grad = False
        self.semiphysical.bias.requires_grad = False
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
        return 'semi-physical.pth', 'blackbox.pth', 'initializer.pth', 'json'

    def get_parameter_count(self):
        semiphysical_count = sum(p.numel() for p in self.semiphysical.parameters())
        blackbox_count = sum(p.numel() for p in self.blackbox.parameters() if p.requires_grad)
        initializer_count = sum(p.numel() for p in self.initializer.parameters() if p.requires_grad)
        return semiphysical_count + blackbox_count + initializer_count

    def train_semiphysical(self, semiphysical_in_seqs, un_control_seqs, un_state_seqs, state_seqs):
        # semi-physical component is trained to predict state minus first-principles prediction
        train_x, train_y = [], []
        for sp_in, un_control, un_state, state \
                in zip(semiphysical_in_seqs, un_control_seqs, un_state_seqs, state_seqs):
            ydot_physical = self.physical.forward(
                torch.from_numpy(un_control[1:, :]).float().to(self.device),
                torch.from_numpy(un_state[:-1, :]).float().to(self.device)
            ).cpu().detach().numpy()
            ydot_physical = ydot_physical / self.state_stddev

            train_x.append(sp_in)
            train_y.append(state[1:] - (self.time_delta * ydot_physical))

        train_x = np.vstack(train_x)
        train_y = np.vstack(train_y)

        # No intercept in linear time invariant systems
        regressor = LinearRegression(fit_intercept=False)
        regressor.fit(train_x, train_y)
        linear_fit = r2_score(regressor.predict(train_x), train_y, multioutput='uniform_average')
        logger.info(f'Whitebox R2 Score: {linear_fit}')

        self.semiphysical.weight = nn.Parameter(torch.from_numpy(regressor.coef_).float().to(self.device),
                                                requires_grad=False)
        if isinstance(regressor.intercept_, float):
            self.semiphysical.bias = nn.Parameter(
                torch.from_numpy(np.array(regressor.intercept_)).float().to(self.device),
                requires_grad=False
            )
        else:
            self.semiphysical.bias = nn.Parameter(
                torch.from_numpy(regressor.intercept_).float().to(self.device),
                requires_grad=False
            )

    def blackbox_forward(self, x_pred, y_wb, hx=None, return_state=False):
        # You can overwrite to blackbox_forward to enable different treatment of inputs to model.
        # TODO: x_pred should instead be x_control.
        # TODO: I don't remember the purpose of this function. Probably to generalize the code in some way?
        return self.blackbox.forward(x_pred, hx=hx, return_state=return_state)

    @abc.abstractmethod
    def get_semiphysical_in_features(self):
        pass

    @abc.abstractmethod
    def expand_semiphysical_input(self, control, state):
        pass


class HybridSerialCombinedLinearLSTMModel(HybridSerialCombinedLSTMModel, abc.ABC):
    def get_semiphysical_in_features(self):
        return self.control_dim + self.state_dim

    def expand_semiphysical_input(self, control, state):
        return torch.cat((control, state), dim=1)


class HybridSerialCombinedQuadraticLSTMModel(HybridSerialCombinedLSTMModel, abc.ABC):
    def get_semiphysical_in_features(self):
        return 2 * (self.control_dim + self.state_dim)

    def expand_semiphysical_input(self, control, state):
        return torch.cat((
            control,
            state,
            control * control,
            state * state
        ), dim=1)


class HybridSerialCombinedBlankeLSTMModel(HybridSerialCombinedLSTMModel, abc.ABC):
    def train_semiphysical(self, semiphysical_in_seqs, un_control_seqs, un_state_seqs, state_seqs):
        # semi-physical component is trained to predict state minus first-principles prediction
        train_x, train_y = [], []
        for sp_in, un_control, un_state, state \
                in zip(semiphysical_in_seqs, un_control_seqs, un_state_seqs, state_seqs):
            ydot_physical = self.physical.forward(
                torch.from_numpy(un_control[1:, :]).float().to(self.device),
                torch.from_numpy(un_state[:-1, :]).float().to(self.device)
            ).cpu().detach().numpy()
            ydot_physical = ydot_physical / self.state_stddev

            train_x.append(sp_in)
            train_y.append(state[1:] - (self.time_delta * ydot_physical))

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

        self.semiphysical.weight = nn.Parameter(torch.from_numpy(weight).float().to(self.device), requires_grad=False)
        self.semiphysical.bias = nn.Parameter(torch.from_numpy(bias).float().to(self.device), requires_grad=False)

    def get_semiphysical_in_features(self):
        return 20# + self.control_dim

    def expand_semiphysical_input(self, control, state):
        # problem: need to include base values for state prediction
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


class HybridSerialFirstPrinciplesOnlyLSTMModel(HybridSerialCombinedLSTMModel, abc.ABC):
    def train_semiphysical(self, semiphysical_in_seqs, un_control_seqs, un_state_seqs, state_seqs):
        weight = np.zeros((self.state_dim, self.get_semiphysical_in_features()))
        bias = np.zeros((self.state_dim,))
        self.semiphysical.weight = nn.Parameter(torch.from_numpy(weight).float().to(self.device), requires_grad=False)
        self.semiphysical.bias = nn.Parameter(torch.from_numpy(bias).float().to(self.device), requires_grad=False)

    def get_semiphysical_in_features(self):
        return 0

    def expand_semiphysical_input(self, control, state):
        return torch.zeros(control.shape[0], control.shape[0], 0, dtype=control.dtype)


class HybridSerialSemiphysicalOnlyLSTMModel(HybridSerialCombinedLSTMModel, abc.ABC):
    def __init__(self, **kwargs):
        super().__init__(
            physical=NoOpFirstPrinciples,
            **kwargs
        )


class HybridSerialCombinedMinimalLinearLSTMModel(HybridSerialCombinedLinearLSTMModel):
    def __init__(self, **kwargs):
        super().__init__(
            physical=first_principles.MinimalManeuveringEquations(**kwargs),
            semiphysical_bias=False,
            **kwargs
        )


class HybridSerialCombinedPropulsionLinearLSTMModel(HybridSerialCombinedLinearLSTMModel):
    def __init__(self, **kwargs):
        super().__init__(
            physical=first_principles.PropulsionManeuveringEquations(**kwargs),
            semiphysical_bias=False,
            **kwargs
        )


class HybridSerialCombinedMinimalBlankeLSTMModel(HybridSerialCombinedBlankeLSTMModel):
    def __init__(self, **kwargs):
        super().__init__(
            physical=first_principles.MinimalManeuveringEquations(**kwargs),
            **kwargs
        )


class HybridSerialCombinedPropulsionBlankeLSTMModel(HybridSerialCombinedBlankeLSTMModel):
    def __init__(self, **kwargs):
        super().__init__(
            physical=first_principles.PropulsionManeuveringEquations(**kwargs),
            **kwargs
        )


class HybridSerialCombinedBasicPelicanLinearLSTMModel(HybridSerialCombinedLinearLSTMModel):
    def __init__(self, **kwargs):
        super().__init__(
            physical=first_principles.BasicPelicanMotionEquations(**kwargs),
            **kwargs
        )


class HybridSerialCombinedBasicPelicanQuadraticLSTMModel(HybridSerialCombinedQuadraticLSTMModel):
    def __init__(self, **kwargs):
        super().__init__(
            physical=first_principles.BasicPelicanMotionEquations(**kwargs),
            **kwargs
        )


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
        self.x_state_unnormed = None

        self.__load_data(control_seqs, state_seqs, semiphysical_in_seqs, un_control_seqs, un_state_seqs)

    def __load_data(self, control_seqs, state_seqs, semiphysical_in_seqs, un_control_seqs, un_state_seqs):
        x_semiphysical_in_seq = list()
        x_init_seq = list()
        x_pred_seq = list()
        ydot_seq = list()
        y_seq = list()
        initial_state_seq = list()
        x_control_unnormed_seq = list()
        x_state_unnormed_seq = list()

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
            x_state_unnormed = np.zeros((n_samples, self.sequence_length, self.state_dim))

            for idx in range(n_samples):
                time = idx * self.sequence_length

                # semiphysical input already is preprocessed with correct time
                x_semiphysical_in[idx, :, :] = sp_in[time + self.sequence_length: time + 2 * self.sequence_length, :]
                x_init[idx, :, :] = np.hstack((control[time:time + self.sequence_length, :],
                                               state[time:time + self.sequence_length, :]))
                x_pred[idx, :, :] = control[time + self.sequence_length:time + 2 * self.sequence_length, :]
                ydot[idx, :, :] = (state[time + self.sequence_length:time + 2 * self.sequence_length, :]
                                   - state[time + self.sequence_length - 1:time + 2 * self.sequence_length - 1, :])
                y[idx, :, :] = state[time + self.sequence_length:time + 2 * self.sequence_length, :]
                initial_state[idx, :] = un_state[time + self.sequence_length - 1, :]
                x_control_unnormed[idx, :, :] = un_control[time + self.sequence_length
                                                           :time + 2 * self.sequence_length, :]
                x_state_unnormed[idx, :, :] = un_state[time + self.sequence_length - 1
                                                       :time + 2 * self.sequence_length - 1, :]

            x_semiphysical_in_seq.append(x_semiphysical_in)
            x_init_seq.append(x_init)
            x_pred_seq.append(x_pred)
            ydot_seq.append(ydot)
            y_seq.append(y)
            initial_state_seq.append(initial_state)
            x_control_unnormed_seq.append(x_control_unnormed)
            x_state_unnormed_seq.append(x_state_unnormed)

        self.x_semiphysical_in = np.vstack(x_semiphysical_in_seq)
        self.x_init = np.vstack(x_init_seq)
        self.x_pred = np.vstack(x_pred_seq)
        self.ydot = np.vstack(ydot_seq)
        self.y = np.vstack(y_seq)
        self.initial_state = np.vstack(initial_state_seq)
        self.x_control_unnormed = np.vstack(x_control_unnormed_seq)
        self.x_state_unnormed = np.vstack(x_state_unnormed_seq)

    def __len__(self):
        return self.x_init.shape[0]

    def __getitem__(self, idx):
        return {'x_semiphysical_in': self.x_semiphysical_in[idx], 'x_init': self.x_init[idx],
                'x_pred': self.x_pred[idx], 'ydot': self.ydot[idx], 'y': self.y[idx],
                'initial_state': self.initial_state[idx], 'x_control_unnormed': self.x_control_unnormed[idx],
                'x_state_unnormed': self.x_state_unnormed[idx]}
