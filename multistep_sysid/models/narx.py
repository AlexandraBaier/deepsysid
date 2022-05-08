import json
import logging
from typing import List, Tuple

import numpy as np
import torch
from pydantic import BaseModel
from torch import optim
from torch.nn import functional
from torch.utils import data

from multistep_sysid.models import base
from multistep_sysid.networks.fnn import DenseReLUNetwork
from multistep_sysid import utils


logger = logging.getLogger()


class NARXDenseNetworkConfig(BaseModel):
    device_name: str = 'cpu'
    control_names: List[str]
    state_names: List[str]
    window_size: int
    learning_rate: float
    batch_size: int
    epochs: int
    layers: List[int]
    dropout: float


class NARXDenseNetwork(base.DynamicIdentificationModel):
    CONFIG = NARXDenseNetworkConfig

    def __init__(self, config: NARXDenseNetworkConfig):
        super().__init__(config)

        self.device_name = config.device_name
        self.device = torch.device(self.device_name)

        self.control_dim = len(config.control_names)
        self.state_dim = len(config.state_names)
        self.window_size = config.window_size

        self.learning_rate = config.learning_rate
        self.batch_size = config.batch_size
        self.epochs = config.epochs

        self.model = DenseReLUNetwork(
            input_dim=self.window_size*(self.control_dim + self.state_dim),
            output_dim=self.state_dim,
            layers=config.layers,
            dropout=config.dropout
        ).float().to(self.device)

        self.optimizer = optim.Adam(params=self.model.parameters(), lr=self.learning_rate)

        self.state_mean, self.state_std = None, None
        self.control_mean, self.control_std = None, None

    def train(self, control_seqs: List[np.ndarray], state_seqs: List[np.ndarray]):
        self.model.train()

        self.control_mean, self.control_std = utils.mean_stddev(control_seqs)
        self.state_mean, self.state_std = utils.mean_stddev(state_seqs)

        control_seqs = [utils.normalize(control, self.control_mean, self.control_std) for control in control_seqs]
        state_seqs = [utils.normalize(state, self.state_mean, self.state_std) for state in state_seqs]

        dataset = _Dataset(self.window_size, control_seqs, state_seqs)
        for i in range(self.epochs):
            data_loader = data.DataLoader(dataset, self.batch_size, shuffle=True, drop_last=True)
            total_loss = 0.0
            for batch_idx, batch in enumerate(data_loader):
                self.model.zero_grad()
                window_input = batch['window_input'].float().to(self.device)
                state_pred = self.model.forward(window_input)
                batch_loss = functional.mse_loss(state_pred, batch['state_true'].float().to(self.device))
                total_loss += batch_loss.item()
                batch_loss.backward()
                self.optimizer.step()

            logger.info(f'Epoch {i + 1}/{self.epochs} - Epoch Loss: {total_loss}')

    def simulate(self, initial_control: np.ndarray, initial_state: np.ndarray, control: np.ndarray) -> np.ndarray:
        self.model.eval()

        initial_control = utils.normalize(initial_control, self.control_mean, self.control_std)
        initial_state = utils.normalize(initial_state, self.state_mean, self.state_std)
        control = utils.normalize(control, self.control_mean, self.control_std)

        total_time = control.shape[0]
        states = []
        with torch.no_grad():
            state_window = torch.from_numpy(initial_state.flatten()).float().to(self.device).unsqueeze(0)
            control_window = torch.from_numpy(initial_control.flatten()).float().to(self.device).unsqueeze(0)
            control_in = torch.from_numpy(control).float().to(self.device)

            for time in range(total_time):
                control_window = torch.cat((
                    control_window[:, self.control_dim:], control_in[time, :].unsqueeze(0)
                ), dim=1)
                state = self.model.forward(torch.cat((control_window, state_window), dim=1))
                state_np = state.detach().cpu().numpy().squeeze()
                states.append(state_np)
                state_window = torch.cat((state_window[:, self.state_dim:], state), dim=1)

        return utils.denormalize(np.vstack(states), self.state_mean, self.state_std)

    def save(self, file_path: Tuple[str, str]):
        torch.save(self.model.state_dict(), file_path[0])
        with open(file_path[1], mode='w') as f:
            json.dump({
                'state_mean': self.state_mean.tolist(),
                'state_std': self.state_std.tolist(),
                'control_mean': self.control_mean.tolist(),
                'control_std': self.control_std.tolist()
            }, f)

    def load(self, file_path: Tuple[str, str]):
        self.model.load_state_dict(torch.load(file_path[0], map_location=self.device_name))
        with open(file_path[1], mode='r') as f:
            norm = json.load(f)
        self.state_mean = np.array(norm['state_mean'])
        self.state_std = np.array(norm['state_std'])
        self.control_mean = np.array(norm['control_mean'])
        self.control_std = np.array(norm['control_std'])

    def get_file_extension(self) -> Tuple[str, str]:
        return 'pth', 'json'

    def get_parameter_count(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)


class _Dataset(data.Dataset):
    def __init__(self, window_size, control_seqs, state_seqs):
        self.window_input = []
        self.state_true = []
        for control, state in zip(control_seqs, state_seqs):
            for time in range(window_size, control.shape[0] - 1, int(window_size/4)+1):
                self.window_input.append(
                    np.concatenate((
                        control[time - window_size + 1: time + 1, :].flatten(),
                        state[time - window_size: time, :].flatten()
                    ))
                )
                self.state_true.append(
                    state[time + 1, :]
                )
        self.window_input = np.vstack(self.window_input)
        self.state_true = np.vstack(self.state_true)

    def __len__(self):
        return self.window_input.shape[0]

    def __getitem__(self, idx):
        return dict(window_input=self.window_input[idx], state_true=self.state_true[idx])
