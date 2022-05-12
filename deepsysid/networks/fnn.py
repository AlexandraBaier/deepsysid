from typing import List

import torch.nn as nn


class DenseReLUNetwork(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, layers: List[int], dropout: float):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout

        nn_layers = []
        for input_dim, output_dim in zip([input_dim] + layers[:-1], layers):
            nn_layers.extend((
                nn.Linear(input_dim, output_dim), nn.Dropout(self.dropout), nn.ReLU()
            ))
        nn_layers.append(nn.Linear(layers[-1], self.output_dim))
        self.network = nn.Sequential(
            *nn_layers
        )

    def forward(self, x):
        """
        Output layer is linear.
        :param x: (batch, input_dim)
        :return: (batch, output_dim)
        """
        return self.network.forward(x)
