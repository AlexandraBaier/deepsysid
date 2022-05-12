import torch.nn as nn
import torch.nn.functional as F


class BasicLSTM(nn.Module):
    def __init__(self, input_dim, recurrent_dim, num_recurrent_layers, output_dim, dropout):
        super().__init__()

        self.num_recurrent_layers = num_recurrent_layers
        self.recurrent_dim = recurrent_dim

        self.predictor_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=recurrent_dim,
            num_layers=num_recurrent_layers,
            dropout=dropout,
            batch_first=True
        )

        layer_dim = [recurrent_dim] + output_dim
        self.out = nn.ModuleList([nn.Linear(in_features=layer_dim[i-1], out_features=layer_dim[i])
                                  for i in range(1, len(layer_dim))])

        for name, param in self.predictor_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        for layer in self.out:
            nn.init.xavier_normal_(layer.weight)

    def forward(self, x_pred, y_init=None, hx=None, return_state=False):
        if y_init is not None:
            h0 = y_init.new_zeros((self.num_recurrent_layers, y_init.shape[0], self.recurrent_dim))
            for i in range(self.num_recurrent_layers):
                h0[i, :, :y_init.shape[1]] = y_init
            c0 = h0.new_zeros(h0.shape)
            hx = (h0, c0)

        x, (h0, c0) = self.predictor_lstm(x_pred, hx)
        for layer in self.out[:-1]:
            x = F.relu(layer(x))
        x = self.out[-1](x)

        if return_state:
            return x, (h0, c0)
        else:
            return x


class LinearOutputLSTM(nn.Module):
    def __init__(self, input_dim, recurrent_dim, num_recurrent_layers, output_dim, dropout):
        super().__init__()

        self.num_recurrent_layers = num_recurrent_layers
        self.recurrent_dim = recurrent_dim

        self.predictor_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=recurrent_dim,
            num_layers=num_recurrent_layers,
            dropout=dropout,
            batch_first=True
        )

        self.out = nn.Linear(in_features=recurrent_dim, out_features=output_dim, bias=False)

        for name, param in self.predictor_lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_normal_(param)

        nn.init.xavier_normal_(self.out.weight)

    def forward(self, x_pred, hx=None, return_state=False):
        x, (h0, c0) = self.predictor_lstm.forward(x_pred, hx)
        x = self.out.forward(x)

        if return_state:
            return x, (h0, c0)
        else:
            return x
