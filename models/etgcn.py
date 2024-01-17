import argparse
import torch
import torch.nn as nn
from utils.graph_conv import calculate_laplacian_with_self_loop


class ETGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, num_features: int, output_dim: int, bias: float = 0.0):
        super(ETGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._num_features = num_features
        self._output_dim = output_dim
        self._bias_init_value = bias
        self.register_buffer(
            "laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj))
        )
        self.weights = nn.Parameter(
            torch.FloatTensor(self._num_gru_units + self._num_features, self._output_dim)
        )
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes, num_features = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, num_features)
        inputs = inputs.reshape((batch_size, num_nodes, num_features))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape(
            (batch_size, num_nodes, self._num_gru_units)
        )
        # [x, h] (batch_size, num_nodes, num_gru_units + num_features)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        concatenation = concatenation.reshape(
            (num_nodes, (self._num_gru_units + num_features) * batch_size)
        )
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + num_features, batch_size)
        a_times_concat = a_times_concat.reshape(
            (num_nodes, self._num_gru_units + num_features, batch_size)
        )
        # A[x, h] (batch_size, num_nodes, num_gru_units + num_features)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + num_features)
        a_times_concat = a_times_concat.reshape(
            (batch_size * num_nodes, self._num_gru_units + num_features)
        )
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "num_features": self._num_features,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class ETGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, num_features: int, hidden_dim: int):
        super(ETGCNCell, self).__init__()
        self._input_dim = input_dim
        self._num_features = num_features
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = ETGCNGraphConvolution(
            self.adj, self._hidden_dim, self._num_features, self._hidden_dim * 2, bias=1.0
        )
        self.graph_conv2 = ETGCNGraphConvolution(
            self.adj, self._hidden_dim, self._num_features, self._hidden_dim
        )

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        # notice: output and hidden are the same
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "num_features": self._num_features, "hidden_dim": self._hidden_dim}


class ETGCN(nn.Module):
    def __init__(self, adj, num_features, hidden_dim: int, **kwargs):
        super(ETGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._num_features = num_features
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.etgcn_cell = ETGCNCell(adj, self._input_dim, self._num_features, self._hidden_dim)

    def forward(self, inputs):
        batch_size, sql_len, num_nodes, num_features = inputs.shape
        assert self._input_dim  == num_nodes
        assert self._num_features == num_features
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
        output = None
        for i in range(sql_len):
            output, hidden_state = self.etgcn_cell(inputs[:, i, :, :], hidden_state)
            output = output.reshape(batch_size, num_nodes, self._hidden_dim)
        return output

    @staticmethod
    def add_model_specific_arguments(parent_parser):
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--hidden_dim", type=int, default=64)
        return parser

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "num_features": self._num_features, "hidden_dim": self._hidden_dim}
