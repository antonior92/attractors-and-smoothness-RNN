import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append("../")
from rnn_wrap import RNNwrap


# Define RNN model
class MyRNN(nn.Module):
    """RNN model for sine generation"""
    def __init__(self, rnn_type='LSTM', hidden_size=200, num_layers=1, K=100, init='henaff'):
        super(MyRNN, self).__init__()
        input_size = 1
        output_size = 1
        self.rnn = RNNwrap(input_size, hidden_size, rnn_type, num_layers, K, init)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, inp, state):
        """Forward propagate throught RNN.

        Parameters
        ----------
        inp: torch.Tensor
            Input vector of shape (seq_len, batch_size, 1)
        state: tuple of torch.Tensor
            Tuple (h_0, c_0), both torch.Tensor with shape: (num_layers, batch_size, hidden_size).

        Returns
        -------
        out: torch.Tensor
            Output vector of shape (seq_len, batch_size, 1)
        state_next: tuple of torch.Tensor
            Tuple (h_0, c_0), both torch.Tensor with shape: (num_layers, batch_size, hidden_size).
        """
        o1, state_next = self.rnn(inp, state)
        out = self.linear(o1)
        return out, state_next


# Generate dataset
def data_generator(min_freq, max_freq, n_sequences, seq_len, device):
    # Generate input, shape = (seq_len, n_sequences, 1)
    freqs = np.linspace(min_freq, max_freq, n_sequences)  # Define frequencies
    u = np.tile(freqs, (seq_len, 1, 1)).transpose((0, 2, 1))
    # Generate output, shape = (seq_len, n_sequences, 1)
    phase = np.cumsum(u, axis=0)
    y = np.sin(phase)
    # Generate pytorch tensor
    u = torch.tensor(u, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    return {"u": u, "y": y, "freqs": freqs}


# Generate initial state
def initial_state_generator(data, model, device, rnn_type, zero_initial_condition=True):
    # Define initial state
    bs = data["u"].size(1)

    with torch.no_grad():
        zero_state = model.rnn.initial_state(bs, device, rnn_type)
        if zero_initial_condition:
            return zero_state
        _, initial_state = model(data["u"] + data["y"], zero_state)


    if rnn_type == 'LSTM':
        h0 = torch.tensor(initial_state[0], dtype=torch.float32,
                          device=device, requires_grad=True)
        c0 = torch.tensor(initial_state[1], dtype=torch.float32,
                          device=device, requires_grad=True)
        return h0, c0
    else:
        x0 = torch.tensor(initial_state, dtype=torch.float32,
                          device=device, requires_grad=True)
        return x0


def generate_const_input(model, device):
    """Return constant inputs shape = (n_different_inputs, input_size)"""
    n_inputs = 10
    freqs = np.linspace(np.pi/16, np.pi/8, n_inputs)
    return torch.tensor(freqs, dtype=torch.float32, device=device).reshape((n_inputs, 1))

# Return generator that gives the data
def data_loader(data, bs, n):
    def _data_loader():
        for i in range(0, n, bs):
            s = slice(i, min(i+bs, n))
            u_batch = data['u'][:, s, :]
            y_batch = data['y'][:, s, :]
            yield u_batch, y_batch
    return _data_loader
