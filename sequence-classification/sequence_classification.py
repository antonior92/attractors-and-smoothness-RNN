import numpy as np
import torch
import torch.nn as nn
import sys
sys.path.append("../")
from rnn_wrap import RNNwrap


# Define RNN model
class MyRNN(nn.Module):
    """RNN model for temporal order problem"""
    def __init__(self, n_classes, input_size, rnn_type='LSTM', hidden_size=50, num_layers=1, K=100, init='henaff', bias=True):
        super(MyRNN, self).__init__()
        self.input_size = input_size
        self.output_size = n_classes
        self.n_classes = n_classes
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.rnn = RNNwrap(input_size, hidden_size, rnn_type, num_layers, K, init, bias)
        self.linear = nn.Linear(hidden_size, n_classes)

    def init_hidden(self, batch_size, device, rnn_type):
        return self.rnn.initial_state(batch_size, device, rnn_type)

    def forward(self, inp, state):
        """Forward propagate through RNN.
        Parameters
        ----------
        inp: torch.Tensor
            Input vector of shape (seq_len, batch_size, n_symbols + n_distractors)
        state: torch.Tensor or tuple of torch.Tensor
            Depend on rnn_type
        Returns
        -------
        out: torch.Tensor
            Output vector of shape (seq_len, batch_size, n_classes)
        next_state: torch.Tensor or tuple of torch.Tensor
            Depend on rnn_type
        """
        o1, state_next = self.rnn(inp, state)
        out = self.linear(o1)
        return out, state_next


# Generate dataset
def data_generator(n_symbols, n_distractors, interval_positions, seq_len, n_batches, seed, device):
    rng = np.random.RandomState(seed)
    n_positions = len(interval_positions)  # Number of positions with non-distractor symbols
    positions_symbols = np.zeros(n_positions, dtype=int)  # Initialize vector with positions non-distractor symbols
    u = np.zeros((seq_len, n_batches, n_symbols + n_distractors))  # Initialize input
    y = np.zeros((n_batches,), dtype=np.int64)  # Initialize index indicating the class of the output
    for b in range(n_batches):
        # Generate input sequence with distractor symbols
        for t in range(seq_len):
            u[t, b, n_symbols + rng.randint(n_distractors)] = 1
        # Replace distractor symbols with non-distractor symbols at the appropriate positions
        for i in range(n_positions):
            positions_symbols[i] = rng.randint(interval_positions[i][0], interval_positions[i][1])
            index_symbol = rng.randint(n_symbols)
            y[b] = y[b] + (index_symbol * (n_symbols ** (n_positions - i - 1)))
            u[positions_symbols[i], b, index_symbol] = 1
            u[positions_symbols[i], b, n_symbols:n_symbols + n_distractors] = np.zeros(n_distractors)
    # Convert to PyTorch tensor
    u = torch.tensor(u, dtype=torch.float32, device=device)
    y = torch.tensor(y, dtype=torch.long, device=device)
    return {"u": u, "y": y, "n_classes": n_symbols ** n_positions, "input_size": n_distractors+n_symbols}


def generate_const_input(model, device):
    """Return constant inputs shape = (n_different_inputs, input_size)"""
    return torch.eye(model.input_size, dtype=torch.float32, device=device)


# Return generator that gives the data
def data_loader(data, bs, n):
    def _data_loader():
        for i in range(0, n, bs):
            # Get batch
            s = slice(i, min(i+bs, n))
            u = data["u"][:, s, :]
            y = data["y"][s]
            yield u, y
    return _data_loader


class Criterion(object):
    def __init__(self):
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")
        self.softmax = nn.Softmax()

    def __call__(self, pred, target, return_num_correct_values=False):
        prediction = pred[-1, :, :]
        loss = self.cross_entropy(prediction, target)
        if return_num_correct_values:
            pred_probabilities = self.softmax(prediction)  # Get probabilities
            _, indice = torch.max(pred_probabilities, 1)  # Get maximum probability
            cv = indice == target
            return loss, torch.sum(cv)
        else:
            return loss