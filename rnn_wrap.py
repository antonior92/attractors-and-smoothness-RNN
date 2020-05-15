import numpy as np
import torch
import torch.nn as nn
from expRNN.orthogonal import OrthogonalRNN


class RNNwrap(nn.Module):
    def __init__(self, input_size, hidden_size, rnn_type='LSTM', num_layers=1, K=100, init='henaff', bias=True):
        super(RNNwrap, self).__init__()
        self.hidden_size = hidden_size
        if rnn_type in ['RNN', 'LSTM', 'GRU']:
            self.num_layers = num_layers
            self.rnn = getattr(nn, rnn_type)(input_size, hidden_size, num_layers, bias=bias)
        # rnn_type = 'dtriv', 'cayley' should also be possible alternatives here to 'exprnn' (untested)
        elif rnn_type == 'exprnn':
            self.rnn = OrthogonalRNN(input_size, hidden_size, K, init, rnn_type)
        else:
            raise ValueError("Unknown rnn_type")

    def forward(self, inp, state):
        return self.rnn(inp, state)

    def initial_state(self, bs, device, rnn_type, fn=torch.zeros):
        with torch.no_grad():
            if rnn_type == 'LSTM':
                zero_state = (fn(self.num_layers, bs, self.hidden_size, device=device),
                              fn(self.num_layers, bs, self.hidden_size, device=device))
            # rnn_type = 'dtriv', 'cayley' should also be possible alternatives here to 'exprnn' (untested)
            elif rnn_type == 'exprnn':
                zero_state = fn(bs, self.hidden_size, device=device)
            else:
                zero_state = fn(self.num_layers, bs, self.hidden_size, device=device)
        return zero_state


def stabilize_lstm(lstm):
    # One set of weights satisfying stability requirement
    recur_weights = lstm.weight_hh_l0.data
    wi, wf, wz, wo = recur_weights.chunk(4, 0)

    trimmed_wi = wi * 0.395 / torch.sum(torch.abs(wi), 0)
    trimmed_wf = wf * 0.155 / torch.sum(torch.abs(wf), 0)
    trimmed_wz = wz * 0.099 / torch.sum(torch.abs(wz), 0)
    trimmed_wo = wo * 0.395 / torch.sum(torch.abs(wo), 0)
    new_recur_weights = torch.cat([trimmed_wi, trimmed_wf, trimmed_wz, trimmed_wo], 0)
    lstm.weight_hh_l0.set_(new_recur_weights)

    # Also trim the input to hidden weight for the forget gate
    ih_weights = lstm.weight_ih_l0.data
    ui, uf, uz, uo = ih_weights.chunk(4, 0)
    trimmed_uf = uf * 0.25 / torch.sum(torch.abs(uf), 0)
    new_ih_weights = torch.cat([ui, trimmed_uf, uz, uo], 0)
    lstm.weight_ih_l0.set_(new_ih_weights)

    lstm.flatten_parameters()
    return lstm