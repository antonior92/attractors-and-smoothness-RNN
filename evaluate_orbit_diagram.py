import torch


def state_0_loader_per_batch(state_0_full, bs, n, rnn_type):
    def _state_0_loader():
        # Define states
        if rnn_type == 'LSTM':
            h0, c0 = state_0_full
        else:
            x0 = state_0_full
        for i in range(0, n, bs):
            s = slice(i, min(i+bs, n))
            # Get initial conditions
            if rnn_type == 'LSTM':
                yield h0[:, s, :], c0[:, s, :]
            elif rnn_type == 'EURNN':
                yield x0[s, :, :]
            elif rnn_type in ['exprnn', 'dtriv', 'cayley']:
                yield x0[s, :]
            else:
                yield x0[:, s, :]
    return _state_0_loader


class EvaluateOrbitDiagram(object):
    """Evaluate orbit diagram. It should  generate a total of ``n_points``. ``const_inp`` is a
     tensor of size (n_different_inputs, input_size). ``model`` is  the pytorch module that should
      be called as ``model(inp, state)``. ``transformation`` is either None or a pytorch module:
       ``transformation(torch.Tensor shape = (seq_len-burnout, batch_size, output_size)) -> vector_size``.
      Notice that ``vector_size`` refer to the size after the transformation. If use_feedback=True, reconect
      auto regressive terms to generate the orbit."""

    def __init__(self, n_points, const_inp, model, device,
                 batch_size, n_sequences, seq_len, burnout,
                 vector_size, std, transformation, rnn_type,
                 use_feedback=False, squeeze=True):  # Set squeeze = True for lm example and squeeze = False for sine-wave example
        # Input
        self.n_inputs = const_inp.size(0)
        const_inputs = const_inp.repeat(seq_len, n_sequences, 1)
        # Random initial states
        sz = self.n_inputs * n_sequences
        state_0 = model.rnn.initial_state(sz, device, rnn_type, lambda *args, **kwargs: std * torch.randn(args, **kwargs))
        self.state_0_load = state_0_loader_per_batch(state_0, batch_size, sz, rnn_type)
        # Compute trajectory
        use_feedback_multiplier = 2 if use_feedback else 1
        self.trajectory = torch.zeros((seq_len - burnout, sz, use_feedback_multiplier * vector_size, n_points),
                                      device=device)
        self.i = 0
        self.transformation = transformation
        if use_feedback:
            self.model_feedback = model.feedback_model(seq_len)
        self.model = model
        self.batch_size = batch_size
        self.const_inp = const_inputs
        self.burnout = burnout
        self.n_sequences = n_sequences
        print('vector size = ', vector_size, 'use_feedback_multiplier = ', use_feedback_multiplier)
        self.vector_size = vector_size
        self.seq_len = seq_len
        self.n_points = n_points
        self.use_feedback = use_feedback
        self.squeeze = squeeze

    def set_next(self):
        state_0_loader = self.state_0_load()
        for j in range(0, self.n_inputs * self.n_sequences, self.batch_size):
            # Compute loss
            with torch.no_grad():
                # Get input and initial state
                s = slice(j, min(j + self.batch_size, self.n_inputs * self.n_sequences))
                inp = self.const_inp[:, s, :]
                if self.squeeze:
                    inp = inp.squeeze()
                # compute state
                state_0 = next(state_0_loader)
                output, _ = self.model(inp, state_0)
                if self.transformation is None:
                    traj = output[self.burnout:, :, :]
                else:
                    traj = self.transformation(output[self.burnout:, :, :])
                # Deal with the case where we want
                if self.use_feedback:
                    self.trajectory[:, s, :self.vector_size, self.i] = traj
                    output, _ = self.model_feedback(inp[0, :], state_0)
                    if self.transformation is None:
                        traj = output[self.burnout:, :, :]
                    else:
                        traj = self.transformation(output[self.burnout:, :, :])
                    self.trajectory[:, s, self.vector_size:, self.i] = traj
                else:
                    self.trajectory[:, s, :, self.i] = traj

        self.i += 1

    def get_trajectory(self):
        n_points = min(self.i, self.n_points)
        traj = self.trajectory[:, :, :, :n_points]
        use_feedback_multiplier = 2 if self.use_feedback else 1
        return traj.reshape(self.seq_len - self.burnout, self.n_sequences, -1,
                            use_feedback_multiplier*self.vector_size, n_points)