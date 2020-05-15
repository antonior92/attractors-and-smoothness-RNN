# %% Example 2: Sine generator (training)
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import torch.nn.functional as F
import argparse
import pandas as pd
# Add parent directory to path (https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
# Import things from parent directory
from sine_generator import (MyRNN, data_generator, data_loader,
                            initial_state_generator,
                            generate_const_input)
from util import get_config
from evaluate_orbit_diagram import state_0_loader_per_batch, EvaluateOrbitDiagram
from rnn_wrap import stabilize_lstm
from expRNN.parametrization import parametrization_trick

description = "Train recurrent neural network to generate sine waves from const input with given frequency."

# Arguments that will be saved in config file
config_parser = argparse.ArgumentParser(add_help=False)
config_parser.add_argument('--epochs', type=int, default=10000,
                           help='maximum number of epochs (default: 10000)')
config_parser.add_argument('--seq_len', type=int, default=400,
                           help='sequence length used during training (default: 400)')
config_parser.add_argument('--batch_size', type=int, default=10,
                           help='batch size (default: 10)')
config_parser.add_argument('--n_sequences', type=int, default=100,
                           help='Number of sequences used for training (default: 100)')
config_parser.add_argument('--lr', type=float, default=0.001,
                           help='learning rate (default: 0.001)')
config_parser.add_argument('--clip_value', type=float, default=0.25,
                           help='maximum value for the gradient norm (default: 0.25)')
config_parser.add_argument('--min_freq', type=float, default=np.pi/16,
                           help='minimum frequency to be generated (default: pi/16)')
config_parser.add_argument('--max_freq', type=float, default=np.pi/8,
                           help='maximum frequency to be generated (default: pi/8)')
config_parser.add_argument('--seed', type=int, default=2,
                           help='random seed for number generator (default: 2)')
config_parser.add_argument('--gamma', type=float, default=0.1,
                           help='decrease factor for lr scheduler (default: 0.25)')
config_parser.add_argument('--milestones', nargs='+', type=int,
                           default=[500, 1000, 2000, 8000, 15000, 25000],
                           help='milestones for lr scheduler (default: [500, 1000, 2000, 8000, 15000, 25000])')
config_parser.add_argument('--interval_orbit', type=int, default=-1,
                           help='how often to sample from orbit  (in epochs). By default doesnt generate orbit diagram.')
config_parser.add_argument('--interval_save_orbit', type=int, default=500,
                           help='How often to save orbit diagram (in epochs). Default: 500')
config_parser.add_argument('--n_sequences_orbit', type=int, default=4,
                           help='Number of sequences used for generating the orbit diagram (default: 4)')
config_parser.add_argument('--seq_len_orbit', type=int, default=1600,
                           help='Sequence lenght used for generating orbit diagram (default: 1600).')
config_parser.add_argument('--burnout_orbit', type=int, default=1200,
                           help='Burnout for orbit diagram (default: 1200).')
config_parser.add_argument('--std_orbit', type=float, default=100,
                           help='standard deviation used in random state initialization (default: 100)')
config_parser.add_argument('--rnn_type', type=str, default='LSTM',
                           help="Which rnn to use. Options are {'LSTM', 'GRU', 'RNN', 'EURNN',"
                                "'exprnn', 'dtriv', 'cayley'}")
config_parser.add_argument('--hidden_size', type=int, default=200,
                           help="Hidden size rnn. Default is 200.")
config_parser.add_argument('--num_layers', type=int, default=1,
                           help="Number of layers. Default is 1.")
config_parser.add_argument('--stabilize', action='store_true',
                           help='enforce stability of the model as in Miller and Hardt (2018). Default: false.')

args, config, device, folder = get_config(config_parser, generate_output_folder=True, description=description)

print(config)

# %%  Define problem
# Set seed
torch.manual_seed(config.seed)
# Generate data
data = data_generator(config.min_freq, config.max_freq, config.n_sequences, config.seq_len, device)
# Create model instance
model_kwargs = {'rnn_type': config.rnn_type, 'hidden_size': config.hidden_size, 'num_layers': config.num_layers}
model = MyRNN(**model_kwargs)
model.to(device=device)
# Generate initial state
state_0 = initial_state_generator(data, model, device, config.rnn_type)
state_0_load = state_0_loader_per_batch(state_0, config.batch_size, config.n_sequences, config.rnn_type)
# Define optimizer
if config.rnn_type == 'LSTM':
    params_list = [*list(model.parameters()), *state_0]
else:
    params_list = [*list(model.parameters()), state_0]
optimizer = optim.Adam(params_list, config.lr)
# Define learning rate schedules
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)
# Data loader
dataload = data_loader(data, config.batch_size, config.n_sequences)


# %% Train model
print("Start optimization...")
best_loss = np.Inf
history = pd.DataFrame(columns=["epoch", "loss"])
# if necessary configure save
if config.interval_orbit >= 1:
    n_points = config.epochs
    const_input = generate_const_input(model, device)
    vector_size = 1
    transformation = None
    orbit = EvaluateOrbitDiagram(n_points, const_input, model, device,
                                 config.batch_size, config.n_sequences_orbit,
                                 config.seq_len_orbit, config.burnout_orbit,
                                 vector_size, config.std_orbit, transformation, config.rnn_type,
                                 False, squeeze=False)   # Squeeze = true to get the right dimensions
for ep in range(config.epochs):
    model.train()
    total_loss = 0
    state_0_loader = state_0_load()
    for u_batch, y_batch in dataload():
        # Clear stored gradient
        model.zero_grad()
        # Forward pass
        y_pred, _ = model(u_batch, next(state_0_loader))
        loss = F.mse_loss(y_pred, y_batch, reduction="sum")
        # Backward pass
        if config.rnn_type in ['exprnn', 'dtriv', 'cayley']:
            loss = parametrization_trick(model, loss)
        loss.backward()
        # Clip gradient
        clip_grad_norm_(params_list, config.clip_value)
        # Optimize
        optimizer.step()
        # Stabilize
        with torch.no_grad():
            if config.stabilize and config.rnn_type == 'LSTM':
                model.rnn.rnn = stabilize_lstm(model.rnn.rnn)
        # Compute loss
        total_loss += loss.detach()
    total_loss = total_loss/(config.seq_len * config.n_sequences)

    # Save orbit diagram:
    model.eval()
    if config.interval_orbit >= 1:
        if ep % config.interval_orbit == config.interval_orbit - 1:
            orbit.set_next()
        if ep % config.interval_orbit == config.interval_orbit - 1:
            torch.save({"frequency": config.interval_orbit,
                        "trajectory": orbit.get_trajectory()},
                       os.path.join(folder, 'orbit_per_epoch_diagram.pth'))

    # Save best model
    if total_loss < best_loss:
        torch.save({'epoch': ep,
                    'model': model.state_dict(),
                    'model_kwargs': model_kwargs,
                    'optimizer': optimizer.state_dict(),
                    'total_loss': total_loss,
                    'config': config,
                    'state': state_0,
                    'experiment_name': 'sine-generator'},
                   os.path.join(folder, 'model.pth'))
        best_loss = total_loss
    history = history.append({"epoch": ep,
                              "loss": total_loss.detach().cpu().numpy()},
                             ignore_index=True)
    history.to_csv(os.path.join(folder, 'history.csv'))
    # Print
    print('Train Epoch: {:2d} \tLoss: {:.6f}'.format(ep,  total_loss))
    print(60*'-')
    scheduler.step()
