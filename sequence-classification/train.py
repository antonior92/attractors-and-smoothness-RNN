import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import argparse
from sequence_classification import MyRNN, data_generator, data_loader, Criterion, generate_const_input
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from util import get_config
from evaluate_orbit_diagram import EvaluateOrbitDiagram
from rnn_wrap import stabilize_lstm
import pandas as pd
from expRNN.parametrization import parametrization_trick

description = "Train recurrent neural network to classify few relevant, widely separeted, symbols."

# Arguments that will be saved in config file
config_parser = argparse.ArgumentParser(add_help=False)
config_parser.add_argument('--epochs', type=int, default=10000,
                           help='maximum number of epochs (default: 100)')
config_parser.add_argument('--seq_len', type=int, default=200,
                           help='sequence length used during training (default: 200)')
config_parser.add_argument('--seq_len_val', type=int, default=200,
                           help='Ssquence length used during validation (default: 200)')
config_parser.add_argument('--batch_size', type=int, default=100,
                           help='batch size (default: 100)')
config_parser.add_argument('--n_sequences', type=int, default=1000,
                           help='number of sequences used for training (default: 1000)')
config_parser.add_argument('--n_sequences_val', type=int, default=1000,
                           help='number of sequences used for validation(default: 1000)')
config_parser.add_argument('--n_symbols', type=int, default=2,
                           help='number of symbols (default: 2)')
config_parser.add_argument('--n_distractors', type=int, default=4,
                           help='number of distractors (default: 4)')
config_parser.add_argument('--lr', type=float, default=0.001,
                           help='learning rate (default: 0.001)')
config_parser.add_argument('--clip_value', type=float, default=0.25,
                           help='maximum value for the gradient norm (default: 0.25)')
config_parser.add_argument('--seed', type=int, default=2,
                           help='random seed for number generator (default: 2)')
config_parser.add_argument('--gamma', type=float, default=0.25,
                           help='decrease factor for lr scheduler (default: 0.25)')
config_parser.add_argument('--milestones', nargs='+', type=int,
                           default=[500, 1000, 2000],
                           help='milestones for lr scheduler (default: [500, 1000, 2000, 8000, 15000, 25000])')
config_parser.add_argument('--position_symbols_start', nargs='+', type=int,
                           default=[9, 39],
                           help='list containing the start position for relevant symbols (default: [9, 39])')
config_parser.add_argument('--position_symbols_len', nargs='+', type=int,
                           default=10,
                           help='range in which relevant symbols may appear -- start from position_symbols_start.'
                                '(default: 10)')
config_parser.add_argument('--interval_orbit', type=int, default=-1,
                           help='how often to sample from orbit  (in epochs). By default doesnt generate orbit diagram.')
config_parser.add_argument('--interval_save_orbit', type=int, default=100,
                           help='How often to save orbit diagram (in epochs). Default: 1000')
config_parser.add_argument('--n_sequences_orbit', type=int, default=10,
                           help='Number of sequences used for generating the orbit diagram (default: 10)')
config_parser.add_argument('--seq_len_orbit', type=int, default=800,
                           help='Sequence lenght used for generating orbit diagram (default: 800).')
config_parser.add_argument('--burnout_orbit', type=int, default=400,
                           help='Burnout for orbit diagram (default: 400).')
config_parser.add_argument('--std_orbit', type=float, default=100,
                           help='standard deviation used in random state initialization (default: 100)')
config_parser.add_argument('--stabilize', action='store_true',
                           help='enforce stability of the model as in Miller and Hardt (2018). Default: false.')
config_parser.add_argument('--rnn_type', type=str, default='LSTM',
                           help="Which rnn to use. Options are {'LSTM', 'GRU', 'RNN', 'EURNN',"
                                "'exprnn', 'dtriv', 'cayley'}")

# Get config
args, config, device, folder = get_config(config_parser, generate_output_folder=True, description=description)
print(config)

# %%  Define problem
# Set seed
torch.manual_seed(config.seed)
# Generate data
interval_positions = [(s, s+config.position_symbols_len)
                      for s in config.position_symbols_start]
data = data_generator(config.n_symbols, config.n_distractors,
                      interval_positions, config.seq_len, config.n_sequences,
                      config.seed, device)
data_val = data_generator(config.n_symbols, config.n_distractors,
                          interval_positions, config.seq_len_val,
                          config.n_sequences_val, config.seed+1, device)

# Create model instance
model_kwargs = {'rnn_type': config.rnn_type, "n_classes": data["n_classes"], "input_size": data["input_size"]}
model = MyRNN(**model_kwargs)
model.to(device=device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), config.lr)

# Define learning rate schedules
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=config.milestones, gamma=config.gamma)

# Loss function
criterion = Criterion()

# State 0
state_0 = model.init_hidden(config.batch_size, device, config.rnn_type)


# %% Train model

def train(dataset):
    model.train()
    total_loss = 0
    correct_values = 0
    dataload = data_loader(dataset, config.batch_size, config.n_sequences)
    for inp, target in dataload():
        # Clear stored gradient
        model.zero_grad()
        # Forward pass
        pred, _ = model(inp, state_0)
        loss, num_correct_values = criterion(pred, target, return_num_correct_values=True)
        correct_values += num_correct_values
        # Backward pass
        if config.rnn_type in ['exprnn', 'dtriv', 'cayley']:
            loss = parametrization_trick(model, loss)
        loss.backward()
        # Clip gradient
        clip_grad_norm_(model.parameters(), config.clip_value)
        # Optimize
        optimizer.step()
        # Stabilize
        with torch.no_grad():
            if config.stabilize and config.rnn_type == 'LSTM':
                model.rnn.rnn = stabilize_lstm(model.rnn.rnn)
        # Update
        total_loss += loss.detach()
    total_loss = total_loss / config.n_sequences
    # Convert tensor to float, otherwise it just round acc back to zero
    total_acc = torch.tensor(correct_values, dtype=torch.float32, device=device) / config.n_sequences
    return total_loss, total_acc


def evaluate(dataset):
    model.eval()
    total_loss = 0
    correct_values = 0
    dataload = data_loader(dataset, config.batch_size, config.n_sequences)
    for inp, target in dataload():
        with torch.no_grad():
            # Forward pass
            pred, _ = model(inp, state_0)
            loss, num_correct_values = criterion(pred, target, return_num_correct_values=True)
            correct_values += num_correct_values
            # Update
            total_loss += loss
    total_loss = total_loss / config.n_sequences
    # Convert tensor to float, otherwise it just round acc back to zero
    total_acc = torch.tensor(correct_values, dtype=torch.float32, device=device) / config.n_sequences
    return total_loss, total_acc


print("Start optimization...")
history = pd.DataFrame(columns=["epoch", "train loss", "train acc.", "valid. loss",
                                "valid. acc."])  # Dataframe history
# if necessary configure save
if config.interval_orbit >= 1:
    n_points = config.epochs
    const_input = generate_const_input(model, device)
    vector_size = config.n_symbols * len(config.position_symbols_start)
    transformation = None
    orbit = EvaluateOrbitDiagram(n_points, const_input, model, device,
                                 config.batch_size, config.n_sequences_orbit,
                                 config.seq_len_orbit, config.burnout_orbit,
                                 vector_size, config.std_orbit, transformation, config.rnn_type)
for ep in range(config.epochs):
    train_loss, train_acc = train(data)
    val_loss, val_acc = evaluate(data_val)
    # Save orbit diagram:
    model.eval()
    if config.interval_orbit >= 1:
        if ep % config.interval_orbit == config.interval_orbit - 1:
            orbit.set_next()
        if ep % config.interval_orbit == config.interval_orbit - 1:
            torch.save({"frequency": config.interval_orbit,
                        "trajectory": orbit.get_trajectory()},
                       os.path.join(folder, 'orbit_per_epoch_diagram.pth'))

    # Save *last* model
    torch.save({'epoch': ep,
                'model': model.state_dict(),
                'model_kwargs': model_kwargs,
                'optimizer': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'config': config,
                'experiment_name': 'temporal-order'},
               os.path.join(folder, 'model.pth'))
    # Append information to history
    history = history.append({"epoch": ep,
                              "train loss": train_loss.detach().cpu().numpy(),
                              "train acc.": train_acc.detach().cpu().numpy(),
                              "valid. loss": val_loss.detach().cpu().numpy(),
                              "valid. acc.": val_acc.detach().cpu().numpy()},
                             ignore_index=True)
    history.to_csv(os.path.join(folder, 'history.csv'))
    # Print
    print('Train Epoch: {:2d} \tLoss: {:.6f} \tAcc.: {:.6f}, \tVal Loss: {:.6f} \tVal Acc.: {:.6f}'
          .format(ep, train_loss, train_acc, val_loss, val_acc))
    print(100*'-')
    scheduler.step()