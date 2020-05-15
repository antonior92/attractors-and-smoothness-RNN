# %% Example 4: world language model (training)
from word_language_model import (MyRNN, data_generator, Criterion,
                                 data_loader, data_loader_lambada_eval, generate_const_input, CombinedProjection)
import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
# Add parent directory to path (https://stackoverflow.com/questions/714063/importing-modules-from-parent-folder)
import os, sys, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)
from util import get_config
from evaluate_orbit_diagram import EvaluateOrbitDiagram
from rnn_wrap import stabilize_lstm
import argparse
import os
import pandas as pd
from expRNN.parametrization import parametrization_trick

description = "Train recurrent neural network for word language model."

# Arguments that will be saved in config file
config_parser = argparse.ArgumentParser(add_help=False)
config_parser.add_argument('--batch_size', type=int, default=100,
                           help='batch size (default: 100)')
config_parser.add_argument('--epochs', type=int, default=100,
                           help='maximum number of epochs  (default: 100)')
config_parser.add_argument('--lr', type=float, default=0.001,
                           help='learning rate (default: 0.001)')
config_parser.add_argument('--clip_value', type=float, default=0.25,
                           help='maximum value for the gradient norm (default: 0.25)')
config_parser.add_argument('--seed', type=int, default=2,
                           help='random seed for number generator (default: 2)')
config_parser.add_argument("--embed_size", type=int, default=800,
                           help='embedding vector size (default: 800)')
config_parser.add_argument("--tied_weights", action='store_true',
                           help='Tte embedding and softmax weights (default: False)')
config_parser.add_argument("--hidden_size", type=int, default=800,
                           help='RNN number of hidden units (default: 800)')
config_parser.add_argument("--num_layers", type=int, default=1,
                           help='number of stacked RNN layers (default: 1)')
config_parser.add_argument("--seq_len", type=int, default=70,
                           help='sequence length to backpropagate the gradient (default: 70)')
config_parser.add_argument("--patience", type=int, default=7,
                           help='maximum number of epochs without reducing the learning rate (default: 7)')
config_parser.add_argument("--min_lr", type=float, default=1e-7,
                           help='minimum learning rate (default: 1e-7)')
config_parser.add_argument("--lr_factor", type=float, default=0.1,
                           help='reducing factor for the lr in a plateu (default: 0.1)')
config_parser.add_argument("--dropout", type=float, default=0.5,
                           help='dropout rate (default: 0.5)')
config_parser.add_argument("--lambada", action='store_true',
                           help='using lambada dataset (default: false)')
config_parser.add_argument('--stabilize', action='store_true',
                           help='enforce stability of the model as in Miller and Hardt (2018). Default: false.')
config_parser.add_argument('--rnn_type', type=str, default='LSTM',
                           help="Which rnn to use. Options are {'LSTM', 'GRU', 'RNN', 'EURNN',"
                                "'exprnn', 'dtriv', 'cayley'}")
config_parser.add_argument('--interval_orbit', type=int, default=-1,
                           help='how often to sample from orbit  (in epochs). By default doesnt generate orbit diagram.')
config_parser.add_argument('--interval_save_orbit', type=int, default=3,
                           help='How often to save orbit diagram (in epochs). Default: 3')
config_parser.add_argument('--n_sequences_orbit', type=int, default=5,
                           help='Number of sequences used for generating the orbit diagram (default: 10)')
config_parser.add_argument('--seq_len_orbit', type=int, default=1600,
                           help='Sequence lenght used for generating orbit diagram (default: 1200).')
config_parser.add_argument('--burnout_orbit', type=int, default=0,
                           help='Burnout for orbit diagram (default: 0).')
config_parser.add_argument('--std_orbit', type=float, default=100,
                           help='standard deviation used in random state initialization (default: 100)')
config_parser.add_argument('--orbit_use_feedback', action='store_true',
                           help='Add additional feedback path when ploting orbit diagram')

# System dependent arguments
sys_parser = argparse.ArgumentParser(add_help=False)
sys_parser.add_argument('--data', type=str, default='./wikitext-2',
                        help='location of the data corpus. (default: ./wikitext-2)')
sys_parser.add_argument('--corpus', action='store_true',
                        help='force re-make the corpus. (default: False)')
sys_parser.add_argument('--verbose', type=int, default=1,
                        help='verbose (default: False)')
# Get config
args, config, device, folder = get_config(config_parser, sys_parser,
                                          generate_output_folder=True, description=description)
print(config)


# %% Define problem
# Set torch seed
torch.manual_seed(config.seed)
# Load dataset
corpus = data_generator(args.data, args.corpus, config.lambada)
n_words = len(corpus.dictionary)

# Load model
model_kwargs = {"rnn_type": config.rnn_type, "embed_size": config.embed_size,
                "n_words": n_words, "hidden_size": config.hidden_size, "num_layers": config.num_layers,
                "tied_weights": config.tied_weights, "dropout": config.dropout}
model = MyRNN(**model_kwargs)
model.to(device=device)

# Define optimizer
optimizer = optim.Adam(model.parameters(), config.lr)

# Loss function
criterion = Criterion()

# Define learning rate schedules
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.patience,
                                                 min_lr=config.lr_factor*config.min_lr,
                                                 factor=config.lr_factor)


# %% Train model

def train(dataset):
    model.train()
    total_loss = 0
    n_entries = 0
    # Data loader
    dataload = data_loader(dataset, config.batch_size, config.seq_len, device)
    n_total_entries = (len(dataset) // config.batch_size) * config.batch_size
    # Start with zero state
    next_state = model.init_hidden(config.batch_size, device=device,
                                   rnn_type=config.rnn_type)
    for inp, out in dataload():
        model.zero_grad()
        # Use state from previous iteration. Alternatively:
        # model.init_hidden(config.batch_size, device=device)
        if config.rnn_type == 'LSTM':
            h0, c0 = next_state
            state = (h0.detach(), c0.detach())
        else:
            state = next_state.detach()
        # Forward pass
        pred, next_state = model(inp, state)
        loss, n = criterion(pred, out, return_size=True)
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
        n_entries += n
        if args.verbose >= 2:
            print("(train) {0}/{1} ({2:2.2f} %)"
                  .format(n_entries, n_total_entries,
                          100*n_entries/n_total_entries))
    return total_loss/n_entries


def evaluate(dataset):
    model.eval()
    total_loss = 0
    # Data loader
    if config.lambada:
        dataload = data_loader_lambada_eval(dataset, device)
        n_total_entries = sum([len(d) for d in dataset])
    else:
        dataload = data_loader(dataset, config.batch_size, config.seq_len, device)
        n_total_entries = (len(dataset) // config.batch_size) * config.batch_size
    # Start with zero state
    next_state = model.init_hidden(config.batch_size, device=device, rnn_type=config.rnn_type)
    n_entries = 0
    for inp, out in dataload():
        with torch.no_grad():
            # Use state from previous iteration. Alternatively:
            if config.lambada:
                state = model.init_hidden(1, device=device, rnn_type=config.rnn_type)
            else:
                state = next_state
            # Forward pass
            pred, next_state = model(inp, state)
            loss, n = criterion(pred, out, return_size=True)
            # Update
            total_loss += loss
            n_entries += n
        if args.verbose >= 2:
            print("(eval) {0}/{1} ({2:2.2f} %)"
                  .format(n_entries, n_total_entries,
                          100*n_entries/n_total_entries))
    return total_loss/n_entries


print("Start optimization...")
best_loss = np.Inf
history = pd.DataFrame(columns=["Epoch", "Train Loss", "Valid Loss", "Valid PPL", "Learning Rate"])  # Dataframe history
# if necessary configure save
if config.interval_orbit >= 1:
    n_points = config.epochs
    const_input = generate_const_input(model, device)
    vector_size = 1
    # Define average as transformation
    transformation = CombinedProjection(n_words, device)
    orbit = EvaluateOrbitDiagram(n_points, const_input, model, device,
                                 config.batch_size, config.n_sequences_orbit,
                                 config.seq_len_orbit, config.burnout_orbit,
                                 len(transformation), config.std_orbit, transformation, config.rnn_type,
                                 config.orbit_use_feedback)
for ep in range(config.epochs):
    train_loss = train(corpus.train)
    valid_loss = evaluate(corpus.valid)
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
    if valid_loss < best_loss:
        test_loss = evaluate(corpus.test)
        torch.save({'epoch': ep,
                    'model': model.state_dict(),
                    'model_kwargs': model_kwargs,
                    'optimizer': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': valid_loss,
                    'test_loss': test_loss,
                    'config': config,
                    'experiment_name': 'word-language-model'},
                   os.path.join(folder, 'model.pth'))
        best_loss = valid_loss
    # Get learning rate
    for param_group in optimizer.param_groups:
        learning_rate = param_group["lr"]
    # Interrupt for minimum learning rate
    if learning_rate < config.min_lr:
        break
    # Print
    valid_ppl = torch.exp(valid_loss)
    print('Train Epoch: {:2d} \tTrain Loss: {:.6f} '
          '\tValid Loss: {:.6f} \tValid PPL {:2.6f} '
          '\tLearning Rate: {:.7f}'
          .format(ep,  train_loss, valid_loss,
                  valid_ppl, learning_rate))
    print(120*'-')
    # Append information to history
    history = history.append({"Epoch": ep,
                              "Train Loss": train_loss.detach().cpu().numpy(),
                              "Valid Loss": valid_loss.detach().cpu().numpy(),
                              "Valid PPL": valid_ppl.detach().cpu().numpy(),
                              "Learning Rate": learning_rate},
                             ignore_index=True)
    history.to_csv(os.path.join(folder, 'history.csv'))
    # Update learning rate
    scheduler.step(valid_loss)

print(120*'-')
test_ppl = torch.exp(test_loss)
print('Test Loss: {:.6f} \tTest PPL {:2.6f}'
      .format(test_loss, test_ppl))
print(120*'-')
