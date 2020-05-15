# %%
import os
import re
import torch
import pickle
import torch.nn as nn
import sys
sys.path.append("../")
from rnn_wrap import RNNwrap


class FeedbackModel(nn.Module):
    def __init__(self, model, seq_len):
        super(FeedbackModel, self).__init__()
        self.model = model
        self.seq_len = seq_len
        print('feedback model!')

    def forward(self, inp, state_0):
        """Forward propagate throught LSTM in parallel mode.

        Parameters
        ----------
        inp: torch.Tensor (long int)
            Input vector of shape (batch_size,), if receive extra dimension they will just be squeezed out.
        state: tuple of torch.Tensor
            Tuple (h_0, c_0), both torch.Tensor with shape: (num_layers, batch_size, hidden_size).

        Returns
        -------
        out: torch.Tensor
            Output vector of shape (seq_len, batch_size)
        next_state: tuple of torch.Tensor
            Tuple (h_0, c_0), both torch.Tensor with shape: (num_layers, batch_size, hidden_size).
        """
        next_inp = inp.view(1, -1)
        next_state = state_0
        outputs = []
        for i in range(self.seq_len):
            out, next_state = self.model(next_inp, next_state)
            outputs.append(out)
            next_inp = torch.argmax(out, dim=-1)

        return torch.cat(outputs, dim=0), next_state


# Define RNN model
class MyRNN(nn.Module):
    """RNN model for language modeling"""
    def __init__(self, embed_size, n_words, rnn_type='LSTM', hidden_size=200, num_layers=1,
                 tied_weights=False, dropout=0, K=100, init='henaff', bias=True):
        super(MyRNN, self).__init__()
        self.embed_size = embed_size
        self.n_words = n_words
        self.encoder = nn.Embedding(n_words, embed_size)
        self.drop = nn.Dropout(dropout)
        self.rnn = RNNwrap(embed_size, hidden_size, rnn_type, num_layers, K, init, bias=bias)
        self.decoder = nn.Linear(hidden_size, n_words)
        if tied_weights:
            if hidden_size != embed_size:
                raise ValueError('When using the tied flag, hidden_size must be equal to embed_size')
            self.decoder.weight = self.encoder.weight
            print("Weight tied")
        self.num_layers = num_layers
        self.hidden_size = hidden_size

    def init_hidden(self, batch_size, device, rnn_type):
        return self.rnn.initial_state(batch_size, device, rnn_type)

    def forward(self, inp, state):
        """Forward propagate throught LSTM.

        Parameters
        ----------
        inp: torch.Tensor (long int)
            Input vector of shape (seq_len, batch_size), if receive extra dimension they will just be squeezed out.
        state: tuple of torch.Tensor
            Tuple (h_0, c_0), both torch.Tensor with shape: (num_layers, batch_size, hidden_size).

        Returns
        -------
        out: torch.Tensor
            Output vector of shape (seq_len, batch_size, n_words)
        next_state: tuple of torch.Tensor
            Tuple (h_0, c_0), both torch.Tensor with shape: (num_layers, batch_size, hidden_size).
        """
        emb = self.drop(self.encoder(inp))
        y, state_next = self.rnn(emb, state)
        y = self.drop(y)
        y = self.decoder(y)
        return y, state_next

    def feedback_model(self, seq_len):
        return FeedbackModel(self, seq_len)

bs = 20
embed_size = 10
n_words = 30
seq_len = 100
n_layers = 1
hidden_size=200


inp = torch.randint(embed_size, (bs,), dtype=torch.long)
state_0 = (torch.ones(n_layers, bs, hidden_size), torch.ones(n_layers, bs, hidden_size))

rnn = MyRNN(embed_size, n_words)

fb_rnn = rnn.feedback_model(seq_len)

out, state_next = fb_rnn(inp, state_0)

# %%
# Generate dataset
def data_generator(path, new_corpus, lambada=False):
    corpus_path = os.path.join(path, 'corpus')
    if os.path.exists(corpus_path) and not new_corpus:
        corpus = pickle.load(open(corpus_path, 'rb'))
    else:
        if lambada:
            corpus = CorpusLambada(path)
        else:
            corpus = Corpus(path)
        pickle.dump(corpus, open(corpus_path, 'wb'))
    return corpus


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = []

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word.append(word)
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)


class Corpus(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.train = self.tokenize(os.path.join(path, 'train.txt'))
        self.valid = self.tokenize(os.path.join(path, 'valid.txt'))
        self.test = self.tokenize(os.path.join(path, 'test.txt'))

    def tokenize(self, path):
        """Tokenizes a text file."""
        assert os.path.exists(path)
        # Add words to the dictionary
        with open(path, 'r') as f:
            tokens = 0
            for line in f:
                words = line.split() + ['<eos>']
                tokens += len(words)
                for word in words:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = []
            token = 0
            for line in f:
                words = line.split() + ['<eos>']
                for word in words:
                    ids.append(self.dictionary.word2idx[word])
                    token += 1

        return ids


class CorpusLambada(object):
    def __init__(self, path):
        self.dictionary = Dictionary()
        self.prep_dict(os.path.join(path, 'lambada-vocab-2.txt'))
        self.train = self.tokenize(os.path.join(path, 'train-novels'))
        self.valid = self.tokenize(os.path.join(path, 'lambada_development_plain_text.txt'), eval=True)
        self.test = self.tokenize(os.path.join(path, 'lambada_test_plain_text.txt'), eval=True)

    def prep_dict(self, dict_path):
        assert os.path.exists(dict_path)

        # Add words to the dictionary
        with open(dict_path, 'r', encoding='utf-8') as f:
            tokens = 0
            for line in f:
                word, _ = line.strip().split('\t')
                tokens += 1
                self.dictionary.add_word(word)

        if "<eos>" not in self.dictionary.word2idx:
            self.dictionary.add_word("<eos>")
            tokens += 1

        print("The dictionary captured a vocabulary of size {0}.".format(tokens))

    def tokenize(self, path, eval=False):
        assert os.path.exists(path)

        ids = []
        token = 0
        misses = 0
        if not path.endswith(".txt"):   # it's a folder
            for subdir in os.listdir(path):
                for filename in os.listdir(path + "/" + subdir):
                    if filename.endswith(".txt"):
                        full_path = "{0}/{1}/{2}".format(path, subdir, filename)
                        # Tokenize file content
                        delta_ids, delta_token, delta_miss = self._tokenize_file(full_path, eval=eval)
                        ids += delta_ids
                        token += delta_token
                        misses += delta_miss
        else:
            ids, token, misses = self._tokenize_file(path, eval=eval)

        print(token, misses)
        return ids

    def _tokenize_file(self, path, eval=False):
        with open(path, 'r', encoding='utf-8') as f:
            token = 0
            ids = []
            misses = 0
            for line in f:
                line_ids = []
                words = line.strip().split() + ['<eos>']
                if eval:
                    words = words[:-1]
                for word in words:
                    # These words are in the text but not vocabulary
                    if word == "n't":
                        word = "not"
                    elif word == "'s":
                        word = "is"
                    elif word == "'re":
                        word = "are"
                    elif word == "'ve":
                        word = "have"
                    elif word == "wo":
                        word = "will"
                    if word not in self.dictionary.word2idx:
                        word = re.sub(r'[^\w\s]', '', word)
                    if word not in self.dictionary.word2idx:
                        misses += 1
                        continue
                    line_ids.append(self.dictionary.word2idx[word])
                    token += 1
                if eval:
                    ids.append(line_ids)
                else:
                    ids += line_ids
        return ids, token, misses


# Generator for any text dataset except for ``valid" and ``test" in lambada
def data_loader(dataset, bs, seq_len, device):
    def _data_loader():
        # Convert to pytorch tensor
        data = torch.tensor(dataset, dtype=torch.long, device=device)
        # Work out how cleanly we can divide the dataset into batch_size parts (i.e. continuous seqs).
        n_batch = data.size(0) // bs
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, n_batch * bs)
        # Evenly divide the data across the batch_size batches.
        data_bs = data.view(bs, -1).transpose(1, 0)
        total_len = data_bs.size(0)
        for i in range(0, total_len, seq_len):
            # get batch
            inp = data_bs[i:min(i+seq_len, total_len-1), :]
            out = data_bs[i+1:min(i+1+seq_len, total_len), :]
            yield inp, out
    return _data_loader


# Generator for ``valid" and ``test" in lambada
def data_loader_lambada_eval(data, device):
    def _data_loader():
        for sequence in data:
            # get batch
            inp = torch.tensor(sequence[:-1], dtype=torch.long, device=device).view(-1, 1)
            out = torch.tensor([sequence[-1]], dtype=torch.long, device=device)
            yield inp, out
    return _data_loader


def generate_const_input(model, device):
    """Return constant inputs shape = (n_different_inputs, input_size)"""
    n_different_inputs = 2
    const_input = torch.randint(model.embed_size, (n_different_inputs, 1), dtype=torch.long, device=device)   # Random index
    return const_input


class Criterion(object):

    def __init__(self):
        self.cross_entropy = nn.CrossEntropyLoss(reduction="sum")

    def __call__(self, pred, out, return_size=False):
        target = out.contiguous().view(-1)
        prediction = pred.view(-1, pred.size(-1))
        if target.size(0) < prediction.size(0):
            prediction = prediction[-target.size(0):, :]
        if return_size:
            return self.cross_entropy(prediction, target), target.size(0)
        else:
            return self.cross_entropy(prediction, target)


# Some projections to use for generating orbit
class MeanProjection(object):
    def __init__(self, n_words, device):
        transformation = nn.Linear(n_words, 1, bias=False)
        transformation.weight = nn.Parameter(
            data=1 / n_words * torch.ones((1, n_words), device=device, dtype=torch.float32), requires_grad=False)
        transformation.to(device)

        self.transformation = transformation

    def __call__(self, traj):
        return self.transformation(traj)

    def __len__(self):
        return 1


class RandomLinearProjection(object):
    def __init__(self, n_words, device, nproj=2):
        transformation = nn.Linear(n_words, nproj, bias=False)
        # Here .uniform_(0, 2) overide values with uniform distribution
        transformation.weight = nn.Parameter(
            data=1 / n_words * torch.ones((nproj, n_words), device=device, dtype=torch.float32).uniform_(0, 2),
            requires_grad=False)
        self.transformation = transformation
        self.nproj = nproj

    def __len__(self):
        return self.nproj

    def __call__(self, traj):
        return self.transformation(traj)


class SingleStateProjection(object):
    def __init__(self, n_words, device):
        nproj = 12
        transformation = nn.Linear(n_words, nproj, bias=False)
        data = torch.zeros((nproj, n_words), device=device, dtype=torch.float32)
        data[0, 0] = 1  # projection 0 - <eos>
        data[1, 16] = 1  # projection 1 - 'of'
        data[2, 17] = 1  # projection 2 - 'the'
        data[3, 13] = 1  # projection 3 - ','
        data[4, 15] = 1  # projection 4 - '.'
        data[5, 22] = 1  # projection 5 - 'to'
        data[6, 26] = 1  # projection 6 - 'is'
        data[7, 35] = 1  # projection 7 - 'by'
        data[8, 75] = 1  # projection 8 - 'secret'
        data[9, 25] = 1  # projection 9 - 'Japan'
        data[10, 2] = 1  # projection 10 - 'Valkyria'
        data[11, 9] = 1  # projection 11 - <unk>
        transformation.weight = nn.Parameter(
            data=data,
            requires_grad=False)
        self.transformation = transformation
        self.nproj = nproj

    def __len__(self):
        return self.nproj

    def __call__(self, traj):
        return self.transformation(traj)


class CombinedProjection(object):
    # Combine all projections into a single one
    def __init__(self, n_words, device):
        self.mean = MeanProjection(n_words, device)
        self.randproj = RandomLinearProjection(n_words, device)
        self.ssproj = SingleStateProjection(n_words, device)

    def __len__(self):
        return len(self.mean) + len(self.randproj) + len(self.ssproj)

    def __call__(self, traj):
        t1 = self.mean(traj)
        t2 = self.randproj(traj)
        t3 = self.ssproj(traj)

        return torch.cat((t1, t2, t3), dim=-1)