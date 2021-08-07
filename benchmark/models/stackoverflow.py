# MIT License

# Copyright (c) 2021 FederalLab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import torch.nn as nn

from .base import Model
from .utils import top_one_acc


class StackOverFlow(Model):
    """Creates a RNN model using LSTM layers for StackOverFlow (next word prediction task).
        This replicates the model structure in the paper:
    "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)
    Table 9
    Args:
        vocab_size: the size of the vocabulary, used as a dimension in the input embedding.
        sequence_length: the length of input sequences.
    Returns:
        An un-compiled `torch.nn.Module`.
    """

    def __init__(self, 
                 vocab_size      = 10000,
                 num_oov_buckets = 1,
                 embedding_size  = 96,
                 latent_size     = 670,
                 num_layers      = 1):
        super().__init__()
        # For pad/bos/eos/oov.
        extended_vocab_size = vocab_size + 3 + num_oov_buckets
        self.word_embeddings = nn.Embedding(
            num_embeddings = extended_vocab_size,
            embedding_dim  = embedding_size,
            padding_idx    = 0)
        self.lstm = nn.LSTM(
            input_size  = embedding_size,
            hidden_size = latent_size,
            num_layers  = num_layers)

        self.fc1 = nn.Linear(latent_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, extended_vocab_size)

        self.loss_fn = nn.CrossEntropyLoss()
        self.acc_fn  = top_one_acc

    def forward(self, input_seq, hidden_state=None):
        embeds = self.word_embeddings(input_seq)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        fc1_output = self.fc1(lstm_out[:, :])
        output = self.fc2(fc1_output)
        output = torch.transpose(output, 1, 2)
        return output
