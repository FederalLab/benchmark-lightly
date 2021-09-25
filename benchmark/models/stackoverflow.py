# @Author            : FederalLab
# @Date              : 2021-09-26 00:28:37
# @Last Modified by  : Chen Dengsheng
# @Last Modified time: 2021-09-26 00:28:37
# Copyright (c) FederalLab. All rights reserved.

import torch
import torch.nn as nn

from .base import Model
from .utils import top_one_acc


class Stackoverflow(Model):
    """Creates a RNN model using LSTM layers for StackOverFlow (next word
    prediction task).

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
                 vocab_size=10000,
                 num_oov_buckets=1,
                 embedding_size=96,
                 latent_size=670,
                 num_layers=1):
        super().__init__()
        # For pad/bos/eos/oov.
        extended_vocab_size = vocab_size + 3 + num_oov_buckets

        self.word_embeddings = nn.Embedding(num_embeddings=extended_vocab_size,
                                            embedding_dim=embedding_size,
                                            padding_idx=0)

        self.lstm = nn.LSTM(input_size=embedding_size,
                            hidden_size=latent_size,
                            num_layers=num_layers)

        self.fc1 = nn.Linear(latent_size, embedding_size)
        self.fc2 = nn.Linear(embedding_size, extended_vocab_size)
        self.loss_fn = nn.CrossEntropyLoss()

        self.accuracy_fn = top_one_acc

    def forward(self, input_seq, hidden_state=None):
        embeds = self.word_embeddings(input_seq)
        lstm_out, hidden_state = self.lstm(embeds, hidden_state)
        fc1_output = self.fc1(lstm_out[:, :])
        output = self.fc2(fc1_output)
        output = torch.transpose(output, 1, 2)
        return output
