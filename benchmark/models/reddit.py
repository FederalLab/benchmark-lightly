
import torch.nn as nn

from .base import Model
from .utils import top_one_acc


class Reddit(Model):
    """
    Args:
        vocab_size: the size of the vocabulary, used as a dimension in the input embedding.
        sequence_length: the length of input sequences.
    Returns:
        An uncompiled `torch.nn.Module`.
    """

    def __init__(self, embedding_dim=8, vocab_size=90, hidden_size=256):
        super().__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.logits = nn.Linear(hidden_size, vocab_size)


        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_fn = top_one_acc


    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        final_hidden_state = lstm_out[:, -1]
        return self.logits(final_hidden_state)