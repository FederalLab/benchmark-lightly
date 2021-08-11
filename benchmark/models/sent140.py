
import torch.nn as nn

from .base import Model
from .utils import top_one_acc


class Sent140(Model):
    """
    Args:
        vocab_size: the size of the vocabulary, used as a dimension in the input embedding.
        sequence_length: the length of input sequences.
    Returns:
        An un-compiled `torch.nn.Module`.
    """

    def __init__(self, 
            task: str = 'bag_log_reg',
            num_classes  : int = 2,
            embedding_dim: int = 25,
            vocab_size   : int = 400000,
            hidden_size  : int = 100):
        super().__init__()
        assert task in ['bag_log_reg', 'stacked_lstm']
        self.task = task

        self.embeddings = nn.Embedding(
            num_embeddings = vocab_size,
            embedding_dim  = embedding_dim,
            padding_idx    = 0)
        if task == 'stacked_lstm':
            self.lstm = nn.LSTM(
                input_size  = embedding_dim,
                hidden_size = hidden_size,
                num_layers  = 2,
                batch_first = True)
            self.logits = nn.Linear(hidden_size, num_classes)
        else:
            self.logits = nn.Linear(vocab_size, num_classes)


        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy_fn = top_one_acc


    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)
        if self.task == 'stacked_lstm':
            # Note that the order of mini-batch is random so there is no hidden relationship among batches.
            # So we do not input the previous batch's hidden state,
            # leaving the first hidden state zero `self.lstm(embeds, None)`.
            lstm_out, _ = self.lstm(embeds)
            # use the final hidden state as the next character prediction
            final_hidden_state = lstm_out[:, -1]
        else:
            final_hidden_state = embeds
        return self.logits(final_hidden_state)
