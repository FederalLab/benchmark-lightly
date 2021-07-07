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


from .utils import top_one_acc
import torch.nn as nn


class ShakespeareNCP(nn.Module):
    """Creates a RNN model using LSTM layers for Shakespeare language models (next character prediction task).
    This replicates the model structure in the paper:
        Communication-Efficient Learning of Deep Networks from Decentralized Data
        H. Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, Blaise Agueray Arcas. AISTATS 2017.
        https://arxiv.org/abs/1602.05629
    This is also recommended model by "Adaptive Federated Optimization. ICML 2020" (https://arxiv.org/pdf/2003.00295.pdf)
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
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_seq):
        embeds = self.embeddings(input_seq)
        # Note that the order of mini-batch is random so there is no hidden relationship among batches.
        # So we do not input the previous batch's hidden state,
        # leaving the first hidden state zero `self.lstm(embeds, None)`.
        lstm_out, _ = self.lstm(embeds)
        # use the final hidden state as the next character prediction
        final_hidden_state = lstm_out[:, -1]
        output = self.fc(final_hidden_state)
        return output


loss_fn = nn.CrossEntropyLoss()
acc_fn = top_one_acc
