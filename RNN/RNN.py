import torch
from torch import nn


class RNN(nn.Module):
    def __init__(self, xt_size, ht_size, num_layers):
        super(RNN, self).__init__()
        self.net = nn.RNN(xt_size, ht_size, num_layers)

    def forward(self, a_batch_of_seqs):
        output, final_state = self.net(a_batch_of_seqs)
        return output, final_state
