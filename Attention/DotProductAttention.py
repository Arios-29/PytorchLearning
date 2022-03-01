import math

import torch
from torch import nn

from Attention.masked_softmax import masked_softmax


class DotProductAttention(nn.Module):
    def __init__(self, drop_prob=0):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, keys, values, queries, valid_lens=None):
        """
        :param keys: (batch_size, 键值对个数, d)
        :param values: (batch_size, 键值对个数, value_size)
        :param queries: (batch_size, 查询的个数, d)
        :param valid_lens:
        :return: (batch_size, 查询个数, value_size)
        """
        d = queries.shape[-1]
        # keys.transpose(1, 2): (batch_size, d, 键值对个数)
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(d)  # (batch_size, 查询的个数, 键值对个数)
        alpha = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(alpha), values)  # (batch_size, 查询的个数, value_size)

