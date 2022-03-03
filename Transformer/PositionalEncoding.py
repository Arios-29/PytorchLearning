import torch
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, num_embed, drop_prob=0, max_len=1000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.P = torch.zeros((1, max_len, num_embed))
        numerator = torch.arange(max_len, dtype=torch.float32).view(-1, 1)  # (max_len, 1)
        denominator = torch.pow(10000, torch.arange(0, num_embed, 2, dtype=torch.float32) / num_embed) # (num_embed)
        alpha = numerator / denominator  # (max_len, num_embed)
        self.P[:, :, 0::2] = torch.sin(alpha)
        self.P[:, :, 1::2] = torch.cos(alpha)

    def forward(self, X):
        """
        :param X: (batch_size, num_steps, num_embed), num_steps <= max_len
        :return:
        """
        X = X + self.P[:, :X.shape[1], :]
        return self.dropout(X)




