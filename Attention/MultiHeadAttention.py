import torch
from torch import nn

from Attention.DotProductAttention import DotProductAttention


def transpose_qkv(X, num_heads):
    """
    :param X: (batch_size, nums, num_hidden), 每一num_hidden/num_heads的部分对应一个头
    :param num_heads:
    :return: (batch_size*num_heads, nums, num_hidden/num_heads)
    """
    X = X.view(X.shape[0], X.shape[1], num_heads, -1)
    # (batch_size, nums, num_heads, num_hidden/num_heads), 即batch_size个立方体,每个立方体对应一个样本,每个立方体:
    # 一共有nums层,每一竖切面对应一个头的线性变化后的各键(值或查询）,竖切面每一行对应一个键(值或查询)
    X = X.permute(0, 2, 1, 3)
    # (batch_size, num_heads, nums, num_hidden/num_heads), 即batch_size个立方体,每个立方体对应一个样本,每个立方体:
    # 一共有num_heads层,每一横切面对应一个头的线性变化后的各键(值或查询),横切面每一行对应一个键(值或查询)
    return X.view(-1, X.shape[2], X.shape[3])  # 将各立方体叠放


def transpose_output(X, num_heads):
    """
    :param X: (batch_size*num_heads, num, num_hidden/num_heads)
    :param num_heads:
    :return:
    """
    X = X.view(-1, num_heads, X.shape[1], X.shape[2])  # 分开各立方体,(batch_size, num_heads, num, num_hidden/num_head)
    X = X.permute(0, 2, 1, 3)  # (batch_size, num, num_heads, num_hidden/num_head)
    return X.view(X.shape[0], X.shape[1], -1)  # (batch_size, num, num_hidden)


class MultiHeadAttention(nn.Module):
    def __init__(self, key_size, value_size, query_size, num_proj, num_heads, drop_prob=0, bias=False):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.attention = DotProductAttention(drop_prob)
        # 各头的投影可以并行
        self.multi_key_proj = nn.Linear(key_size, num_heads * num_proj, bias=bias)
        self.multi_value_proj = nn.Linear(value_size, num_heads * num_proj, bias=bias)
        self.multi_query_proj = nn.Linear(query_size, num_heads * num_proj, bias=bias)
        self.output_layer = nn.Linear(num_heads * num_proj, num_heads * num_proj, bias=bias)

    def forward(self, queries, keys, values, valid_lens=None):
        """
        :param queries: (batch_size, num_queries, query_size)
        :param keys: (batch_size, num_keys, key_size)
        :param values: (batch_size, num_values, value_size)
        :param valid_lens: (batch_size)或(batch_size, num_queries)
        :return: (batch_size, num_queries, num_hidden), num_hidden = num_heads*num_proj
        """
        multi_projected_keys = transpose_qkv(self.multi_key_proj(keys), self.num_heads)
        multi_projected_values = transpose_qkv(self.multi_value_proj(values), self.num_heads)
        multi_projected_queries = transpose_qkv(self.multi_query_proj(queries), self.num_heads)
        # (batch_size*num_heads, num, num_hidden/num_heads)
        if valid_lens is not None:
            valid_lens = torch.repeat_interleave(valid_lens, repeats=self.num_heads, dim=0)
            # (batch_size*num_heads)或(batch_size*num_heads, num_queries)
        output = self.attention(multi_projected_keys, multi_projected_values, multi_projected_queries, valid_lens)
        # (batch_size*num_heads, num_queries, num_hidden/num_heads)
        output = transpose_output(output, self.num_heads)
        # (batch_size, num_queries, num_hidden),每一横切面一行对应同一原始查询对应多个头的注意力向量的拼接
        return self.output_layer(output)
