"""
当键向量k(k维向量)和查询向量q(q维向量)维度不同时用加性注意力机制计算注意力分布
a(k,q)=w*tanh(W1*k+W2*q)
W1形状为(h,k), W2形状为(h,q),w形状为(1,h)
在实际实现中可以表示为a(k,q)=w*tanh(W*(k,q)),其中W为[W1,W2]
"""
import torch
from torch import nn

from Attention.masked_softmax import masked_softmax


class AdditiveAttention(nn.Module):
    def __init__(self, key_size, query_size, num_hidden, drop_pro=0):
        super(AdditiveAttention, self).__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(key_size + query_size, num_hidden, bias=False), nn.Tanh())
        self.output_layer = nn.Linear(num_hidden, 1, bias=False)
        self.dropout = nn.Dropout(drop_pro)

    def forward(self, keys, values, queries, valid_lens=None):
        """
        :param keys: 形状为(batch_size, num_steps, key_size)
        :param values: 形状为(batch_size, num_steps, value_size)
        :param queries: 形状为(batch_size,query_size)
        :param valid_lens:
        :return:
        """
        queries = queries.view(-1, 1, queries.shape[-1]).expand(-1, keys.shape[1],
                                                                -1)  # (batch_size, num_steps, query_size)
        keys_and_queries = torch.cat((keys, queries), dim=2)
        hidden = self.hidden_layer(keys_and_queries)  # (batch_size, num_steps, num_hidden)
        scores = self.output_layer(hidden)  # (batch_size, num_steps, 1)
        scores = scores.transpose(1, 2)  # (batch_size, 1, num_steps)
        alpha = masked_softmax(scores, valid_lens)
        return torch.bmm(self.dropout(alpha), values).view(alpha.shape[0], -1)  # (batch_size, value_size)


if __name__ == "__main__":
    key_size = 10
    query_size = 8
    num_hidden = 12

    attention = AdditiveAttention(key_size, query_size, num_hidden)
    keys = torch.rand((2, 5, 10))
    values = torch.rand((2, 5, 10))
    queries = torch.rand((2, 8))
    print(attention(keys, values, queries))
