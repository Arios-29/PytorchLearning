from torch import nn

from Attention.MultiHeadAttention import MultiHeadAttention
from Transformer.AddNorm import AddNorm
from Transformer.PoistionWiseFNN import PositionWiseFNN


class EncoderBlock(nn.Module):
    """key_size=value_size=query_size=num_embed
       num_proj*num_heads = num_embed"""
    def __init__(self, key_size, value_size, query_size, num_proj, num_heads, norm_shape, num_hidden, drop_prob=0,
                 use_bias=False):
        """
        :param key_size: 输入键的维度长度
        :param value_size: 输入值的维度长度
        :param query_size: 查询向量的维度长度
        :param num_proj: 多头自注意力中一个头各个键、值、查询投影后的长度
        :param num_heads: 指定头数  num_proj * num_heads应该等于key_size=value_size=query_size=num_embed
        :param norm_shape: 归一化层的输入形状, 即每一层的形状(n, key_size=value_size=query_size)
        :param num_hidden: 前馈神经网络的隐藏层维度,应该设置为key_size=value_size=query_size
        :param drop_prob:
        :param use_bias:
        """
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(key_size=key_size, value_size=value_size, query_size=query_size,
                                            num_proj=num_proj, num_heads=num_heads, drop_prob=drop_prob, bias=use_bias)
        self.add_norm_1 = AddNorm(normalized_shape=norm_shape, drop_prob=drop_prob)
        self.fnn = PositionWiseFNN(num_input=num_proj * num_heads, num_hidden=num_hidden,
                                   num_output=num_proj * num_heads)
        self.add_norm_2 = AddNorm(normalized_shape=norm_shape, drop_prob=drop_prob)

    def forward(self, X, valid_lens=None):
        """
        :param X: (batch_size, num_steps, num_embed)
        :param valid_lens:
        :return:
        """
        Y = self.add_norm_1(X, self.attention(X, X, X, valid_lens))
        return self.add_norm_2(Y, self.fnn(Y))

