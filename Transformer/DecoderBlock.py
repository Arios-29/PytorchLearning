import torch
from torch import nn

from Attention.MultiHeadAttention import MultiHeadAttention
from Transformer.AddNorm import AddNorm
from Transformer.PoistionWiseFNN import PositionWiseFNN


class DecoderBlock(nn.Module):
    def __init__(self, i, key_size, value_size, query_size, num_proj, num_heads, norm_shape, num_hidden, drop_prob=0,
                 use_bias=False):
        super(DecoderBlock, self).__init__()
        self.i = i
        self.attention_1 = MultiHeadAttention(key_size=key_size, value_size=value_size, query_size=query_size,
                                              num_heads=num_heads, num_proj=num_proj, drop_prob=drop_prob,
                                              bias=use_bias)
        self.add_norm_1 = AddNorm(normalized_shape=norm_shape, drop_prob=drop_prob)
        self.attention_2 = MultiHeadAttention(key_size=key_size, value_size=value_size, query_size=query_size,
                                              num_heads=num_heads, num_proj=num_proj, drop_prob=drop_prob,
                                              bias=use_bias)
        self.add_norm_2 = AddNorm(normalized_shape=norm_shape, drop_prob=drop_prob)
        self.fnn = PositionWiseFNN(num_input=num_proj * num_heads, num_hidden=num_hidden,
                                   num_output=num_proj * num_heads)
        self.add_norm_3 = AddNorm(normalized_shape=norm_shape, drop_prob=drop_prob)

    def forward(self, X, state):
        """
        :param X: (batch_size, num_steps, num_embed)
        :param state: [enc_output, enc_valid_lens, [....]]
        :return:
        """
        enc_output, enc_valid_lens = state[0], state[1]
        # 这一段代码只会在训练时用到
        # 在训练时decoder的输入是一个序列(强制教学,输入为标签序列),通过mask_multi_heads_attention让第i个词的查询向量只能对0~i个词求注意力分布,
        # 但是第i个词的mask_multi_heads_attention的输出可以作为查询,对encoder的输出求注意力分布得到注意力向量,也就是利用样本样例部分的所有信息
        # 每训练一批X后, state[2][i]会在下一批训练前情况,因此每一批训练时state[2][i]永远是空的
        # 在测试时,decoder首先输入<bos>,得到输出word1,然后输入word1,但是word1要利用前面已预测词的信息,而state[2][i]中就包含了前面词的信息
        # 将word1的信息拼接上去,作为mask_multi_heads_attention的键值对部分,word1作为查询向量,输出的只有word1综合了前面词的新信息,也就没有形状改变
        # 由于测试阶段没有后面词的输入,因此mask就没有设置必要了
        if state[2][self.i] is None:
            key_values = X
        else:
            key_values = torch.cat((state[2][self.i], X), dim=1)
        state[2][self.i] = key_values

        if self.training:
            batch_size, num_steps = X.shape[0], X.shape[1]
            dec_valid_lens = torch.arange(1, num_steps + 1).repeat(batch_size, 1)  # (batch_size, num_steps)
        else:
            dec_valid_lens = None

        # mask-multi-heads-attention
        X2 = self.attention_1(X, key_values, key_values, dec_valid_lens)
        Y = self.add_norm_1(X, X2)

        # 使用encoder的输出作keys、values
        Y2 = self.attention_2(Y, enc_output, enc_output, enc_valid_lens)
        Z = self.add_norm_2(Y, Y2)

        return self.add_norm_3(Z, self.fnn(Z)), state
