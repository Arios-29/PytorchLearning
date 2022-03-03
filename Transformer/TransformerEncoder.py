import math
from torch import nn
from Transformer.EncoderBlock import EncoderBlock
from Transformer.PositionalEncoding import PositionalEncoding


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, num_embed, num_heads, num_proj, norm_shape, ffn_num_hidden, num_blocks, drop_prob=0,
                 use_bias=False):
        super(TransformerEncoder, self).__init__()
        assert num_proj * num_heads == num_embed, "num_proj * num_heads should equal num_embed"
        self.num_embed = num_embed
        self.embedding = nn.Embedding(vocab_size, num_embed)
        self.pos_encoding = PositionalEncoding(num_embed, drop_prob=drop_prob)
        self.blocks = nn.Sequential()
        for i in range(num_blocks):
            self.blocks.add_module(name="block_" + str(i),
                                   module=EncoderBlock(key_size=num_embed, value_size=num_embed, query_size=num_embed,
                                                       num_proj=num_proj,
                                                       num_heads=num_heads, num_hidden=ffn_num_hidden,
                                                       drop_prob=drop_prob, use_bias=use_bias,
                                                       norm_shape=norm_shape))

    def forward(self, X, valid_lens=None):
        """
        :param X: (batch_size, num_steps, ..), norm_shape should be (n, num_embed), num_heads*num_proj = num_embed
        :param valid_lens:
        :return: (batch_size, num_steps, num_embed)
        """
        X = self.embedding(X)
        X = X * math.sqrt(self.num_embed)  # 对嵌入向量进行缩放
        X = self.pos_encoding(X)  # 加入位置信息
        return self.blocks(X, valid_lens)
