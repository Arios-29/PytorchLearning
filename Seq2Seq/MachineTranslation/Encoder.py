from torch import nn


class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hidden, num_layers, drop_prob=0):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_size)  # 输出(batch_size, embed_size)的一批词向量
        self.rnn = nn.GRU(embed_size, num_hidden, num_layers, dropout=drop_prob)

    def forward(self, inputs, state):
        """
        :param inputs: 形状为(batch_size, max_seq_len)
        :param state: 编码器的初始状态, (num_layers, batch_size, num_hidden)
        :return: (max_seq_len, batch_size, num_hidden),(num_layers, batch_size, num_hidden)
        """
        word_embeddings = self.embedding(
            inputs.long())  # (batch_size, max_seq_len, embed_size),每一个横切面对应一个输入序列,横切面每一行对应一个词向量
        word_embeddings = word_embeddings.permute(1, 0, 2)  # (max_seq_len, batch_size, embed_size),每一个竖切面对应一个输入序列
        return self.rnn(word_embeddings, state)

    def begin_state(self):
        return None
