import torch
from torch import nn

from Seq2Seq.MachineTranslation.attention import Attention


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, num_hidden, num_layers, attention_size, drop_prob=0):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim=embed_size)
        self.attention = Attention(2 * num_hidden, attention_size)
        self.rnn = nn.GRU(embed_size + num_hidden, num_hidden, num_layers,
                          dropout=drop_prob)  # 将attention的输出向量与嵌入层输出拼接输入rnn
        self.out = nn.Linear(num_hidden, vocab_size)

    def forward(self, cur_input, state, enc_states):
        """
        :param cur_input: (batch_size)
        :param state: (num_layers, batch_size, num_hidden)
        :param enc_states: (max_seq_len, batch_size, num_hidden)
        :return: (batch_size, vocab_size), state
        """
        c = self.attention(enc_states, state[-1])
        input_and_c = torch.cat((self.embedding(cur_input.long()), c), dim=1)  # (batch_size, embed_size + num_hidden)
        output, state = self.rnn(input_and_c.unsqueeze(0), state)  # 只进行一个时间步
        # output: (1, batch_size, num_hidden)
        output = output.squeeze(dim=0)
        # output: (batch_size, num_hidden)
        return self.out(output), state

    def begin_state(self, enc_state):
        """
        :param enc_state: (num_layers, batch_size, num_hidden)
        :return:
        """
        return enc_state

