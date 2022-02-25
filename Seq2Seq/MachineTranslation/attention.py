import torch
from torch import nn
from torch.nn.functional import softmax


class Attention(nn.Module):
    def __init__(self, input_size, attention_size):
        super(Attention, self).__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(input_size, attention_size), nn.Tanh())
        self.output_layer = nn.Linear(attention_size, 1)

    def forward(self, enc_states, dec_state):
        """
        :param enc_states: (max_seq_len, batch_size, num_hidden_of_encoder),每一竖切面对应一个输入序列的encoder的各时间步的隐藏状态
        :param dec_state: (batch_size, num_hidden_of_decoder),每一行对应该时间步decoder一个输入序列的隐藏状态
        :return:  (batch_size, num_hidden_of_encoder),每一行,对应decoder一个样本对encoder各时间步隐藏状态的注意力输出向量
        """
        dec_states = dec_state.unsqueeze(dim=0).expand_as(
            enc_states)  # (max_seq_len, batch_size, num_hidden_of_decoder),每一个竖切面的每一行相同,对应同一时间步decoder一个输入序列的隐藏状态
        enc_and_dec_states = torch.cat((enc_states, dec_states),
                                       dim=2)  # (max_seq_len, batch_size, num_hidden_of_encoder+num_hidden_of_decoder)
        e = self.output_layer(
            self.hidden_layer(enc_and_dec_states))  # (max_seq_len, batch_size, 1), 在attention计算时将每一个竖切面看成一个批次
        alpha = softmax(e, dim=0)  # (max_seq_len, batch_size, 1),每一个竖切面对应decoder一个样本对encoder各时间步隐藏状态的注意力分布
        c = (alpha * enc_states).sum(
            dim=0)  # (batch_size, num_hidden_of_encoder),每一行,对应decoder一个样本对encoder各时间步隐藏状态的注意力输出向量
        return c
