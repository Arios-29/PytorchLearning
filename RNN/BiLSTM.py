import torch
from torch import nn

"""
双向LSTM的参数是对应单向LSTM的两倍
双向LSTM的内部状态是对应单向LSTM的两倍
双向LSTM的外部状态是对应单向LSTM的两倍
指定参数bidirectional=True设置一个双向LSTM
"""


class BiLSTM(nn.Module):
    def __init__(self, xt_size, ht_size, num_layers):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(xt_size, ht_size, num_layers, bidirectional=True)

    def forward(self, a_batch_of_seqs, initial_outer_state, initial_inner_state):
        output, final_outer_stater, final_inner_state = self.lstm(a_batch_of_seqs, initial_outer_state,
                                                                  initial_inner_state)
        return output, final_outer_stater, final_inner_state


# 每一时刻输入的词x维度为10,输出的维度为10,一共两层
bi_lstm = BiLSTM(10, 10, 2)

# 每个序列的长度为5,序列中每个词x维度为10,一共200个序列
seq_batch = torch.rand(5, 200, 10)

# 每个竖切面对应一个初始外部状态, 竖切面每一行对应一层的前向或后向初始外部状态, 一共200个初始外部状态对应一批输入
h0 = torch.rand(4, 200, 10)

# 每个竖切面对应一个初始内部状态, 竖切面每一行对应一层的前向或后向初始内部状态, 一共200个初始外部状态对应一批输入
c0 = torch.rand(4, 200, 10)
