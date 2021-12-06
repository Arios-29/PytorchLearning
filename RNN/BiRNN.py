import torch
from torch import nn

"""
双向RNN的参数是对应单向RNN的两倍。
双向RNN的隐状态元素数目是单向RNN的两倍
通过指定参数bidirectional=True定义一个双向RNN
"""


class BiRNN(nn.Module):
    def __init__(self, xt_size, ht_size, num_layers):
        super(BiRNN, self).__init__()
        self.rnn = nn.RNN(xt_size, ht_size, num_layers, bidirectional=True)

    def forward(self, a_batch_of_seqs, initial_states):
        output, final_state = self.rnn(a_batch_of_seqs, initial_states)
        return output, final_state


# 每一时刻输入的词x的维度为10,每一时刻输出的维度为10,一共2层
bi_rnn = BiRNN(10, 10, 2)

# 每个序列长度为5,序列中每个词用10维向量表示,一批一共200个序列
seq_batch = torch.rand(5, 200, 10)

# 每一个竖切面对应一个初始的前向后向隐状态,竖切面每一行对应某一层的前向或后向隐状态
h0 = torch.rand(4, 200, 10)
