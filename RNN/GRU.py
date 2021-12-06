from torch import nn

"""
GRU(门控循环单元)
每一个神经元接收上一时刻同层神经元的输出和本时刻下一层神经元的输出,根据两个(仿射变换+激活函数)分别得到重置门r和更新门z
将上一时刻的同层的神经元的输出通过重置门后在与本时刻下一层神经元的输出,经过一个(仿射变换+激活函数)得到候选状态
将上一时刻的同层神经元的输出通过更新门,将候选状态通过1-z门,两个输出相加得到该神经元输出
GRU和RNN接收一样形状的输入,产生一样形状的输出
"""


class UniGRU(nn.Module):
    def __init__(self, xt_size, ht_size, num_layers):
        super(UniGRU, self).__init__()
        self.gru = nn.GRU(xt_size, ht_size, num_layers)

    def forward(self, a_batch_of_seqs, initial_state):
        output, final_state = self.gru(a_batch_of_seqs, initial_state)
        return output, final_state


class BiGRU(nn.Module):
    def __init__(self, xt_size, ht_size, num_layers):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(xt_size, ht_size, num_layers, bidirectional=True)

    def forward(self, a_batch_of_seqs, initial_state):
        output, final_state = self.gru(a_batch_of_seqs, initial_state)
        return output, final_state