import torch
from torch import nn

"""
nn.RNN(input_size, hidden_size, num_layers)是pytorch已实现的循环网络
input_size是xt的维度,对应t时刻循环网络的输入层
hidden_size是ht的维度,对应t时刻循环网络的输出层
num_layers是网络的层数

nn.RNN可以接收一批长度相同的序列并行处理(一个Iteration),此时RNN的时间跨度就是序列的长度
一批长度相同的序列按三维张量(序列长度,一批中的序列数目,xt的维度)输入, 每一竖切面表示一个序列, 竖切面每一行是一个词xt。多个竖切面表示多个序列,构成一批。
nn.RNN还需要接收初始的隐藏状态(层数, 一批中的序列数目,每一层神经元的数目),每一个竖切面对应一个序列输入时的初始隐状态,竖切面每一行对应一层神经元隐状态。多竖切面与多个序列输入对应

nn.RNN产生两个输出
output:(序列长度,一批中的序列数目,输出层神经元的数目),每一个竖切面对应一个序列的全部输出,一个竖切面的一行对应某一个时刻的输出。多个竖切面表示与输入序列数目对应
final_state:(层数,一批中的序列数目,每一层神经元的数目),每一竖切面对应一个序列输入后RNN的最终隐状态,竖切面每一行对应一层神经元隐状态。多个竖切面对应各个序列输入后RNN的最终隐状态
"""


# 一个单向循环网络
class RNN(nn.Module):
    def __init__(self, xt_size, ht_size, num_layers):
        super(RNN, self).__init__()
        self.net = nn.RNN(xt_size, ht_size, num_layers)

    def forward(self, a_batch_of_seqs, initial_state):
        output, final_state = self.net(a_batch_of_seqs, initial_state)
        return output, final_state


basic_rnn = RNN(10, 10, 2)
# basic_rnn.weight_ih_l0为第一层的w参数,每一行对应第一层一个神经元的各w参数
w1_ = basic_rnn.weight_ih_l0

# basic_rnn.bias_ih_l0为第一层的偏置,每一行对应第一层一个神经元的偏置参数
bias_w = basic_rnn.bias_ih_l0

# basic_rnn.weight_hh_l0为第一层状态转移权重,每一行对应第一层一个神经元的状态转移参数
h1_ = basic_rnn.weight_hh_l0

# basic_rnn.bias_hh_l0为第一层状态转移偏置,每一行对应第一层一个神经元的状态转移偏置
bias_h = basic_rnn.bias_hh_l0
