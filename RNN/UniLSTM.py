import torch
from torch import nn

"""
LSTM神经网络的网络参数是RNN的4倍
每一个神经元对应4个(线性变换+激活函数)分别得到遗忘门f、输入门i、输出门o、候选状态c'
每一个线性变换都是以上一时刻同层神经元的外部状态和同一时刻下一层神经元的外部状态作为输入
每个神经元还要接收上一个时刻自己的内部状态c_{t-1}
每个神经元最终产生一个外部状态向上层传递、向下一时刻传递,产生一个内部状态传递给下一时刻的本神经元

LSTM同RNN一样接收一批序列、对应的初始外部状态,此外LSTM还要接收对应的内部状态。
内部状态用三维张量表示,每一个竖切面对应一个输入序列时LSTM的内部状态,竖切面每一行对应一层。多个竖切面对应一批输入序列
"""


class UniLSTM(nn.Module):

    def __init__(self, xt_size, ht_size, num_layers):
        super(UniLSTM, self).__init__()
        self.net = nn.LSTM(xt_size, ht_size, num_layers)

    def forward(self, a_batch_of_seqs, initial_outer_state, initial_inner_state):
        output, final_outer_state, final_inner_state = self.net(a_batch_of_seqs, initial_outer_state,
                                                                initial_inner_state)
        return output, final_outer_state, final_inner_state


# 输入的x维度为10,每一时刻的输出维度为10,一共两层神经元
lstm = UniLSTM(10, 10, 2)

# 输入的每个序列长度为5,每个词x的维度为10,一批一共200个序列
seq_batch = torch.rand(5, 200, 10)

# 初始的外部状态
# 每一个竖切面有2行对应2层神经元,每一层10个神经元,一共200个竖切面对应一批
h0 = torch.rand(2, 200, 10)

# 初始的内部状态
# 每一个竖切面有2行对应2层神经元,每一层10个神经元,一共200个竖切面对应一批
c0 = torch.rand(2, 200, 10)
