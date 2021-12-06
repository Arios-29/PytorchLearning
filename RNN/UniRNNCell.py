from torch import nn

"""
nn.RNNCell相当于nn.RNN中的一个时刻,即nn.RNNCell接收词,产生对应的隐状态。而nn.RNN接收序列,直接产生各个时刻的隐状态和最终隐状态
nn.RNN每一层神经元的数目都是相同的
nn.RNNCell可以拼接出每层神经元数目不同的多层nn.RNNCell
nn.RNNCell每次接收的一批数据为(批量大小,词维度),产生对应时刻的一批输出(批量大小,输出维度)
"""


# 定义一个两层的循环神经网络
class UniRNN(nn.Module):
    def __init__(self, xt_size, nums_layer1, ht_size):
        self.rnn_layer1 = nn.RNNCell(xt_size, nums_layer1)
        self.rnn_layer2 = nn.RNNCell(nums_layer1, ht_size)

    def forward(self, a_batch_of_seqs, initial_state_layer1, initial_state_layer2):
        h1 = initial_state_layer1
        h2 = initial_state_layer2
        output = []
        for xt in a_batch_of_seqs:  # 一批序列(序列长度,批量大小,词维度)
            h1 = self.rnn_layer1(xt, h1)
            h2 = self.rnn_layer2(h1, h2)
            output.append(h2)  # 记录各个时刻的输出
        final_state = [h1, h2]
        return output, final_state



