import torch

from torch import nn

"""
nn.Linear全连接层(仿射变换)的使用
"""

# 定义一个全连接层
xt_size = 5  # 每个神经元接收5个输入
yt_size = 8  # 一层8个神经元
linear = nn.Linear(xt_size, yt_size)

# linear的输入形状为(批量大小,输入维度),每一行对应一个输入
# linear的输出形状为(批量大小,输出维度),每一行对应一个输出
batch_size = 10
xt_batch = torch.rand(batch_size, xt_size)
yt_batch = linear(xt_batch)

print(yt_batch)
