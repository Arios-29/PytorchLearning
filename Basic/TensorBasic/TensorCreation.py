import torch

"""
tensor: 张量
零维张量表示标量,一维张量表示向量(张量维度的取值对应向量的维度),二维张量表示矩阵
"""

# torch.tensor()创建张量
# 指定形状和内容
# tensor_2为二维张量,形状为(2, 3)
tensor_2 = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(tensor_2)

# torch.ones(size)创建元素全为1的张量
tensor_3 = torch.ones(4, 4)
print(tensor_3)

# torch.zeros(size)创建元素全为0的张量
tensor_4 = torch.zeros(4, 4)
print(tensor_4)

# torch.eye(size)创建对角线全为1,其余元素全为0的张量
tensor_5 = torch.eye(3, 4)
print(tensor_5)

# torch.rand(size)或torch.randn(size)创建元素随机的张量
tensor_6 = torch.rand(2, 3)
print(tensor_6)

# torch.arange(start, end, step)创建一维张量
# 从[start, end)中间隔step取值作为元素
tensor_7 = torch.arange(1, 10, 2)
print(tensor_7)

# torch.linspace(start, end, parts)创建一维张量
# 从[start, end)均匀间隔取parts个数作元素
tensor_8 = torch.linspace(1, 10, 5)
print(tensor_8)

# torch.randperm(m)创建一个元素随机的一维张量,元素个数为m个
tensor_9 = torch.randperm(8)
print(tensor_9)

