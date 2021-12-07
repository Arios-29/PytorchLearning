import torch

"""
tensor.size()或tensor.shape()返回一个torch.Size对象
tensor的size为(k1, k2, ... , kn),说明这是一个n维张量, k1表示最高维度, k0=1是默认的最低维度
"""
# torch.Size()接收一个元组,创建一个Size对象
size = (2, 3)
size = torch.Size(size)
tensor_1 = torch.rand(size)
tensor_2 = torch.rand(tensor_1.size())
print(tensor_2.size())

# numel()计算张量中的元素总数
print(tensor_2.numel())

# tensor.view(size)用tensor的元素按指定形状返回一个张量
tensor_3 = torch.arange(0, 6)
tensor_3 = tensor_3.view(2, 3)
print(tensor_3)

# tensor.unsqueeze(m)在km后加一个维度,取值1
# k0后面是k1
tensor_4 = torch.arange(0, 10)
tensor_4 = tensor_4.unsqueeze(1)

# tensor.squeeze(m)将km后的维度值为1的维度压缩
tensor_5 = torch.rand((2, 1, 3))
tensor_5 = tensor_5.squeeze(1)

# tensor.squeeze()将所有维度值为1的维度压缩
tensor_6 = torch.rand((1, 1, 1, 2, 1, 3))
tensor_6 = tensor_6.squeeze()
print(tensor_6.size())

# tensor.size()可以返回torch.size类型或者元组类型
tensor_7 = torch.rand(3, 3)
row, col = tensor_7.size()
print(row, col)
