import random

import torch
from matplotlib import pyplot as plt

"""
实现线性回归的计算过程
"""
example_num = 100  # 一共100个样本
x_size = 2  # 每个样本用2维向量表示
batch_size = 20  # 一批20个样本

#  初始化参数
w = torch.zeros(x_size, 1)
torch.nn.init.normal_(w, mean=0, std=0.01)  # 以norm(0,0.01**2)随机初始化w
b = torch.zeros(1).float()  # b初始化为0
w.requires_grad = True
b.requires_grad = True


# 定义线性回归模型
# X的形状为(batch_size, x_size),即每一行是一个样本
# 输出的形状为(batch_size, 1),即每一行是一个预测标记
def linear(X, w, b):
    return X.mm(w) + b


# 定义平方损失函数
def squared_loss(predict_y, y):
    return ((predict_y - y) ** 2) / 2


# 定义优化算法
def sgd(params, lr, batch_size):
    for param in params:
        param.data -= (lr * param.grad) / batch_size  # Variable的data和grad形状一样


# 生成样本数据
# data_x的每一行是一个样本
data_x = torch.zeros(example_num, 2)
torch.nn.init.normal_(data_x, mean=0, std=1)
true_w = torch.tensor([[2], [-3.4]])
true_b = torch.tensor([4.2])
labels = data_x.mm(true_w) + true_b
# 引入噪声
noise = torch.zeros(labels.size())
torch.nn.init.normal_(noise, mean=0, std=0.01)
labels += noise


# 读取数据
def data_iter(batch_size, data_x, labels):
    indexes = list(range(example_num))
    random.shuffle(indexes)
    for i in range(0, example_num, batch_size):
        j = torch.LongTensor(indexes[i: min(i + batch_size, example_num)])
        # index_select, 第一个参数0表示按行选取,第二个参数为对应的行号组成的张量
        batch_x = data_x.index_select(0, j)
        batch_y = labels.index_select(0, j)
        yield batch_x, batch_y


# 训练模型
lr = 0.03
num_epochs = 10
for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, data_x, labels):  # 每个epoch要用完一次所有数据
        loss = squared_loss(linear(X, w, b), y)  # loss的形状为(batch_size, 1),每一行对应一个样本输入的损失
        batch_loss = loss.sum()  # 计算一个批量的总损失
        batch_loss.backward()  # 反向求导
        sgd([w, b], lr=lr, batch_size=batch_size)  # 参数优化
        # 每次优化后对参数梯度清零
        w.grad.data.zero_()
        b.grad.data.zero_()
    mean_of_all_loss = squared_loss(linear(data_x, w, b), labels).mean()
    print("epoch %d , mean loss: %f" % (epoch + 1, mean_of_all_loss.item()))
print(true_w, '\n', w)
print(true_b, '\n', b)

# 画出散点图(x1,x2,y)和学习后的函数拟合图
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter(data_x[:, 0].numpy(), data_x[:, 1].numpy(), labels.numpy(), cmap='Blues')
# 要求梯度的tensor不可转化numpy的数组类型,用detach复制一个与原tensor共享内存但是不在计算图中的tensor
all_predict = linear(data_x, w, b).detach()
ax.plot3D(data_x[:, 0].numpy(), data_x[:, 1].numpy(), all_predict[:, 0].numpy(), 'gray')
plt.show()
