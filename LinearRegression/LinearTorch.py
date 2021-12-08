from torch import nn, optim
import torch.utils.data as Data

from LinearRegression import LinearImpl

"""
用nn.linear实现线性回归
"""


# 定义线性回归函数
class LinearRegression(nn.Module):
    def __init__(self, x_size):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(x_size, 1)

    def forward(self, x_batch):
        return self.linear(x_batch)


# 使用LinearImpl中的数据
x_size = LinearImpl.x_size
data_x = LinearImpl.data_x
labels = LinearImpl.labels
true_w = LinearImpl.true_w
true_b = LinearImpl.true_b

# 读取数据
batch_size = LinearImpl.batch_size
dataset = Data.TensorDataset(data_x, labels)  # 将数据封装成数据集合类型
date_iter = Data.DataLoader(dataset, batch_size=batch_size, shuffle=True)  # 通过DataLoader抽取一批样本

net = LinearRegression(x_size)
# 初始化参数
nn.init.normal_(net.linear.weight, mean=0, std=0.01)
nn.init.constant_(net.linear.bias, val=0)

# 定义平方损失函数
# nn.MSELoss()求一个批量的平均平方损失
squared_loss = nn.MSELoss()

# 定义优化函数
lr = LinearImpl.lr
optimizer = optim.SGD(net.parameters(), lr=lr)

# 训练模型
num_epochs = LinearImpl.num_epochs
for epoch in range(num_epochs):
    for X, y in date_iter:
        batch_mean_loss = squared_loss(net(X), y.view(-1, 1))
        batch_mean_loss.backward()
        optimizer.step()
        optimizer.zero_grad()  # 每次更新完参数把梯度归零
    print("epoch %d last batch mean loss %f" % (epoch+1, batch_mean_loss.item()))
print(true_w, '\n', net.linear.weight)
print(true_b, '\n', net.linear.bias)

