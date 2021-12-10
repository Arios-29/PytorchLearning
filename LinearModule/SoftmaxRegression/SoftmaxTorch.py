from torch import nn, optim

from LinearModule.SoftmaxRegression import SoftmaxImpl

"""
通过nn.Linear和nn.CrossEntropyLoss实现softmax回归
nn.CrossEntropyLoss的输入形状是(batch_size, y_size),一行对应一个样本各个类别的几率(得分),另一个输入是真实类别,形状为(batch_size),
即每一元素对应一个样本的真实类别标签
nn.CrossEntropyLoss先对一批得分进行softmax操作,再计算该批样本的平均交叉熵损失
即nn.CrossEntropyLoss等效于先LogSoftmax运算,再计算nn.NLLLoss
LogSoftmax先将得分归一化成概率分布,再取对数
"""


class SoftmaxRegression(nn.Module):
    def __init__(self, x_size, y_size):
        super(SoftmaxRegression, self).__init__()
        self.x_size = x_size
        self.linear = nn.Linear(x_size, y_size)

    def forward(self, x_batch):
        """
        :param x_batch: 一批样本
        :return: 返回一批样本的各类别得分,形状为(batch_size, y_size)
        """
        x_batch = x_batch.view(-1, self.x_size)
        return self.linear(x_batch)


x_size = SoftmaxImpl.x_size
y_size = SoftmaxImpl.y_size
lr = SoftmaxImpl.lr
epoch_num = SoftmaxImpl.num_epochs
train_iter = SoftmaxImpl.train_iter
test_iter = SoftmaxImpl.test_iter
net = SoftmaxRegression(x_size, y_size)
loss = nn.CrossEntropyLoss()
optimizer = optim.SGD(params=net.parameters(), lr=lr)

for epoch in range(epoch_num):
    for X, y in train_iter:
        predict_y = net(X)
        batch_mean_loss = loss(predict_y, y)
        batch_mean_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    print("epoch: %d, last batch mean loss: %f" % (epoch + 1, batch_mean_loss.item()))

# 测试模型
acc_num = 0
x_num = 0
for X, y in test_iter:
    predict_y = net(X)
    acc_num += (predict_y.argmax(dim=1) == y).float().sum().item()  # 分类正确的数目
    x_num += predict_y.size()[0]
print(acc_num / x_num)  # 正确率
