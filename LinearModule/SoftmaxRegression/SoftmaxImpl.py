import sys
from typing import List

import torch
from matplotlib import pyplot as plt
import torch.utils.data as Data
import torchvision.datasets
from torchvision import transforms

"""
softmax被用于n分类问题
对样本输入进行n个线性回归(全连接层),第i个线性回归的输出表示该样本为第i类的几率(得分)
将所有几率进行softmax处理得到该样本对应各类的概率(预测分布)

样本的标签j表示该样本是第j类,则其one-hot向量表示标签可以看成该样本对应各类的真实概率分布
可以计算真实分布与预测分布的交叉熵,当交叉熵越小则说明两个分布越接近

按批量进行迭代,则可以用一批量样本的平均交叉熵作为损失函数
"""

#  获取训练数据集和测试集
# train=True表示下载训练集,通过transforms.ToTensor()将图片转化为张量
# 每一张图片的张量形状为(1,长度像素数,宽度像素数),每一个图片对应一个数字标签
# mnist_train[i]表示(图片i张量,标签)
mnist_train = torchvision.datasets.FashionMNIST(root='-/Datasets/FashionMNIST', train=True, download=True,
                                                transform=transforms.ToTensor())
mnist_test = torchvision.datasets.FashionMNIST(root='-/Datasets/FashionMNIST', train=False, download=True,
                                               transform=transforms.ToTensor())
train_data_nums = len(mnist_train)
test_data_nums = len(mnist_test)


# FashionMNIST数据集中一共有10类图片
def get_fashion_mnist_labels(labels: List) -> List:
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle', 'boot']
    return [text_labels[i] for i in labels]


# 根据张量画出图片
def show_fashion_mnist(images: List, labels: List) -> None:
    """
    :param images: 图片张量列表
    :param labels: 图片标签名列表
    """
    _, figs = plt.subplots(1, len(images), figsize=(12, 12))
    for fig, image, label in zip(figs, images, labels):  # fig是一张子图
        fig.imshow(image.view(28, 28).numpy())  # 将图片张量转化成28x28的numpy数组
        fig.set_title(label)  # 以图片标签为子图标题
        fig.axes.get_xaxis().set_visible(False)  # 不显示x轴
        fig.axes.get_yaxis().set_visible(False)  # 不显示y轴
    plt.show()


X, y = [], []
for i in range(10):
    X.append(mnist_train[i][0])
    y.append(mnist_train[i][1])
show_fashion_mnist(X, get_fashion_mnist_labels(y))

#  读取数据
batch_size = 100
if sys.platform.startswith('win'):
    num_workers = 0
else:
    num_workers = 4  # 读取数据的线程数
train_iter = Data.DataLoader(mnist_train, shuffle=True, batch_size=batch_size, num_workers=num_workers)
test_iter = Data.DataLoader(mnist_test, shuffle=True, batch_size=batch_size, num_workers=num_workers)

x_size = 784  # 每张图片一共784个像素
y_size = 10  # 对应10个类别的概率分布
# 初始化参数
W = torch.zeros(x_size, y_size)  # 每一列对应一个线性变化的权重参数
B = torch.zeros(1, y_size)  # 每一列对应一个线性变化的偏移参数
torch.nn.init.normal_(W, mean=0, std=0.01)
W.requires_grad = True
B.requires_grad = True


#  定义模型
def linear_layer(batch_x: torch.Tensor):
    """
    :param batch_x: 读取出来的一批样本
    :return: batch_o, 形状为(批量大小, y_size),即每一行是一个样本各类别的几率(得分)
    """
    batch_x = batch_x.view(-1, x_size)
    return batch_x.mm(W) + B


def softmax(batch_o: torch.Tensor):
    """
    :param batch_o: 形状为(批量大小, y_size),即每一行是一个样本各类别的几率(得分)
    :return: 输出形状(批量大小, y_size),每一行是一个样本各类别的概率
    """
    batch_o = batch_o.exp()
    partition = batch_o.sum(dim=1, keepdim=True)  # 每一行相加,得到大小为(批量大小,1)的张量,每一行是一个样本softmax操作中的分母
    return batch_o / partition  # 广播机制


def net(batch_x):
    return softmax(linear_layer(batch_x))


# 定义损失函数
def cross_entropy_loss(predict_y, y):
    """
    :param predict_y: 形状为(batch_size, y_size),即每一行是一个样本的各类别概率预测值
    :param y: 形状为(1, batch_size),每一列对应各个样本的实际类别,需要转化成(batch_size,1)形状
    :return: 返回一个批量的平均交叉熵损失
    """
    # predict_h[i][j]=predict_y[i][y.view(-1,1)[i][j]]
    # 形状为(batch_size,1),每一行对应一个样本的交叉熵
    predict_h = -predict_y.gather(1, y.view(-1, 1)).log()
    batch_loss = predict_h.mean()
    return batch_loss


# 定义优化算法
def sgd(params, lr):
    for param in params:
        param.data -= lr * param.grad  # tensor的data和grad形状一样


lr = 0.1
num_epochs = 5
if __name__ == '__main__':
    # 训练模型
    for epoch in range(num_epochs):
        for X, y in train_iter:
            predict_y = net(X)
            batch_mean_loss = cross_entropy_loss(predict_y, y)
            batch_mean_loss.backward()
            sgd([W, B], lr)
            W.grad.data.zero_()
            B.grad.data.zero_()
        print("epoch: %d, last batch mean loss: %f" % (epoch + 1, batch_mean_loss.item()))
    # 测试模型
    acc_sum = 0
    x_num = 0
    for X, y in test_iter:
        predict_y = net(X)
        acc_sum += (predict_y.argmax(dim=1) == y).float().sum().item()  # 分类正确的数目
        x_num += predict_y.size()[0]
    print(acc_sum / x_num)  # 正确率
