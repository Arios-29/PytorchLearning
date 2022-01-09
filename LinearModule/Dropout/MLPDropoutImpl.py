import torch
from torch.nn.functional import softmax

from LinearModule import Loader

"""
丢弃层以p的概率随机丢弃上一层的一些隐状态,可以起到正则化，避免过拟合的作用
丢弃层只在训练时起作用,测试时应去掉丢弃层
"""

num_input = 784
num_hidden_1 = 256
num_hidden_2 = 256
num_output = 10
batch_size = 100
num_epoch = 5
lr = 0.1
p1 = 0.2  # 第一层丢弃层的丢弃概率
p2 = 0.5  # 第二层丢弃层的丢弃概率

# 定义要训练的参数
W1 = torch.zeros(num_input, num_hidden_1)
B1 = torch.zeros(num_hidden_1)
W2 = torch.zeros(num_hidden_1, num_hidden_2)
B2 = torch.zeros(num_hidden_2)
W3 = torch.zeros(num_hidden_2, num_output)
B3 = torch.zeros(num_output)
torch.nn.init.normal_(W1, mean=0, std=0.01)
torch.nn.init.normal_(W2, mean=0, std=0.01)
torch.nn.init.normal_(W3, mean=0, std=0.01)
params = [W1, B1, W2, B2, W3, B3]
for param in params:
    param.requires_grad = True


# 定义丢弃层
def dropout(X, p):
    """
    :param X: 对应一批样本(batch_size, num_hidden_x)
    :param p: 丢弃的概率
    :return:
    """
    X = X.float()
    assert 0 <= p <= 1
    if p == 1:
        return torch.zeros_like(X)
    else:
        q = 1 - p
        mask = (torch.randn(X.size()) < q).float()  # mask[i][j]以q的概率为1, p的概率为0, 则X的原始以q的概率被拉伸, 以p的概率被丢弃
        return mask * X / q


# 定义激活函数relu
def relu(X: torch.Tensor):
    """
    :param X: 对应一批样本(batch_size, num_hidden)
    :return:
    """
    return torch.max(X, other=torch.tensor(0.0))


# 定义多层感知机
def mlp(batch_x: torch.Tensor, is_training: bool):
    """
    :param batch_x: 一批样本(batch_size, num_input)
    :param is_training: True表示训练过程,否则为测试过程
    :return: 形状为(batch_size, num_output),一行对应一个样本各类别的得分
    """
    H1 = relu(batch_x.mm(W1) + B1)
    if is_training:
        H1 = dropout(H1, p1)
    H2 = relu(H1.mm(W2) + B2)
    if is_training:
        H2 = dropout(H2, p2)
    return H2.mm(W3) + B3


# 定义训练好的MLP的预测函数
def predict(output):
    return torch.max(softmax(output, dim=1), dim=1)[1]


# 定义损失函数, softmax和交叉熵损失一起计算
loss = torch.nn.CrossEntropyLoss()

# 定义优化函数
optimizer = torch.optim.SGD(params=params, lr=lr)


if __name__ == "__main__":
    mnist_data = Loader.MnistData()
    train_iter = mnist_data.get_train_iter(batch_size=batch_size)
    test_iter = mnist_data.get_test_iter(batch_size=10)

    # 进行训练
    for epoch in range(num_epoch):
        for X, y in train_iter:
            X = X.view(-1, num_input)
            scores = mlp(X, is_training=True)
            batch_mean_loss = loss(scores, y)  # 对scores进行softmax后与y求交叉熵损失
            batch_mean_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("epoch %d, last batch mean loss: %f" % (epoch + 1, batch_mean_loss.item()))

    # 进行预测
    for X, y in test_iter:
        X = X.view(-1, num_input)
        outputs = mlp(X, is_training=False)
        pred_num = predict(outputs)
        pred_labels = mnist_data.get_labels(pred_num.numpy())
        true_labels = mnist_data.get_labels(y.numpy())
        labels = [pred + '\n' + truth for pred, truth in zip(pred_labels, true_labels)]
        Loader.paint_images(X, labels)
        break

