import torch
from torch.nn.functional import softmax

import LinearModule.Loader as Loader

"""
MLP,多层感知机,即前馈神经网络
基本结构为：
输入层->(仿射变换)->隐藏层->(激活函数)->...->输出层
回归问题:输出层维度为1,使用平方损失函数
分类问题:输出层维度为n,使用softmax后用交叉熵损失函数
"""

num_input = 784
num_hidden = 256
num_output = 10
batch_size = 100
num_epoch = 5
lr = 0.1

# 定义要训练的参数
W1 = torch.zeros(num_input, num_hidden)
B1 = torch.zeros(num_hidden)
W2 = torch.zeros(num_hidden, num_output)
B2 = torch.zeros(num_output)
torch.nn.init.normal_(W1, mean=0, std=0.01)
torch.nn.init.normal_(W2, mean=0, std=0.01)
params = [W1, B1, W2, B2]
for param in params:
    param.requires_grad = True


# 定义激活函数relu
def relu(X: torch.Tensor):
    """
    :param X: 对应一批样本(batch_size, num_hidden)
    :return:
    """
    return torch.max(X, other=torch.tensor(0.0))


# 定义MLP
def mlp(batch_x: torch.Tensor):
    """
    :param batch_x: 形状为(batch_size, num_input),一行为一个样本
    :return: 形状为(batch_size, num_output),一行对应一个样本各类别的得分
    """
    H = relu(batch_x.mm(W1) + B1)
    return H.mm(W2) + B2


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
            scores = mlp(X)
            batch_mean_loss = loss(scores, y)  # 对scores进行softmax后与y求交叉熵损失
            batch_mean_loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print("epoch %d, last batch mean loss: %f" % (epoch + 1, batch_mean_loss.item()))

    # 进行预测
    for X, y in test_iter:
        X = X.view(-1, num_input)
        outputs = mlp(X)
        pred_num = predict(outputs)
        pred_labels = mnist_data.get_labels(pred_num.numpy())
        true_labels = mnist_data.get_labels(y.numpy())
        labels = [pred + '\n' + truth for pred, truth in zip(pred_labels, true_labels)]
        Loader.paint_images(X, labels)
        break
