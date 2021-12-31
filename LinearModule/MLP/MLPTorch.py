import torch
from torch import nn

from LinearModule import Loader
from LinearModule.MLP.MLPImpl import predict

num_input = 784
num_hidden = 256
num_output = 10
batch_size = 100
num_epoch = 5
lr = 0.1


class MLP(nn.Module):
    def __init__(self, x_size, hidden_size, y_size):
        super(MLP, self).__init__()
        self.hidden_layer = nn.Sequential(nn.Linear(x_size, hidden_size), nn.ReLU())
        self.output_layer = nn.Linear(hidden_size, y_size)

    def forward(self, batch_x):
        hidden_state = self.hidden_layer(batch_x)
        scores = self.output_layer(hidden_state)
        return scores


mlp = MLP(num_input, num_hidden, num_output)
# 初始化参数
for param in mlp.parameters():
    nn.init.normal_(param, mean=0, std=0.01)

# 定义损失函数, softmax和交叉熵损失一起计算
loss = torch.nn.CrossEntropyLoss()

# 定义优化函数
optimizer = torch.optim.SGD(params=mlp.parameters(), lr=lr)

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
