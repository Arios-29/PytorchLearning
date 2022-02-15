from torch import nn


class MLPDropout(nn.Module):
    def __init__(self, num_input, num_hidden_1, num_hidden_2, num_output, p1=0.2, p2=0.5):
        super(MLPDropout, self).__init__()
        self.hidden_layer_1 = nn.Sequential(nn.Linear(num_input, num_hidden_1), nn.ReLU(), nn.Dropout(p1))
        self.hidden_layer_2 = nn.Sequential(nn.Linear(num_hidden_1, num_hidden_2), nn.ReLU(), nn.Dropout(p2))
        self.output_layer = nn.Linear(num_hidden_2, num_output)

    def forward(self, x_batch):
        H1 = self.hidden_layer_1(x_batch)
        H2 = self.hidden_layer_2(H1)
        scores = self.output_layer(H2)
        return scores


num_input = 784
num_hidden_1 = 256
num_hidden_2 = 256
num_output = 10
batch_size = 100
num_epoch = 5
lr = 0.1
p1 = 0.2  # 第一层丢弃层的丢弃概率
p2 = 0.5  # 第二层丢弃层的丢弃概率
