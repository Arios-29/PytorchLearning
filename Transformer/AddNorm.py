from torch import nn


class AddNorm(nn.Module):
    """
    将两个输入相加(残差连接,第一个输入为X,第二个输入为F(X))后进行层归一化
    """
    def __init__(self, normalized_shape, drop_prob=0):
        super(AddNorm, self).__init__()
        self.dropout = nn.Dropout(drop_prob)
        self.layer_norm = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        """
        :param X: (batch_size, normalized_shape[0], normalized_shape[1]),即对每一层每一行单独归一化
        :param Y: 与X的形状一致
        :return: 与X的形状一致
        """
        return self.layer_norm(self.dropout(Y) + X)
