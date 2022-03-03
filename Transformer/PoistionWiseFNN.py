from torch import nn


class PositionWiseFNN(nn.Module):
    def __init__(self, num_input, num_hidden, num_output):
        super(PositionWiseFNN, self).__init__()
        self.mlp = nn.Sequential(nn.Linear(num_input, num_hidden), nn.Tanh(), nn.Linear(num_hidden, num_output))

    def forward(self, X):
        """
        :param X: (batch_size, num_steps, num_input)
        :return: (batch_size, num_steps, num_output)
        """
        return self.mlp(X)
