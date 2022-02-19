import torch

from LyricsGenerator.Loader import JayLyrics
from LyricsGenerator.grad_clipping import grad_clipping
from LyricsGenerator.one_hot import one_hot, get_index, to_one_hot

"""
自定义的简单RNN
"""

loader = JayLyrics()
# 超参数
num_inputs = loader.get_vocab_size()
num_hidden = 256
num_outputs = num_inputs


# 初始化训练参数
def get_params():
    def _get_a_param(shape):
        param = torch.zeros(shape)
        torch.nn.init.normal_(param, mean=0, std=0.01)
        return torch.nn.Parameter(param, requires_grad=True)

    # 输入层到隐藏层的参数
    W_xh = _get_a_param((num_inputs, num_hidden))
    b_xh = torch.nn.Parameter(torch.zeros(num_hidden), requires_grad=True)

    # 记忆参数
    W_hh = _get_a_param((num_hidden, num_hidden))
    b_hh = torch.nn.Parameter(torch.zeros(num_hidden), requires_grad=True)

    # 隐藏层到输出层的参数
    W_hq = _get_a_param((num_hidden, num_outputs))
    b_hq = torch.nn.Parameter(torch.zeros(num_outputs), requires_grad=True)

    return torch.nn.ParameterList([W_xh, b_xh, W_hh, b_hh, W_hq, b_hq])


params = get_params()


def init_rnn_state(batch_size, num_hidden):
    """
    :param batch_size:
    :param num_hidden:
    :return: 每一行对应一个样本输入时的初始状态,同一批样本输入时都是同样的初始状态
    """
    return torch.zeros((batch_size, num_hidden))


def rnn(inputs, state, params):
    """
    :param inputs: 形状为(batch_size, num_steps, n_class)的一批样本,每个横切面为一个样本,每个样本的一行为一个字符的one_hot向量表示
    :param state:
    :param params:
    :return: (num_steps, batch_size, num_outputs), 每个竖切面对应一个样本
    """
    inputs = inputs.transpose(0, 1)  # 将输入形状变为(num_steps, batch_size, n_class),即每个竖切面对应一个样本,每个横切面对应同一位置的字符one_hot向量
    W_xh, b_xh, W_hh, b_hh, W_hq, b_hq = params
    H = state
    outputs = torch.zeros(inputs.size())
    i = 0
    for x in inputs:
        # x为一个批量的同一位置的字符向量,即RNN每一时刻输入一个字符
        H = torch.tanh(torch.matmul(x, W_xh) + torch.matmul(H, W_hh) + b_xh + b_hh)  # b_xh和b_hh可以看成一个参数
        # (batch_size, n_class)*(n_class, num_hidden) + (batch_size, num_hidden)*(num_hidden,num_hidden)
        Y = torch.matmul(H, W_hq) + b_hq
        # (batch_size, num_hidden)*(num_hidden, num_outputs)
        outputs[i] = Y
        i += 1
        # Y为对应时刻的一个批量的输出,(batch_size, num_outputs)
    return outputs, H


def predict_rnn(prefixes, num_chars, rnn, params, init_rnn_state, num_hidden, vocab_size, index_to_char, char_to_index):
    state = init_rnn_state(1, num_hidden)
    output = [char_to_index[prefixes[0]]]  # 存放字符编号
    for t in range(len(prefixes) + num_chars - 1):
        x = one_hot(torch.tensor([[output[-1]]]), n_class=vocab_size)  # (1, n_class)
        x = x.view(1, 1, -1)  # 将一个样本包装成一个批量
        (Y, state) = rnn(x, state, params)
        if t < len(prefixes) - 1:
            output.append(char_to_index[prefixes[t + 1]])  # 将prefixes中的下一个字符作输入
        else:
            output.append(get_index(Y[0]))  # 将上一步的输出字符作输入
    return ''.join([index_to_char[index] for index in output])


def train(is_random_iter, rnn, params, init_rnn_state, num_hidden, vocab_size, num_steps, batch_size, num_epochs, lr,
          theta):
    if is_random_iter:
        data_iter = loader.random_data_iter(batch_size, num_steps)
    else:
        data_iter = loader.consecutive_data_iter(batch_size, num_steps)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params, lr)

    for epoch in range(num_epochs):
        if not is_random_iter:  # 连续采样在epoch开始时初始化初始状态
            state = init_rnn_state(batch_size, num_hidden)
        for X, Y in data_iter:
            if is_random_iter:
                state = init_rnn_state(batch_size, num_hidden)
            else:
                state = state.detach()
            inputs = to_one_hot(X, n_class=vocab_size)
            outputs, state = rnn(inputs, state, params)
            outputs = outputs.transpose(0, 1).contiguous()
            outputs = outputs.view(-1, outputs.shape[2])  # (batch_size*num_steps, n_class), 连续num_steps行对应同一样本
            y = Y.contiguous().view(-1)  # (batch_size*num_steps), 连续num_steps行对应一个样本的各字符编号
            loss_value = loss(outputs, y.long())
            optimizer.zero_grad()
            loss_value.backward()
            grad_clipping(params, theta)
            optimizer.step()
        if (epoch + 1) % 20 == 0:
            print("epoch: %d, loss: %f" % (epoch + 1, loss_value))


train(True, rnn, params, init_rnn_state, num_hidden, vocab_size=num_inputs, num_steps=35, batch_size=40,
      num_epochs=250, lr=1e2, theta=1e-2)
#print(params[1])
s = predict_rnn("分开", 50, rnn, params, init_rnn_state, num_hidden, num_inputs, loader.get_index_to_char(),
                loader.get_char_to_index())
print(s)
