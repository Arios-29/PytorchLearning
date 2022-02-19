import torch


def one_hot(x, n_class, dtype=torch.float32):
    """
    :param x: 字符编号表示的一个样本, 形状为(num_steps)
    :param n_class: 字符库的大小
    :param dtype:
    :return: 返回该样本的one-hot表示, 形状为(num_steps, n_class),即每一行对应一个字符的one_hot向量
    """
    x = x.long()
    res = torch.zeros(x.shape[0], n_class, dtype=dtype)
    res.scatter_(1, x.view(-1, 1), 1.0)
    return res


def to_one_hot(X, n_class):
    """
    :param X: 一批字符编号表示的样本,形状为(batch_size, num_steps)
    :param n_class: 字符库的大小
    :return: 形状为(batch_size, num_steps, n_class)
    """
    ret_torch = torch.zeros(X.shape[0], X.shape[1], n_class)
    for i in range(X.shape[0]):
        ret_torch[i] = one_hot(X[i, :], n_class=n_class)
    return ret_torch


def get_char(one_hot_present, index_to_char):
    """
    :param one_hot_present: (n_class)
    :param index_to_char:
    :return: 将一个one_hot向量转化成字符
    """
    index = int(one_hot_present.argmax(dim=1).item())
    return index_to_char[index]


def get_index(one_hot_present):
    return int(one_hot_present.argmax(dim=1).item())


if __name__ == "__main__":
    X = torch.arange(10).view(2, 5)
    ret = to_one_hot(X, 10)
    print(ret)
    print(ret.transpose(0, 1))

    obj = torch.arange(12).view(2, 2, 3)
    print(obj)
    obj2 = obj.view(-1, 3)
    print(obj2)
    obj3 = obj.transpose(0, 1).contiguous()
    print(obj3)
    obj4 = obj3.view(-1, 3)
    print(obj4)

    obj5 = torch.arange(4).view(2, 2)
    print(obj5)
    obj6 = obj5.view(-1)
    print(obj6)