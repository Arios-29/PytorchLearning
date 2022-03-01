import torch.nn.functional


def sequence_mask(X, valid_len, value=0.0):
    """
    :param X: (seq_len*batch_size, num_output)
    :param valid_len: (seq_len*batch_size),每一元素为对应X一行的有效长度
    :param value:
    :return:
    """
    max_len = X.shape[1]
    mask = torch.arange(max_len).view(1, -1)
    valid_len = valid_len.view(-1, 1)
    X[mask >= valid_len] = value  # 将有效长度外的元素改为value
    return X


def masked_softmax(X, valid_lens=None):
    """
    :param X: 三维张量,对每一层的每一行作softmax运算 (a, b, c)
    :param valid_lens: 如果为一维张量(a),每一元素指定每一层各行的有效softmax长度;如果为二维张量(a, b),每一元素指定对应层某一行的有效softmax长度
    :return:
    """
    if valid_lens is None:
        return torch.nn.functional.softmax(X, dim=-1)  # dim=-1指定按最后一维进行softmax,即每一层的每一行
    else:
        shape = X.size()
        # 将valid_lens形状变为(seq_len*batch_size)
        if valid_lens.dim() == 1:
            valid_lens = torch.repeat_interleave(valid_lens, shape[1])
        else:
            valid_lens = valid_lens.view(-1)
        X = sequence_mask(X.view(-1, shape[-1]), valid_lens, value=-1e6)
        return torch.nn.functional.softmax(X.view(shape), dim=-1)


if __name__ == "__main__":
    x = torch.Tensor([[1, 2, 3], [4, 5, 6]])
    print(sequence_mask(x, torch.Tensor([1, 2])))
    print(masked_softmax(torch.rand(2, 2, 4), torch.tensor([2, 3])))
