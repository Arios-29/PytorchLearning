import collections
import io

import torch
from torch.utils import data
from torchtext.legacy.vocab import Vocab

PAD = '<pad>'
BOS = '<bos>'
EOS = '<eos>'


def process_one_seq(seq_tokens, all_tokens, all_seqs, max_seq_len):
    """
    :param seq_tokens: 待处理的序列,用列表表示,每一个元素为一个token
    :param all_tokens: 记录数据集中出现的所有token
    :param all_seqs: 记录数据集中出现的所有序列
    :param max_seq_len: 序列的最大长度
    :return:
    """
    all_tokens.extend(seq_tokens)  # 将seq_tokens中的token记录入all_tokens
    seq_tokens += [EOS] + [PAD] * (max_seq_len - len(seq_tokens) - 1)  # 序列末尾添加eos,用pad补齐长度
    all_seqs.append(seq_tokens)  # 将seq_tokens记录入all_seqs


def build_data(all_tokens, all_seqs):
    vocab = Vocab(collections.Counter(all_tokens), specials=[PAD, BOS, EOS])
    indices = [[vocab.stoi[word] for word in seq] for seq in all_seqs]
    return vocab, torch.Tensor(indices)  # 以编号表示的输入文本张量,(num_seqs, max_seq_len)


def read_data(max_seq_len):
    in_tokens, out_tokens, in_seqs, out_seqs = [], [], [], []
    with io.open('data/fr-en-small.txt') as file:
        lines = file.readlines()
    for line in lines:
        in_seq, out_seq = line.rstrip().split('\t')
        in_seq_tokens, out_seq_tokens = in_seq.split(' '), out_seq.split(' ')
        if max(len(in_seq_tokens), len(out_seq_tokens)) > max_seq_len - 1:  # 补充了eos符后长度超过了max_seq_len的样本舍去
            continue
        process_one_seq(in_seq_tokens, in_tokens, in_seqs, max_seq_len)
        process_one_seq(out_seq_tokens, out_tokens, out_seqs, max_seq_len)
    in_vocab, in_data = build_data(in_tokens, in_seqs)
    out_vocab, out_data = build_data(out_tokens, out_seqs)
    return in_vocab, out_vocab, data.TensorDataset(in_data, out_data)  # TensorDataset将输入的张量按行组合成样本


if __name__ == "__main__":
    max_seq_len = 7
    in_vocab, out_vocab, dataset = read_data(max_seq_len)
    print(dataset[0])
