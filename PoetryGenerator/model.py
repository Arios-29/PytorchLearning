from torch import nn


class PoetryModel(nn.Module):
    def __init__(self, all_words_num, xt_size, ht_size):
        """
        :param all_words_num: 训练集中字的种类数
        :param xt_size: 将字转化成的向量维度
        :param ht_size: lstm各时刻的输出维度
        """
        super(PoetryModel, self).__init__()
        self.ht_size = ht_size
        self.embedding_layer = nn.Embedding(all_words_num, xt_size)  # nn.Embedding的输出形状为(批量大小,序列长度,词的维度)
        self.lstm = nn.LSTM(xt_size, ht_size, num_layers=2, batch_first=True)  # batch_first=True配合嵌入层的输出
        self.linear = nn.Linear(ht_size, all_words_num)  # 将lstm的输出转为为字对应的序号

    def forward(self, a_batch_of_seq, initial_state=None):
        """
        :param a_batch_of_seq: 形状为(序列长度,批量大小),即每一列为一个序列
        :param initial_state: 为(outer_state, inner_state)的元组
        :return:
        """
        a_batch_of_seq = a_batch_of_seq.t().contiguous()  # 将输入转为(批量大小,序列长度)的形状,对应嵌入层的输入
        batch_size, seq_len = a_batch_of_seq.size()
        seq_batch_embedded = self.embedding_layer(a_batch_of_seq)
        if initial_state is not None:
            output, final_outer_state, final_inner_state = self.lstm(seq_batch_embedded,
                                                                     initial_state[0], initial_state[1])
        else:
            output, final_outer_state, final_inner_state = self.lstm(seq_batch_embedded)
        # output的形状为(序列长度,批量大小,ht_size),需要转化成(序列长度*批量大小,ht_size)作为全连接层的输入
        output = output.view(batch_size*seq_len, self.ht_size)
        output = self.linear(output)
        return output, final_outer_state


