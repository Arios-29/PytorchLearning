import random
from zipfile import ZipFile

import torch

"""
1.读取文本
2.为字符编号
3.用编号表示文本

随机取样:按步长为样本编号,样本标号*步长=样本首字符编号的索引
邻接取样:将用字符编号表示的文本转化成样本数*x的矩阵,按一定长度竖切得到一批样本
"""


class JayLyrics:
    def __init__(self):
        with ZipFile("../data/jaychou_lyrics.txt.zip") as zip_file:
            with zip_file.open("jaychou_lyrics.txt") as file:
                self.lyrics_chars = file.read().decode("utf-8")
        self.lyrics_chars.replace('\n', ' ').replace('\r', ' ')
        self.lyrics_chars = self.lyrics_chars[0:10000]
        self.index_to_char = list(set(self.lyrics_chars))
        self.char_to_index = dict([(char, index) for index, char in enumerate(self.index_to_char)])
        self.lyrics_indices = [self.char_to_index[char] for char in self.lyrics_chars]

    def get_vocab_size(self):
        return len(self.index_to_char)

    def get_index_to_char(self):
        return self.index_to_char

    def get_char_to_index(self):
        return self.char_to_index

    def get_an_example(self, pos, num_steps):
        return self.lyrics_indices[pos:pos + num_steps]

    def random_data_iter(self, batch_size, num_steps):  # 随机取样
        num_examples = (len(self.lyrics_indices) - 1 )// num_steps  # 最后一个字符不能作输入
        num_epochs = num_examples // batch_size
        example_indices = list(range(num_examples))  # 每一个样本一个编号
        random.shuffle(example_indices)  # 将样本编号打乱
        for epoch in range(num_epochs):
            start_indices_index = epoch * batch_size  # 本epoch要抽取的样本编号的开始索引
            batch_example_indices = example_indices[start_indices_index: start_indices_index + batch_size]  # 一批样本的编号
            X = [self.get_an_example(an_example_indice * num_steps, num_steps) for an_example_indice in
                 batch_example_indices]
            # 样本的编号*num_steps为该样本的首字符编号的索引
            Y = [self.get_an_example(an_example_indice * num_steps + 1, num_steps) for an_example_indice in
                 batch_example_indices]
            yield torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)  # (batch_size, num_steps)

    def consecutive_data_iter(self, batch_size, num_steps):  # 相邻取样,相邻批量对应位置的样本是有序的
        lyrics_indices_torch = torch.tensor(self.lyrics_indices, dtype=torch.float32)
        num_chars = len(lyrics_indices_torch)
        batch_len = num_chars // batch_size  # batch_size表示batch的样本数,batch_len=batch的数目*num_steps
        lyrics_indices_torch = lyrics_indices_torch.view(batch_size, batch_len)  # 每一行表示一个批量对应行的样本取样来源
        num_epochs = (batch_len - 1) // num_steps
        for epoch in range(num_epochs):
            start_indices_index = epoch * num_steps
            # 竖切出一批样本,因此每一行对应的相邻样本都是有序的
            # X每一行为一个样本, Y每一行为对应的标签
            X = lyrics_indices_torch[:, start_indices_index:start_indices_index + num_steps]
            Y = lyrics_indices_torch[:, start_indices_index + 1:start_indices_index + 1 + num_steps]
            yield X, Y  # (batch_size, num_steps)


if __name__ == "__main__":
    loader = JayLyrics()
    for X, Y in loader.consecutive_data_iter(10, 5):
        print(X.size(), Y.size())
