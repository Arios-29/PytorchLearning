import torch
from torch import nn

"""
nn.Embedding嵌入层的使用
"""

# 初始的序列,批量大小为3
original_seq_batch = ['I am a boy.', 'How are you?', 'I am very lucky.']

# 将序列标准化(程序过程略),即大写转小写,将标点与词分离,将句子拆分成词表示(标点也视作词)
norm_seq_batch = [['i', 'am', 'a', 'boy', '.'], ['how', 'are', 'you', '?'], ['i', 'am', 'very', 'lucky', '.']]

# 建立词索引word_to_index(程序过程略)
# <start>表示序列的开始,<EOS>表示序列的结束,<PAD>用于填充,使得各序列长度一致
word_to_index = {'<start>': 0, 'i': 1, 'am': 2, 'a': 3, 'boy': 4, 'how': 5, 'are': 6, 'you': 7,
                 'very': 8, 'lucky': 9, '.': 10, '?': 11, '<EOS>': 12, '<PAD>': 13}

# 用索引表示序列,序列长度不足用<PAD>填充
input_seq_batch = torch.LongTensor([[0, 1, 2, 3, 4, 10, 12],
                                    [0, 5, 6, 7, 11, 13, 12],
                                    [0, 1, 2, 8, 9, 10, 12]])

# 定义一个嵌入层
num_embeddings = len(word_to_index)  # 语料库词的数目,词的索引下标不能超过num_embeddings
embedding_dim = 5  # 将词映射到5维空间,每个词用一个5维的向量表示
embed = nn.Embedding(num_embeddings, embedding_dim)

# nn.embedding的输入形状为(批量大小,序列长度),即每一行对应一个序列
# nn.embedding的输出形状为(批量大小,序列长度,词向量维度),即每一横平面对应一个序列,横平面每一行是一个词
output_seq_batch = embed(input_seq_batch)

# 将嵌入层的输出作为后续层的输入,最后反向训练时嵌入层也会得到更准确的词向量表示
