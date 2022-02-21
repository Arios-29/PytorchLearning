import time

import torch
from torch.utils.data import DataLoader

from Word2Vec.Loader import PTBLoader, negative_sampling
from Word2Vec.PTBDataset import PTBDataset, read_a_batch
from Word2Vec.SigmoidBinaryCrossEntropyLoss import SigmoidBinaryCrossEntropyLoss


def skip_gram(centers, context_negatives, embed_v, embed_u):
    """
    :param centers: 中心词,形状为(batch_size,1)
    :param context_negatives: 背景词,形状为(batch_size,max_len)
    :param embed_v: 将中心词转变为词向量的嵌入层
    :param embed_u: 将背景词转变为词向量的嵌入层
    :return: 形状为(batch_size, 1, max_len)
    """
    v = embed_v(centers)  # v的形状为(batch_size, 1, num_embeddings),即每一横切面对应一个中心词,该横切面只有一行,为该中心词的词向量
    u = embed_u(context_negatives)  # u的形状为(batch_size, max_len, num_embeddings),每一横切面对应一个中心词的各背景词+噪声词,横切面每一行对应一个词向量
    u = u.permute(0, 2, 1)  # permute(0,2,1)把1、2维度互换,此时每一横切面的一列对应一个词向量
    # bmm为小批量乘法运算, 对形状为(n,a,b)和(n,b,c)的张量进行bmm,得到(n,a,c)形状的张量,即每一层对应进行矩阵乘法
    # 将中心词与背景词+噪声词作内积运算
    return torch.bmm(v, u)


def train(center_embed, context_embed, lr, num_epochs, data_iter):
    net = torch.nn.Sequential(center_embed, context_embed)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss = SigmoidBinaryCrossEntropyLoss()
    for epoch in range(num_epochs):
        start_time = time.time()
        loss_sum = 0.0
        n = 0  # 统计批次
        for batch in data_iter:
            centers, context_negatives, masks, labels = batch[0], batch[1], batch[2], batch[3]
            pred = skip_gram(centers, context_negatives, center_embed, context_embed)
            pred = pred.view(labels.size())  # (batch_size, max_len)
            batch_loss = loss(pred, labels, masks) * masks.shape[1] / masks.float().sum(
                dim=1)  # (batch_size),每一项对应一个样本的损失
            batch_mean_loss = batch_loss.mean()  # 这批样本的平均损失
            optimizer.zero_grad()
            batch_mean_loss.backward()
            optimizer.step()
            loss_sum += batch_mean_loss.item()
            n += 1
        print("epoch: %d, loss: %.2f, time: %.2fs" % (epoch + 1, loss_sum / n, time.time() - start_time))


if __name__ == "__main__":
    loader = PTBLoader()
    centers, contexts = loader.get_centers_and_contexts(5)
    negatives = negative_sampling(contexts, loader.get_sampling_weights(), 5)
    ptb_dataset = PTBDataset(centers, contexts, negatives)
    data_iter = DataLoader(ptb_dataset, batch_size=512, shuffle=True, collate_fn=read_a_batch, num_workers=4)

    embed_size = 100
    num_vocab = loader.get_num_vocab()
    center_embed = torch.nn.Embedding(num_embeddings=num_vocab, embedding_dim=embed_size)
    context_embed = torch.nn.Embedding(num_embeddings=num_vocab, embedding_dim=embed_size)
    train(center_embed, context_embed, lr=0.01, num_epochs=10, data_iter=data_iter)
    loader.get_similar_words("chip", 4, center_embed)
