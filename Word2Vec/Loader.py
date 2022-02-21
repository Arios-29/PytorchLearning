import collections
import math
import random

import torch


def negative_sampling(contexts, sampling_weights, K):
    """
    :param contexts:
    :param sampling_weights: 各个词对应的权重
    :param K:
    :return: 负采样,all_negatives[i]表示正样本(Context(wi),wi)的所有负样本词
    """
    all_negatives, neg_candidates, i = [], [], 0
    population = list(range(len(sampling_weights)))  # population存有各个词的编号
    for context in contexts:
        negatives = []
        while len(negatives) < len(context) * K:  # 每一对中心词和背景词都需要选出K个负样本
            if i == len(neg_candidates):  # 还没选出足够的负样本就遍历完了负样本候选列表
                i = 0
                neg_candidates = random.choices(population, sampling_weights, k=int(1e5))
            neg = neg_candidates[i]
            i = i + 1
            if neg not in set(context):  # 负样本不能为正样本背景词
                negatives.append(neg)
        all_negatives.append(negatives)
    return all_negatives


class PTBLoader:
    def __init__(self):
        # 读取数据
        with open("data/ptb.train.txt", 'r') as file:
            lines = file.readlines()
            self.sentences = [sentence.split() for sentence in lines]

        # 建立词语索引,只保留出现次数>5的词
        counter = collections.Counter([word for sentence in self.sentences for word in sentence])  # word:频次

        def is_geq_5(x):
            return x[1] >= 5

        counter = dict(filter(is_geq_5, counter.items()))
        self.idx_to_word = [word for word, _ in counter.items()]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.idx_to_word)}
        self.sampling_weights = [counter[word] ** 0.75 for word in self.idx_to_word]

        # 将文本数据用词语索引表示
        self.dataset = [[self.word_to_idx[word] for word in sentence if word in self.word_to_idx] for sentence in
                        self.sentences]

        # 二次采样,去除一些无意义的高频词，比如"the"
        # 一个词以P(w)的概率丢弃,P(w)=max(1-sqrt(t/f(w)),0)
        # f(w)表示数据集中该词的数目与数据集词数的比值
        num_words = sum([len(sentence) for sentence in self.dataset])

        def discard(idx):
            return random.uniform(0, 1) < 1 - math.sqrt(1e-4 / counter[self.idx_to_word[idx]] * num_words)

        self.sub_sampled_dataset = [[word_idx for word_idx in sentence_idx if not discard(word_idx)] for sentence_idx in
                                    self.dataset]

    def get_centers_and_contexts(self, max_window_size):
        """
        :param max_window_size:
        :return: 中心词和相应的背景词
        """
        centers, contexts = [], []
        for sentence in self.sub_sampled_dataset:
            if len(sentence) < 2:
                continue
            centers += sentence
            for center_i in range(len(sentence)):
                window_size = random.randint(1, max_window_size)
                indices = list(range(max(0, center_i - window_size), min(len(sentence), center_i + 1 + window_size)))
                indices.remove(center_i)
                context = [sentence[indice] for indice in indices]
                contexts.append(context)
        return centers, contexts

    def get_sampling_weights(self):
        return self.sampling_weights

    def get_num_vocab(self):
        return len(self.idx_to_word)

    def get_similar_words(self, query_word, k, embed):
        W = embed.weight
        x = W[self.word_to_idx[query_word]]
        cos = torch.matmul(W, x) / (torch.sum(W * W, dim=1) * torch.sum(x * x, dim=1) + 1e-9).sqrt()
        _, topk = torch.topk(cos, k=k+1)
        topk = topk.numpy()
        for i in topk[1:]:
            print("cosine sim=%.3f: %s" % (cos[i], self.idx_to_word[i]))
