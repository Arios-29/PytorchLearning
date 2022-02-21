import torch.utils.data


class PTBDataset(torch.utils.data.Dataset):
    def __init__(self, centers, contexts, negatives):
        """
        :param centers: centers[i]表示一个中心词
        :param contexts: contexts[i]表示centers[i]的n个背景词
        :param negatives: negatives[i]表示centers[i],contexts[i]的负样本
        """
        assert len(centers) == len(contexts) == len(negatives)
        self.centers = centers
        self.contexts = contexts
        self.negatives = negatives

    def __getitem__(self, index):
        return self.centers[index], self.contexts[index], self.negatives[index]

    def __len__(self):
        return len(self.centers)


def read_a_batch(batch):
    """
    :param batch: 一个长度为batch_size的列表,每个元素为(center,context,negative)
    :return: centers,context_negatives,masks,labels:每一行对应一个样本,形状为(batch_size, xx)
    """
    max_len = max(len(context) + len(negative) for _, context, negative in batch)
    centers, context_negatives, masks, labels = [], [], [], []
    for center, context, negative in batch:
        cur_len = len(context) + len(negative)
        centers.append(center)
        context_negative = context + negative + [0] * (max_len - cur_len)  # 长度补齐,填充0
        context_negatives.append(context_negative)
        mask = [1] * cur_len + [0] * (max_len - cur_len)  # 用于区分填充区段
        masks.append(mask)
        label = [1] * len(context) + [0] * (max_len - len(context))  # 用于区分context区段
        labels.append(label)
    return torch.tensor(centers).view(-1, 1), torch.tensor(context_negatives), torch.tensor(masks), torch.tensor(labels)

