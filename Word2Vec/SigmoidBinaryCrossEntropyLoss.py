import torch.nn
from torch.nn.functional import binary_cross_entropy_with_logits


class SigmoidBinaryCrossEntropyLoss(torch.nn.Module):
    def __init__(self):
        super(SigmoidBinaryCrossEntropyLoss, self).__init__()

    def forward(self, inputs, targets, mask=None):
        """
        :param inputs: 形状为(batch_size, num_outputs)
        :param targets: 形状为(batch_size, num_outputs)
        :param mask: 对每一项的权重
        :return: 形状为(batch_size),每一项为一个样本的平均加权交叉熵损失
        """
        inputs = inputs.float()
        targets = targets.float()
        mask = mask.float()
        # inputs每一行对应一个样本的输出层,将输出sigmoid,与对应的target求交叉熵,在乘上对应的权重
        # 输出形状为(batch_size, num_outputs),每一行对应一个样本输出层各项的加权交叉熵损失
        # mask设置为0的项是需要丢弃的
        batch_loss = binary_cross_entropy_with_logits(inputs, targets, reduction="none", weight=mask)
        return batch_loss.mean(dim=1)  # 按行求均值,形状为(batch_size),每一项为一个样本的平均加权交叉熵损失,注意这里包含了需要丢弃的项


if __name__ == "__main__":
    pred = torch.tensor([[1.5, 0.3, -1, 2], [1.1, -0.6, 2.2, 0.4]])
    label = torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0]])
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]])
    loss = SigmoidBinaryCrossEntropyLoss()
    ret = loss(pred, label, mask) * mask.shape[1] / mask.float().sum(dim=1)
    print(ret)
