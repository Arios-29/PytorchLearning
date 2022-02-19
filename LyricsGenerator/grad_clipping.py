import torch

"""
裁剪梯度可以防止梯度爆炸
g = min(theta/||g||,1)*g
"""


def grad_clipping(params, theta):
    norm = torch.tensor([0.0])
    for param in params:
        norm += (param.grad.data ** 2).sum()
    norm = norm.sqrt().item()
    if norm > theta:
        for param in params:
            param.grad.data *= (theta / norm)
