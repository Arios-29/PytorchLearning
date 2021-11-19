from abc import ABC
from typing import Any

import torch
from torch.autograd import Function

"""
自定义Function
forward():Function的前向计算过程
backward():Function的反向求导过程
jvp():Function的前向求导过程
"""


class LinearFunction(Function):

    @staticmethod
    def forward(ctx: Any, *args: Any, **kwargs: Any) -> Any:
        x = args[0]
        w = args[1]
        b = args[2]
        ctx.save_for_backward(x, w, b)
        return x.mm(w)+b

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        pass



