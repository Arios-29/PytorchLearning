from abc import ABC
from typing import Any

import torch
from torch.autograd import Function, Variable

"""
自定义Function
forward():Function的前向计算过程
backward():Function的反向求导过程
jvp():Function的前向求导过程
"""


# 定义一个线性回归方程
class LinearFunction(Function):

    @staticmethod
    def forward(ctx, w, x, b):
        ctx.save_for_backward(w, x)
        y = w * x
        y = y.sum() + b
        return y

    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def backward(ctx, grad_output):
        w, x = ctx.saved_tensors
        grad_w = x * grad_output
        grad_x = w * grad_output
        grad_b = grad_output
        return grad_w, grad_x, grad_b


w = Variable(torch.rand(1), requires_grad=True)
x = Variable(torch.rand(1), requires_grad=True)
b = Variable(torch.rand(1), requires_grad=True)
y = LinearFunction.apply(w, x, b)
y.backward()
print(w)
print(x.grad)
print(x)
print(w.grad)
print(b.grad)


# 定义一个仿射变换
class LinearLayer(Function):
    @staticmethod
    def jvp(ctx: Any, *grad_inputs: Any) -> Any:
        pass

    @staticmethod
    def backward(ctx, grad_output):
        w, x = ctx.saved_tensors
        grad_x = w.t().mm(grad_output)  # 用矩阵乘法还是点乘法写程序时要具体判断,但数学推导中都是矩阵乘法
        grad_b = grad_output
        grad_w = x.t() * grad_output
        return grad_w, grad_x, grad_b

    @staticmethod
    def forward(ctx, w, x, b):
        """
           w是(a,b)的矩阵
           x是(b,1)的矩阵(向量)
           b是(a,1)的矩阵(向量)
           """
        y = w.mm(x) + b
        ctx.save_for_backward(w, x)
        return y


w = Variable(torch.rand((3, 4)), requires_grad=True)
x = Variable(torch.rand((4, 1)), requires_grad=True)
b = Variable(torch.rand((3, 1)), requires_grad=True)
y = LinearLayer.apply(w, x, b)
z = y.sum()
z.backward()
print(w.grad)
print(x.grad)
print(b.grad)
