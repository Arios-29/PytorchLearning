import torch
from torch.autograd import Variable

"""
对非叶子节点变量的梯度在backward完成后会清除掉
"""

# 非叶子节点变量y通过retain_grad()保留其梯度
x = Variable(torch.rand((1, 4)), requires_grad=True)
y = x ** 2
z = y.sum()
y.retain_grad()
z.backward()
print(y.grad)

# 使用torch.autograd.grad(z, y)得到对y的梯度,但是该梯度不会保留到y.grad中,也不会继续往后传播
x = Variable(torch.rand((1, 4)), requires_grad=True)
y = x ** 2
z = y.sum()
grad_y = torch.autograd.grad(z, y)
print(x.grad)  # x.grad为None说明只计算到y就没有往下求梯度了
# print(y.grad)  # y.grad为None说明此方法对y的梯度计算结果没有保存到y.grad中
print(grad_y)

# 使用hook获取对非叶子节点的梯度
# hook是一种函数,只有一个参数grad,不应该有返回值
x = Variable(torch.rand((1, 4)), requires_grad=True)
y = x ** 2
z = y.sum()


# 1. 定义一个hook函数
def print_grad(grad):
    print("hook执行")
    print(grad)


# 2. 将该hook注册到指定的变量上,这样每次梯度从该节点反向传播前会先传入hook函数
hook_handle = y.register_hook(print_grad)

# 3. backward过程中在y节点传播梯度前会先将梯度传入hook函数
z.backward()

# 4. 将hook函数从对应变量上移除
hook_handle.remove()