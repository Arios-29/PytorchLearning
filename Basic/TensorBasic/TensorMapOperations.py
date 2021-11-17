import torch

"""
对Tensor的元素逐一进行的操作
"""

tensor_1 = torch.tensor([[-1, -2, -3], [-4, -5, -6]])

# tensor.abs()将Tensor每个元素取绝对值
tensor_1 = tensor_1.abs()
print(tensor_1)

# tensor.sqrt()对Tensor每个元素取平方根
tensor_2 = tensor_1.sqrt()
print(tensor_2)

# tensor逐元素对标量a进行加减乘除、取模、幂运算都实现了运算符重载
# tensor与tensor间逐元素加减乘除、取模、幂运算也实现了运算符重载。要求两个tensor可以通过广播机制变成同样的形状。
print(tensor_1 + tensor_2)
print(tensor_1 - tensor_2)
print(tensor_1 * tensor_2)
print(tensor_1 / tensor_2)
print(tensor_1 % tensor_2)
print(tensor_1 ** tensor_2)

# tensor逐元素对标量a进行大小比较的操作都实现了运算符重载
# tensor对tensor间逐元素进行大小比较的操作都实现了运算符重载。要求两个tensor可以通过广播机制变成同样的形状
# 大小判断产生一个BoolTensor,满足判定的位置为True，否则为False
print((tensor_1 > tensor_2))

# tensor.clamp(min, max)对tensor进行逐元素截断
# 如果元素x小于min则将其改为min, min默认为无穷小
# 如果元素x大于max则将其改为max, max默认为无穷大
print(tensor_1.clamp(min=2, max=5))

# 若f为激活函数,则tensor.f()逐一对每一个元素作输入求激活函数输出值
print(tensor_1.sigmoid())
print(tensor_1.tanh())
