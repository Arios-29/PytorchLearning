import torch
from torch.autograd import Variable

"""
pytorch只能用标量来求导
对x求导后, x.grad与x的形状是一致的
"""

# 标量对向量求导
# 本质是y=f(x1, x2,...,xn),对x=(x1, x2,...,xn)求到后得到一个向量,i位置对应y对xi的导数
vec_x = Variable(torch.rand(4), requires_grad=True)
w = Variable(torch.tensor([1.0, 1.0, 1.0, 1.0]), requires_grad=True)
y = (vec_x * w).sum()  # 本质是 y = w.t().mm(vec_x)
y.backward()
print(vec_x.grad)
print(w.grad)

# 向量对向量求导
vec_x.grad.data.zero_()
z = vec_x * w
z.backward(torch.tensor([1, 2, 3, 4]))  # 1*f1' + 2*f2' + 3*f3' + 4*f4' , fi'表示z[i-1]对x的导数
print(vec_x.grad)