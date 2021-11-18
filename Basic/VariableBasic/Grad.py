import torch
from torch.autograd import Variable

"""
pytorch只能用标量来求导
对x求导后, x.grad与x的形状是一致的
"""

# 标量对向量求导
# 本质是y=f(x1, x2,...,xn),对x=(x1, x2,...,xn)求导后得到一个向量,i位置对应y对xi的导数
vec_x = Variable(torch.rand(4), requires_grad=True)
w = Variable(torch.tensor([1.0, 1.0, 1.0, 1.0]), requires_grad=True)
y = (vec_x * w).sum()  # 本质是 y = w.t().mm(vec_x)
y.backward()
print(vec_x.grad)
print(w.grad)

# 标量对矩阵求导
# 本质还是y=f(x11, x12,...,xmn),对x求导后得到一个矩阵, ij位置对应y对xij的导数
matrix_x = Variable(torch.rand((4, 4)), requires_grad=True)
y = matrix_x ** 2
y = y.sum()
y.backward()
print(matrix_x.grad)

# backward()中的gradient参数
# z=f(y), y=g(x), 则z.backward()等价于y.backward(gradient=grad_y), grad_y为z对y求导, grad_y的形状与y一致
# grad(z, xij)=grad(z, y1)*grad(y1, xij)+...+grad(z, ym)*grad(ym, xij) 对应x.grad中ij位置
x = Variable(torch.rand((4, 4)), requires_grad=True)
y = x * 2.0
z = y.mean()
grad_y = Variable(torch.ones((4, 4))*0.0625)
y.backward(gradient=grad_y)
print(x.grad)
