import torch

"""
Tensor的归并操作可以是对全局的数据，也可以通过指定参数dim对某一维度进行归并。
如果指定dim=i,则表示ki后面的维度进行归并
对行归并,就是顺应列(竖着)的方向归并
对列归并,就是顺应行(横着)的方向归并
归并后对应维度大小变为1
指定参数keepdim=True保留被归并的维度
"""

tensor_1 = torch.rand((4, 4))

# tensor.mean()求均值
print(tensor_1.mean())
print(tensor_1.mean(dim=0))
print(tensor_1.mean(dim=1))

# tensor.sum()求和
print(tensor_1.sum())

# tensor.median()求中位数
print(tensor_1.median())

# tensor.var()求方差
print(tensor_1.var())

# tensor.std()求标准差
print(tensor_1.std())

# tensor.norm(p)求Lp范数
# 只能对FloatTensor求范数
vector = torch.arange(1, 6).float()  # tensor.float()等价于tensor.type(torch.FloatTensor)
print(vector.norm(1))
print(vector.norm(2))

# tensor.dist(other,p)计算tensor和other的Lp距离
vector_1 = torch.rand(5)
vector_2 = torch.rand(5)
print(vector_1.dist(vector_2))
