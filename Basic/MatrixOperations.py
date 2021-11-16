import torch

"""
一些矩阵操作, 这些操作只会作用在矩阵维度上
"""

size = torch.Size((4, 4))
matrix = torch.rand(size)
print(matrix)

# matrix.trace()求矩阵的迹
print(matrix.trace())

# matrix.diag()求对角线元素
print(matrix.diag())

# matrix.mm(mat2)为矩阵乘法,要求两个矩阵可以作乘法运算
print(matrix.mm(matrix))

# matrix.inverse()求矩阵的逆
print(matrix.inverse())

# matrix.svd()对矩阵进行奇异值分解,对方阵特征值分解
# 返回(U,S,V)三个张量构成的元组
# matrix = U x diag(S) x V^T
print(matrix.svd())

# matrix.t()求矩阵的转置,但是会导致存储空间不连续
# tensor.contiguous()将张量的存储空间连续化
print(matrix.t().contiguous())
