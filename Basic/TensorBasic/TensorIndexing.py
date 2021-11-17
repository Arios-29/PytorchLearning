import torch

"""
普通索引:得到的还是张量,该张量的storage在原张量的storage中
tensor[k1][k2]...[kn]等价于tensor[k1, k2, ... , kn]
ki指定元素对应维度下的下标
ki的写法:
    (1) start:end:step 
        step默认为1, end默认为最后一个下标+1, start默认为0
    (2) 标量,指定某一个下标
"""
tensor_1 = torch.rand((3, 4))

# 访问某一个元素
a = tensor_1[1][2]
b = tensor_1[1, 2]

# 访问某一行元素
row_1 = tensor_1[1]
row_2 = tensor_1[1, :]
row_3 = tensor_1[1][:]

# 访问某一列元素
col_1 = tensor_1[:, 1]
col_2 = tensor_1[:][1]

"""
高级索引:得到的还是张量,一般storage不在原张量storage中
tensor[k1, k2, ... , kn]
ki写成[a, b, c...x]的形式
i<j,如果ki与kj长度一致，则(a,..., b), a与b对应,否则(a,..., b)中的(a, b)为ki与kj的笛卡尔积元素
"""
tensor_2 = torch.rand((3, 3, 3))

# tensor_2[1][1][2]和tensor_2[2][2][0]
sub_1 = tensor_2[[1, 2], [1, 2], [2, 0]]

# tensor_2[1][0][2]和tensor_2[2][0][0]
sub_2 = tensor_2[[1, 2], [0], [2, 0]]

# tensor_a[tensor_b],tensor_b是一个和tensor_a同样形状的类型为BoolTensor的张量,输出tensor_b元素为True对应位置的tensor_a的值构成的向量
tensor_3 = torch.rand((4, 4))
tensor_4 = torch.rand((4, 4)).type(torch.BoolTensor)
print(tensor_3[tensor_4])
