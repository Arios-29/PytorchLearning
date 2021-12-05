from torch import nn

"""
双向RNN的参数是对应单向RNN的两倍。
双向RNN的隐状态元素数目是单向RNN的两倍
通过指定参数bidirectional=True定义一个双向RNN
"""