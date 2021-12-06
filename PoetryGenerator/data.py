import os.path

import numpy as np


class Config(object):
    data_path = ''  # 诗歌数据的存放路径
    author = None  # 只学习某位作者的诗歌
    constrain = None  # 长度限制
    category = 'poet.tang'  # 类别,唐诗还是宋词

    lr = 1e-3  # 学习率
    use_gpu = True  # 是否使用GPU加速
    epoch = 20  # 一个epoch训练一次数据集的全部数据
    batch_size = 128  # 批量大小,一个批量对应一个Iteration

    max_len = 125  # 超过max_len长度之后的字被舍弃,小于这个长度的在前面补空格
    plot_every = 20  # 每20个batch可视化一次
    use_env = True  # 是否使用matplotlib
    env = 'poetry'  # matplotlib env
    max_gen_len = 20  # 生成诗歌的最长长度
    debug_file = '/tmp/debug'
    model_path = None  # 预训练模型路径
    model_prefix = None  # 模型保存路径

    # 生成诗歌的相关配置
    prefix_words = ''
    start_words = ''
    acrostic = False  # 是否是藏头诗


def get_data(opt: Config):
    """
    :param opt: 配置类Config对象
    :return word2ix: dict,每个字对应的序号
    :return ix2word: dict,每个序号对应的字
    :return data: numpy数组,每一行是一首诗,一首诗用字对应的序号表示
    """
    if os.path.exists(opt.data_path):
        all_data = np.load(opt.data_path, allow_pickle=True)
        data = all_data['data']
        word2ix = all_data['word2ix'].item()
        ix2word = all_data['ix2word'].item()
        return data, word2ix, ix2word
    else:
        print("输入路径不存在")

