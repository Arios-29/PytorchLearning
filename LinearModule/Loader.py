import sys
from typing import List

import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt
from torchvision import transforms


class MnistData:
    def __init__(self):
        """
        self.mnist_train为训练样本集, self.mnist_test为测试样本集
        每一个样本为一张28x28的图片, 图片内容为生活用品,一共10类
        self.mnist_train[i]为(i号图片的张量, 标签)
        标签=k, 对应类别self.text_labels[k]
        """
        self.mnist_train = torchvision.datasets.FashionMNIST(root='../data/Datasets/FashionMNIST', train=True,
                                                             download=False,
                                                             transform=transforms.ToTensor())
        # transforms.ToTensor()将图片转为张量,形状为(1, 长度像素数, 宽度像素数)
        self.mnist_test = torchvision.datasets.FashionMNIST(root='../data/Datasets/FashionMNIST', train=False,
                                                            download=False,
                                                            transform=transforms.ToTensor())
        self.text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                            'sandal', 'shirt', 'sneaker', 'bag', 'ankle', 'boot']

    def get_train_data_nums(self):
        """
        :return: 返回训练集的样本数目
        """
        return len(self.mnist_train)

    def get_test_data_nums(self):
        """
        :return: 返回测试集的样本数目
        """
        return len(self.mnist_test)

    def get_train_iter(self, batch_size=100):
        """
        :param batch_size: 训练样本迭代器每次读取一批样本的批量大小
        :return: 训练样本迭代器
        """
        if sys.platform.startswith('win'):
            num_workers = 0
        else:
            num_workers = 4  # 读取数据的线程数
        return Data.DataLoader(self.mnist_train, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    def get_test_iter(self, batch_size=100):
        """
        :param batch_size: 测试样本迭代器每次读取一批样本的批量大小
        :return: 测试样本迭代器
        """
        if sys.platform.startswith('win'):
            num_workers = 0
        else:
            num_workers = 4  # 读取数据的线程数
        return Data.DataLoader(self.mnist_test, shuffle=True, batch_size=batch_size, num_workers=num_workers)

    def get_labels(self, categories: List):
        labels = []
        for category in categories:
            labels.append(self.text_labels[category])
        return labels


def paint_images(images: List, labels: List):
    """
     :param images: 图片张量列表
     :param labels: 图片标签名列表
     """
    _, axes = plt.subplots(1, len(images), figsize=(12, 12))
    for ax, image, label in zip(axes, images, labels):
        ax.imshow(image.view(28, 28).numpy())
        ax.set_title(label)
        ax.axes.get_xaxis().set_visible(False)
        ax.axes.get_yaxis().set_visible(False)
    plt.show()

