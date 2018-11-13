---
title: 基于pytorch的交通标志识别
date: 2018-11-11 11:57:49
categories: Pytorch框架
tags: 交通标志识别
---
本文是在https://www.jianshu.com/p/d8feaddc7bdf文章的基础上用Pytorch实现的

话不多说，直接上代码，具体的可以看代码中的解释
### 代码实现

```py
import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.transform
import numpy as np

# 定义是否使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

'''
使得我们能够手动输入命令行参数，就是让风格变得和Linux命令行差不多
argparse是python的一个包，用来解析输入的参数
如：
    python mnist.py --outf model  
    （意思是将训练的模型保存到model文件夹下，当然，你也可以不加参数，那样的话代码最后一行
      torch.save()就需要注释掉了）

    python mnist.py --net model/net_005.pth
    （意思是加载之前训练好的网络模型，前提是训练使用的网络和测试使用的网络是同一个网络模型，保证权重参数矩阵相等）
'''
parser = argparse.ArgumentParser()

parser.add_argument('--outf', default='./model/', help='folder to output images and model checkpoints')  # 模型保存路径
parser.add_argument('--net', default='./model/net.pth', help="path to netG (to continue training)")  # 模型加载路径
opt = parser.parse_args()  # 解析得到你在路径中输入的参数，比如 --outf 后的"model"或者 --net 后的"model/net_005.pth"，是作为字符串形式保存的

# Load training and testing datasets.
ROOT_PATH = "./traffic"
train_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Training")
test_data_dir = os.path.join(ROOT_PATH, "datasets/BelgiumTS/Testing")


'''
定义LeNet神经网络，进一步的理解可查看Pytorch入门，里面很详细，代码本质上是一样的，这里做了一些封装
'''
class LeNet(nn.Module):
    '''
    该类继承了torch.nn.Modul类
    构建LeNet神经网络模型
    '''
    def __init__(self):
        super(LeNet, self).__init__()  # 这一个是python中的调用父类LeNet的方法，因为LeNet继承了nn.Module，如果不加这一句，无法使用导入的torch.nn中的方法，这涉及到python的类继承问题，你暂时不用深究

        # 第一层神经网络，包括卷积层、线性激活函数、池化层
        self.conv1 = nn.Sequential(     # input_size=(1*28*28)：输入层图片的输入尺寸，我看了那个文档，发现不需要天，会自动适配维度
            nn.Conv2d(3, 32, 5, 1, 2),   # padding=2保证输入输出尺寸相同：采用的是两个像素点进行填充，用尺寸为5的卷积核，保证了输入和输出尺寸的相同
            nn.ReLU(),                  # input_size=(6*28*28)：同上，其中的6是卷积后得到的通道个数，或者叫特征个数，进行ReLu激活
            nn.MaxPool2d(kernel_size=2, stride=2), # output_size=(6*14*14)：经过池化层后的输出
        )

        # 第二层神经网络，包括卷积层、线性激活函数、池化层
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 5),  # input_size=(6*14*14)：  经过上一层池化层后的输出,作为第二层卷积层的输入，不采用填充方式进行卷积
            nn.ReLU(),            # input_size=(16*10*10)： 对卷积神经网络的输出进行ReLu激活
            nn.MaxPool2d(2, 2)    # output_size=(16*5*5)：  池化层后的输出结果
        )

        # 全连接层(将神经网络的神经元的多维输出转化为一维)
        self.fc1 = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),  # 进行线性变换
            nn.ReLU()                    # 进行ReLu激活
        )

        # 输出层(将全连接层的一维输出进行处理)
        self.fc2 = nn.Sequential(
            nn.Linear(128, 84),
            nn.ReLU()
        )

        # 将输出层的数据进行分类(输出预测值)
        self.fc3 = nn.Linear(84, 62)

    # 定义前向传播过程，输入为x
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        # nn.Linear()的输入输出都是维度为一的值，所以要把多维度的tensor展平成一维
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


# 超参数设置
EPOCH = 20   # 遍历数据集次数(训练模型的轮数)
BATCH_SIZE = 3     # 批处理尺寸(batch_size)：关于为何进行批处理，文档中有不错的介绍
LR = 0.001        # 学习率：模型训练过程中每次优化的幅度


# 定义数据预处理方式(将输入的类似numpy中arrary形式的数据转化为pytorch中的张量（tensor）)
transform = transforms.ToTensor()
# transform = torch.FloatTensor

# 定义训练数据集(此处是加载MNIST手写数据集)
trainset = tv.datasets.Traffic(
    root=train_data_dir, # 如果从本地加载数据集，对应的加载路径
    train=True,     # 训练模型
    download=True,  # 是否从网络下载训练数据集
    transform=transform  # 数据的转换形式
)

# 定义训练批处理数据
trainloader = torch.utils.data.DataLoader(
    trainset,                # 加载测试集
    batch_size=BATCH_SIZE,   # 最小批处理尺寸
    shuffle=True,            # 标识进行数据迭代时候将数据打乱
)

# 定义测试数据集
testset = tv.datasets.Traffic(
    root=test_data_dir, # 如果从本地加载数据集，对应的加载路径
    train=True,     # 训练模型
    download=True,  # 是否从网络下载训练数据集
    transform=transform  # 数据的转换形式
)

# 定义测试批处理数据
testloader = torch.utils.data.DataLoader(
    testset,                 # 加载测试集
    batch_size=BATCH_SIZE,   # 最小批处理尺寸
    shuffle=False,           # 标识进行数据迭代时候不将数据打乱
)


def model_train():
    # 定义损失函数loss function 和优化方式（采用SGD）
    net = LeNet().to(device)
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数，通常用于多分类问题上
    optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)  # 优化函数
    for epoch in range(EPOCH):
        sum_loss = 0.0

        # 数据读取（采用python的枚举方法获得标签和数据，这一部分可能和numpy相关）
        for i, data in enumerate(trainloader):
            inputs, labels = data
            # labels = [torch.LongTensor(label) for label in labels]
            # 将输入数据和标签放入构建的图中 注：图的概念可在pytorch入门中查
            inputs, labels = inputs.to(device), labels.to(device)

            # 梯度清零
            optimizer.zero_grad()

            # forward + backward  注: 这一部分是训练神经网络的核心
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward() # 反向自动求导
            optimizer.step() # 进行优化

            # 每训练100个batch打印一次平均loss
            sum_loss += loss.item()
            if i % 48 == 0:
                print('[%d, %d] loss: %.03f'
                      % (epoch + 1, i + 1, sum_loss / 100))
                sum_loss = 0.0
        # 每跑完一次epoch测试一下准确率
        with torch.no_grad():
            correct = 0
            total = 0
            # for i, data in enumerate(testloader):
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                # 取得分最高的那个类
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum()
            print('第%d个epoch的识别准确率为：%d%%' % (epoch + 1, (100 * correct / total)))
    torch.save(net.state_dict(), '%s/net_%03d.pth' % (opt.outf, epoch + 1))


# 训练
if __name__ == "__main__":
    model_train()

```

### 主要问题——数据读取
PyTorch中数据读取的一个重要接口是torch.utils.data.DataLoader，该接口定义在dataloader.py脚本中，只要是用PyTorch来训练模型基本都会用到该接口，为了满足pytorch的数据读取要求，写了一个tv.datasets.Traffic的读取文件，是基于mnist数据集的读取进行编写的：

```py
from __future__ import print_function
import torch.utils.data as data
from PIL import Image
import os
import os.path
import errno
import numpy as np
import torch
import codecs
import skimage.data
import skimage.transform

def load_data(data_dir, train=True):
    """Loads a data set and returns two lists:

    images: a list of Numpy arrays, each representing an image.
    labels: a list of numbers that represent the images labels.
    仅仅只是加载图片到数组，并没有对图片进行缩放比例
    """
    if train:
        # Get all subdirectories of data_dir. Each represents a label.
        directories = [d for d in os.listdir(data_dir)
                    if os.path.isdir(os.path.join(data_dir, d))]
        # Loop through the label directories and collect the data in
        # two lists, labels and images.
        labels = []
        images = []
        for d in directories:
            label_dir = os.path.join(data_dir, d)
            file_names = [os.path.join(label_dir, f)
                        for f in os.listdir(label_dir) if f.endswith(".ppm")]
            # For each label, load it's images and add them to the images list.
            # And add the label number (i.e. directory name) to the labels list.
            for f in file_names:
                images.append(skimage.data.imread(f))
                labels.append(int(d))  # 为每一个图片加上标签
    images28 = [skimage.transform.resize(image, (28, 28)) for image in images]
    labels_a = np.asarray(labels)
    images_a = np.asarray(images28)
    return images_a, labels_a


class Traffic(data.Dataset):
    '''
    Traffic Dataset.
    '''

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        self.root = os.path.expanduser(root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set

        if self.train:
            self.train_data, self.train_labels = load_data(root,train)
        else:
            self.train_data, self.train_labels = load_data(root,train)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, target = self.train_data[index], self.train_labels[index]
        else:
            img, target = self.test_data[index], self.test_labels[index]

        # doing this so that it is consistent with all other datasets
        img = img.astype(np.float32)
        target = torch.LongTensor([target])[0]

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

```

**注意:** 在返回标签的时候，由于数据格式的问题，需要将标签放入一个list中，之后再转换为LongTensor，并取其第一个数据。