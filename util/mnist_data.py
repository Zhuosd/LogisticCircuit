import gzip
import os
import torchvision
import numpy as np

from util.DataSet import DataSet, DataSets


def _read32(bytestream):
    dt = np.dtype(np.uint32).newbyteorder('>')
    return np.frombuffer(bytestream.read(4), dtype=dt)[0]


def extract_images_and_labels(image_file, label_file, percentage=1.0):
    """Extract the images into two 4D uint8 numpy array [index, y, x, depth]: positive and negative images."""
    print('Extracting', image_file, label_file)
    with gzip.open(image_file) as image_bytestream, gzip.open(label_file) as label_bytestream:
        magic = _read32(image_bytestream)
        if magic != 2051:
            raise ValueError(
                'Invalid magic number %d in image file: %s' %
                (magic, image_file))
        magic = _read32(label_bytestream)
        if magic != 2049:
            raise ValueError(
                'Invalid magic number %d in label file: %s' %
                (magic, label_file))
        num_images = _read32(image_bytestream)
        rows = _read32(image_bytestream)
        cols = _read32(image_bytestream)
        num_labels = _read32(label_bytestream)
        if num_images != num_labels:
            raise ValueError(
                'Num images does not match num labels. Image file : %s; label file: %s' %
                (image_file, label_file))
        images = []
        labels = []
        num_images = int(num_images * percentage)
        for _ in range(num_images):
            image_buf = image_bytestream.read(rows * cols)
            image = np.frombuffer(image_buf, dtype=np.uint8)
            image = np.multiply(image.astype(np.float32), 1.0 / 255.0)
            image[np.where(image == 0.0)[0]] = 1e-5
            image[np.where(image == 1.0)[0]] -= 1e-5
            label = np.frombuffer(label_bytestream.read(1), dtype=np.uint8)
            images.append(image)
            labels.append(label)
        images = np.array(images, dtype=np.float32)
        labels = np.array(labels, dtype=np.int32).squeeze()
        return images, labels


def crop_augment(images, target_side_length=26):
    images = np.reshape(images, (-1, 28, 28))
    augmented_images_shape = list(images.shape)
    augmented_images_shape[0] *= 2
    augmented_images = np.zeros(shape=augmented_images_shape, dtype=np.float32) + 1e-5

    diff = (28 - target_side_length) // 2
    for i in range(len(images)):
        images_center = images[i][diff:-diff, diff:-diff]
        augmented_images[2*i] = images[i]
        choice = np.random.random()
        if choice < 0.25:
            augmented_images[2*i+1][:target_side_length, :target_side_length] = images_center
        elif choice < 0.5:
            augmented_images[2*i+1][:target_side_length, -target_side_length:] = images_center
        elif choice < 0.75:
            augmented_images[2*i+1][-target_side_length:, :target_side_length] = images_center
        else:
            augmented_images[2*i+1][-target_side_length:, -target_side_length:] = images_center

    augmented_images = np.reshape(augmented_images, (-1, 784))
    return augmented_images


####  +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++#####  个人测试部分

import numpy as np
import torch   # pytorch机器学习开源框架
import torch.nn as nn
from torchvision import transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

class Net(nn.Module):
    # 定义Net的初始化函数，这个函数定义了该神经网络的基本结构
    def __init__(self):
        # 复制并使用Net的父类的初始化方法，即先运行nn.Module的初始化函数
        super(Net, self).__init__()
        # 定义fc1（fullconnect）全连接函数1为线性函数：y = Wx + b，并将28*28个节点连接到300个节点上。
        self.fc1 = nn.Linear(28*28, 300)
        # 定义fc2（fullconnect）全连接函数2为线性函数：y = Wx + b，并将300个节点连接到100个节点上。
        self.fc2 = nn.Linear(300, 100)
        # 定义fc3（fullconnect）全连接函数3为线性函数：y = Wx + b，并将100个节点连接到10个节点上。
        self.fc3 = nn.Linear(100, 10)

    #定义该神经网络的向前传播函数，该函数必须定义，一旦定义成功，向后传播函数也会自动生成（autograd）
    def forward(self, x):
        # 输入x经过全连接1，再经过ReLU激活函数，然后更新x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输入x经过全连接3，然后更新x
        x = self.fc3(x)
        return x
# # 定义数据转换格式
mnist_transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x : x.resize_(28*28))])
#
# # 导入数据，定义数据接口
# # 1.root，表示mnist数据的加载的相对目录
# # 2.train，表示是否加载数据库的训练集，false的时候加载测试集
# # 3.download，表示是否自动下载mnist数据集
# # 4.transform，表示是否需要对数据进行预处理，none为不进行预处理
traindata = torchvision.datasets.MNIST(root="./mnist", train=True, download=True, transform=mnist_transform)
testdata  = torchvision.datasets.MNIST(root="./mnist", train=False, download=True, transform=mnist_transform)

# 将训练集的*张图片划分成*份，每份256(batch_size)张图，用于mini-batch输入
# shffule=True在表示不同批次的数据遍历时，打乱顺序
# num_workers=n表示使用n个子进程来加载数据
trainloader = torch.utils.data.DataLoader(traindata, batch_size=256, shuffle=True, num_workers=0)
testloader = torch.utils.data.DataLoader(testdata, batch_size=256, shuffle=True, num_workers=0)

net = Net()
loss_function = nn.CrossEntropyLoss() # 交叉熵损失函数
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=1e-04) # 随机梯度下降优化
# 保存整个网络 #...It won't be checked...是保存模型时的输出
PATH1="./mnist_net_all.pkl"
torch.save(net,PATH1)

net = torch.load(PATH1) # 加载模型

image_100 = []
label_100 = []
for i in range(10000):
    index = i
    image = Variable(testdata[index][0].resize_(1,784), requires_grad=True) # requires_grad存储梯度值
    label = torch.tensor([testdata[index][1]])
    outputs = net(image)
    loss = loss_function(outputs, label)
    loss.backward()
    epsilon = 0.2 # 扰动程度
    x_grad = torch.sign(image.grad.data) # 快速梯度符号法
    x_adversarial = torch.clamp(image.data + epsilon * x_grad, 0, 1) # 0和1表示限制范围的下限和上限
    x_adversarial = x_adversarial[0].numpy().tolist()
    label = label[0].numpy().tolist()
    image_100.append(x_adversarial)
    label_100.append(label)
    # print(type(label_100))

image_arry = np.array(image_100)
label_arry = np.array(label_100)


def read_data_sets(dir, percentage=1.0):

    train_image_file = "train-images-idx3-ubyte.gz"
    train_label_file = "train-labels-idx1-ubyte.gz"

    train_image_file = os.path.join(dir, train_image_file)
    train_label_file = os.path.join(dir, train_label_file)
    train_images, train_labels = extract_images_and_labels(train_image_file, train_label_file, percentage)

    perm = np.arange(len(train_images))
    np.random.shuffle(perm)
    valid_images = train_images[perm[:len(train_images)//10]]
    valid_labels = train_labels[perm[:len(train_labels)//10]]
    train_images = train_images[perm[len(train_images)//10:]]
    train_labels = train_labels[perm[len(train_labels)//10:]]

    #train_images = crop_augment(train_images)
    #train_labels = np.repeat(train_labels, 2)

    # test_image_file = "t10k-images-idx3-ubyte.gz"
    # test_label_file = "t10k-labels-idx1-ubyte.gz"
    # test_image_file = os.path.join(dir, test_image_file)
    # test_label_file = os.path.join(dir, test_label_file)
    # test_images, test_labels = extract_images_and_labels(test_image_file, test_label_file)

    test_images = image_arry
    test_labels = label_arry

    train = DataSet(train_images, train_labels)
    valid = DataSet(valid_images, valid_labels)
    test = DataSet(test_images, test_labels)
    return DataSets(train, valid, test)
