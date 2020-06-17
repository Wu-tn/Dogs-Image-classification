# 在本节notebook中，使用后续设置的参数在完整训练集上训练模型，大致需要40-50分钟
# 合理安排GPU时长，尽量只在训练时切换到GPU资源
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import os
import shutil
import time
import pandas as pd
import random
# 设置随机数种子
random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
data_dir = 'data/'  # 数据集目录
label_file, train_dir, test_dir = 'labels.csv', 'train', 'test'  # data_dir中的文件夹、文件
new_data_dir = './train_valid_test'  # 整理之后的数据存放的目录
valid_ratio = 0.1  # 验证集所占比例


def mkdir_if_not_exist(path):
    # 若目录path不存在，则创建目录
    if not os.path.exists(os.path.join(*path)):
        os.makedirs(os.path.join(*path))


def reorg_dog_data(data_dir, label_file, train_dir, test_dir, new_data_dir, valid_ratio):
    # 读取训练数据标签
    labels = pd.read_csv(os.path.join(data_dir, label_file))
    id2label = {Id: label for Id, label in labels.values}  # (key: value): (id: label)

    # 随机打乱训练数据
    train_files = os.listdir(
        os.path.join(data_dir, train_dir))  # os.listdir() 方法用于返回指定的文件夹包含的文件或文件夹的名字的列表。 由train文件夹里所有图片组成的列表。
    random.shuffle(train_files)  # 将train文件里的所有.jpg图片打乱顺序

    # 原训练集
    valid_ds_size = int(len(train_files) * valid_ratio)  # 验证集大小
    for i, file in enumerate(train_files):
        img_id = file.split('.')[0]  # file是形式为id.jpg的字符串
        img_label = id2label[img_id]
        if i < valid_ds_size:
            mkdir_if_not_exist([new_data_dir, 'valid', img_label])  # []里面是path，新建了valid文件夹
            shutil.copy(os.path.join(data_dir, train_dir, file),
                        os.path.join(new_data_dir, 'valid',
                                     img_label))  # 复制train文件夹里的某一file到新建的train_valid_test文件夹的valid子文件夹里的子类别文件夹img_label中
        else:
            mkdir_if_not_exist([new_data_dir, 'train', img_label])
            shutil.copy(os.path.join(data_dir, train_dir, file),
                        os.path.join(new_data_dir, 'train', img_label))
        mkdir_if_not_exist([new_data_dir, 'train_valid', img_label])
        shutil.copy(os.path.join(data_dir, train_dir, file),
                    os.path.join(new_data_dir, 'train_valid', img_label))

    # 测试集
    mkdir_if_not_exist([new_data_dir, 'test', 'unknown'])
    for test_file in os.listdir(os.path.join(data_dir, test_dir)):  # 遍历Dog Breed Identification文件夹中的test子文件夹里.jpg的图片文件
        shutil.copy(os.path.join(data_dir, test_dir, test_file),
                    os.path.join(new_data_dir, 'test', 'unknown'))  # 复制到新文件夹test/unknown中
reorg_dog_data(data_dir, label_file, train_dir, test_dir, new_data_dir, valid_ratio)
transform_train = transforms.Compose([
    # 随机对图像裁剪出面积为原图像面积0.08~1倍、且高和宽之比在3/4~4/3的图像，再放缩为高和宽均为224像素的新图像
    transforms.RandomResizedCrop(224, scale=(0.08, 1.0),
                                 ratio=(3.0/4.0, 4.0/3.0)),
    # 以0.5的概率随机水平翻转
    transforms.RandomHorizontalFlip(),
    # 随机更改亮度、对比度和饱和度
    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    # 对各个通道做标准化，(0.485, 0.456, 0.406)和(0.229, 0.224, 0.225)是在ImageNet上计算得的各通道均值与方差
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet上的均值和方差
])


# 在测试集上的图像增强只做确定性的操作
transform_test = transforms.Compose([
    transforms.Resize(256),
    # 将图像中央的高和宽均为224的正方形区域裁剪出来
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
# new_data_dir目录下有train, valid, train_valid, test四个目录
# 这四个目录中，每个子目录表示一种类别，目录中是属于该类别的所有图像
# train_ds = torchvision.datasets.ImageFolder(root=os.path.join(new_data_dir, 'train'),
#                                             transform=transform_train)
# valid_ds = torchvision.datasets.ImageFolder(root=os.path.join(new_data_dir, 'valid'),
#                                             transform=transform_test)
# train_valid_ds = torchvision.datasets.ImageFolder(root=os.path.join(new_data_dir, 'train_valid'),
#                                             transform=transform_train)
# test_ds = torchvision.datasets.ImageFolder(root=os.path.join(new_data_dir, 'test'),
#                                             transform=transform_test)
# batch_size = 128
# train_iter = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
# valid_iter = torch.utils.data.DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
# train_valid_iter = torch.utils.data.DataLoader(train_valid_ds, batch_size=batch_size, shuffle=True)
# test_iter = torch.utils.data.DataLoader(test_ds, batch_size=batch_size, shuffle=False)  # shuffle=False
