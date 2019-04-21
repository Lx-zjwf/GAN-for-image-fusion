# -*- coding: utf-8 -*-
from utils import (
  read_data,  # 读取数据集及标签
  input_setup,
  gradient
)

import time
import os
import torch
from torch import nn
from torch.optim import Adam
from torch.autograd import Variable

# 定义高频提取模型
class high_extra(nn.Module):
    def __init__(self):
        super(high_extra, self).__init__()
        self.feature = Variable(torch.zeros(4, 1, 120, 120).cuda())

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, stride=1, padding=2), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))

        self.block5 = nn.Sequential(
            nn.Conv2d(128, 1, 9, stride=1, padding=4), nn.Tanh())

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

# 定义残差块
class resblock(nn.Module):
    def __init__(self, i_channel, o_channel):
        super(resblock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(i_channel, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, o_channel, 3, padding=1), nn.BatchNorm2d(o_channel))

    def forward(self, x):
        residual = x
        out = self.layer1(x)
        out = self.layer2(x)
        out = self.layer3(x)
        out = residual + out
        return out

# 定义图像生成模型
class netG(nn.Module):
    def __init__(self):
        super(netG, self).__init__()

        # 第五层的输出：融合结果
        self.fusion_res = Variable(torch.zeros(1, 1, 3, 3).cuda())

        self.block1 = nn.Sequential(
            nn.Conv2d(2, 64, 5), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))

        self.block2 = self.resnet(resblock, 64, 64)

        self.block3 = nn.LeakyReLU(0.2)

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 256, 5), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))

        self.block5 = nn.Sequential(
            nn.Conv2d(256, 1, 5), nn.Tanh())

        # 加入高频提取模块
        self.block6 = high_extra()

    # 定义残差网络
    def resnet(self, block, i_channel, o_channel):
        layers = []
        layers.append(block(i_channel, 64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(block(64, o_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.block1(x)
        residual = x
        x = self.block2(x)
        x += residual
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        # 提取第五层的输出结果为融合后的图像
        self.fusion_res = x
        out = self.block6(x)  # 最终的输出结果为高频图像
        return out


# 定义判别模型
class netD(nn.Module):
    def __init__(self):
        super(netD, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2), nn.LeakyReLU(0.2))

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))

        self.block4 = nn.Sequential(
            nn.Conv2d(128, 256, 3, stride=2), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))

        self.classifier = nn.Sequential(
            nn.Linear(6*6*256, 1), nn.Sigmoid())

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)  # 将卷积结果拉伸成一列便于分类
        x = self.classifier(x)
        return x

# 定义图像标准差损失
class stand_loss(nn.Module):
    def __init__(self):
        super(stand_loss, self).__init__()

    def forward(self, input):
        # 将input按照不同的batch进行分解
        size = input.shape[0]
        self.loss = 0
        for i in range(size):
            image = input[i]
            # 计算图像的均值和标准差
            stand_dev = torch.std(image)
            # mean = abs(torch.mean(image))
            # 定义归一化系数
            # norm_coeff = max(mean, 1-mean)
            loss = abs(1 - stand_dev)  # / norm_coeff
            self.loss += loss

        return self.loss

    def backward(self):
        self.loss.backward()
        return self.loss
