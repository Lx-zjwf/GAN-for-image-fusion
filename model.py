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
import numpy

# 定义残差瓶颈块
class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # 定义瓶颈层
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


# 在生成器最后加上残差块，用于生成低频图像并分解出高频图像
class Residual_Block(nn.Module):

    def __init__(self, i_channel, o_channel, block, stride=1):
        super(Residual_Block, self).__init__()
        self.inplanes = 32
        self.conv1 = nn.Conv2d(in_channels=i_channel, out_channels=3, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)

        # self.layer1 = self._make_layer(block, self.inplanes)

        self.conv2 = nn.Conv2d(in_channels=3, out_channels=o_channel, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(o_channel)
        self.tanh = nn.Tanh()

    def _make_layer(self, block, planes, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        # 如果残差块增大了原输入的维度，通过1*1的瓶颈层使两个相加层维度保持一致
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 将输出降维至符号网络的维度
        for i in range(1, 3):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 加入残差块
        # out = self.layer1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # 用生成图像减去高频图像得到低频图像，作为生成器的输出
        out = residual - out
        out = self.tanh(out)
        return out

# 定义图像生成模型
class netG(nn.Module):
    def __init__(self):
        super(netG, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(2, 256, 5), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))

        self.block2 = nn.Sequential(
            nn.Conv2d(256, 128, 5), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 64, 3), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 32, 3), nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2))

        self.block5 = nn.Sequential(
            nn.Conv2d(32, 1, 1), nn.Tanh()
        )

        # 加入一个残差块
        # self.inplanes = 64
        # self.reslayer1 = self._make_layer(block, self.inplanes)

        # 对模型进行压缩
        # self.block4 = nn.Sequential(
        #     nn.Conv2d(128, 64, 1), nn.BatchNorm2d(64),
        #     nn.LeakyReLU(0.2))

        # self.inplanes = 32
        # self.reslayer2 = self._make_layer(block, self.inplanes)

        # self.block6 = nn.Sequential(
        #     nn.Conv2d(64, 32, 1), nn.BatchNorm2d(32),
        #     nn.LeakyReLU(0.2))

        # 最后一层是残差块，用于生成低频结果
        # self.block4 = Residual_Block(1, 1, block)


    def _make_layer(self, block, planes, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        # 如果残差块增大了原输入的维度，通过1*1的瓶颈层降低计算量（downsample）
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        # 将输出降维至符号网络的维度
        for i in range(1, 2):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x

# 生成网络测试
generator = netG()
print(generator)


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
            mean = torch.mean(image)
            # 定义归一化系数
            norm_coeff = max(mean, 1-mean)
            loss = 1 - stand_dev / norm_coeff
            self.loss += loss

        return self.loss

    def backward(self):
        self.loss.backward()
        return self.loss
