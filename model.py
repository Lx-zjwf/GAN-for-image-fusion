# -*- coding: utf-8 -*-

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

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 64, 5, padding=2), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))

        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 5, padding=2), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))

        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))

        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))

        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))

        self.block6 = nn.Sequential(
            nn.Conv2d(512, 1024, 1), nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2))

        self.block7 = nn.Sequential(
            nn.Conv2d(1024, 512, 1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))

        self.block8 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2))

        self.block9 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1), nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2))

        self.block10 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2))

        self.block11 = nn.Sequential(
            nn.Conv2d(128, 64, 5, padding=2), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))

        self.block12 = nn.Sequential(
            nn.Conv2d(64, 1, 5, padding=2), nn.Tanh())

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out3 = self.block3(out2)
        out4 = self.block4(out3)
        out5 = self.block5(out4)
        out6 = self.block6(out5)
        out7 = self.block7(out6)
        add_out = out7 + out5
        out8 = self.block8(add_out)
        add_out = out8 + out4
        out9 = self.block9(add_out)
        add_out = out9 + out3
        out10 = self.block10(add_out)
        add_out = out10 + out2
        out11 = self.block11(add_out)
        add_out = out11 + out1
        high_out = self.block12(add_out)
        return high_out

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
            nn.Conv2d(2, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2))

        self.block2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2))

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2))

        self.block4 = nn.Sequential(
            nn.Conv2d(48, 16, 3, padding=1), nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2))

        self.block5 = nn.Sequential(
            nn.Conv2d(64, 64, 5, padding=2), nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2))

        self.block6 = nn.Sequential(
            nn.Conv2d(64, 32, 5, padding=2), nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2))

        self.block7 = nn.Sequential(
            nn.Conv2d(32, 16, 5, padding=2), nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2))

        self.block8 = nn.Sequential(
            nn.Conv2d(16, 1, 3, padding=1))

        # 加入高频提取模块
        self.block9 = high_extra()

    # 定义残差网络
    def resnet(self, block, i_channel, o_channel):
        layers = []
        layers.append(block(i_channel, 64))
        layers.append(nn.ReLU(inplace=True))
        layers.append(block(64, o_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        out1 = self.block1(x)
        out2 = self.block2(out1)
        out_cat = torch.cat((out1, out2), 1)
        out3 = self.block3(out_cat)
        out_cat = torch.cat((out1, out2, out3), 1)
        out4 = self.block4(out_cat)
        out_cat = torch.cat((out1, out2, out3, out4), 1)
        out5 = self.block5(out_cat)
        out6 = self.block6(out5)
        out7 = self.block7(out6)
        out8 = self.block8(out7)
        # 提取第八层的输出结果为融合后的图像
        self.fusion_res = out8
        out = self.block9(out8.detach())
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
            nn.Linear(7*7*256, 1), nn.Sigmoid())

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
