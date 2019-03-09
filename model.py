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

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        return x



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

        self.classifier = nn.Linear(6*6*256, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = x.view(x.size(0), -1)  # 将卷积结果拉伸成一列便于分类
        x = self.classifier(x)
        return x


class CGAN(object):
    def __init__(self,
                 image_size=132,  # 训练图像的尺寸
                 label_size=120,  # 生成图像的尺寸
                 batch_size=4,
                 c_dim=1,
                 checkpoint_dir=None,
                 sample_dir=None):
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size

        self.c_dim = c_dim

        self.checkpoint_dir = checkpoint_dir
        self.sample_dir = sample_dir


    def train(self, opt, netD, netG):
        # 根据输入参数读入数据并进行训练/测试
        # if opt.is_train:
        #     input_setup(opt, "Train_ir")
        #     input_setup(opt, "Train_vi")
        # else:
        #     nx_ir, ny_ir = input_setup(opt, "Test_ir")
        #     nx_vi, ny_vi = input_setup(opt, "Test_vi")

        if opt.is_train:
            data_dir_ir = os.path.join('./{}'.format(opt.checkpoint_dir), "Train_ir", "train.h5")
            data_dir_vi = os.path.join('./{}'.format(opt.checkpoint_dir), "Train_vi", "train.h5")
        else:
            data_dir_ir = os.path.join('./{}'.format(opt.checkpoint_dir), "Test_ir", "test.h5")
            data_dir_vi = os.path.join('./{}'.format(opt.checkpoint_dir), "Test_vi", "test.h5")

        train_data_ir, train_label_ir = read_data(data_dir_ir)
        train_data_vi, train_label_vi = read_data(data_dir_vi)
        # 定义参数的更新方式（D和G分开训练）
        fusion_model = netG()
        discriminator = netD()
        optimizerG = Adam(fusion_model.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))
        optimizerD = Adam(discriminator.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))
        counter = 0
        start_time = time.time()

        # if self.load(self.checkpoint_dir):
        # print(" [*] Load SUCCESS")
        # else:
        # print(" [!] Load failed...")

        # 模型训练
        if opt.is_train:
            print("Training...")

            for ep in range(opt.epoch):
                # Run by batch images
                batch_idxs = len(train_data_ir) // opt.batch_size  # 计算batch的个数作为索引
                for idx in range(0, batch_idxs):
                    batch_images_ir = train_data_ir[idx * opt.batch_size: (idx + 1) * opt.batch_size]
                    batch_labels_ir = train_label_ir[idx * opt.batch_size: (idx + 1) * opt.batch_size]
                    batch_images_vi = train_data_vi[idx * opt.batch_size: (idx + 1) * opt.batch_size]
                    batch_labels_vi = train_label_vi[idx * opt.batch_size: (idx + 1) * opt.batch_size]

                    if opt.gpu:
                        fusion_model = fusion_model.cuda()
                        discriminator = discriminator.cuda()
                        batch_images_ir = torch.autograd.Variable(torch.Tensor(batch_images_ir).cuda())
                        batch_labels_ir = torch.autograd.Variable(torch.Tensor(batch_labels_ir).cuda())
                        batch_images_vi = torch.autograd.Variable(torch.Tensor(batch_images_vi).cuda())
                        batch_labels_vi = torch.autograd.Variable(torch.Tensor(batch_labels_vi).cuda())

                    # 将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
                    input_image = torch.cat((batch_images_ir, batch_images_vi), -1)
                    input_image = input_image.permute(0, 3, 1, 2)
                    batch_labels_ir = batch_labels_ir.permute(0, 3, 1, 2)
                    batch_labels_vi = batch_labels_vi.permute(0, 3, 1, 2)

                    fusion_image = fusion_model(input_image).detach()  # 生成器生成的结果
                    loss_fn = torch.nn.MSELoss(reduce=True, size_average=True)  # 计算均方损失
                    counter += 1  # 用于统计与显示
                    # 生成器每训练依次，判别器训练两次
                    for i in range(2):
                        # ----- train netd -----
                        discriminator.zero_grad()
                        ## train netd with real img
                        # 分别计算正样本和负样本的分类损失
                        pos = discriminator(batch_labels_vi)
                        neg = discriminator(fusion_image)
                        pos_param = torch.autograd.Variable((torch.randn(self.batch_size, 1) * 0.5 + 0.7).cuda())  # 定义一组随机数
                        pos_loss = loss_fn(pos, pos_param)
                        neg_param = torch.autograd.Variable((torch.randn(self.batch_size, 1) * 0.3).cuda())  # 定义一组随机数
                        neg_loss = loss_fn(neg, neg_param)
                        d_loss = neg_loss + pos_loss
                        d_loss.backward()
                        optimizerD.step()

                    ## train netd with fake img
                    fusion_model.zero_grad()
                    fusion_image = fusion_model(input_image)
                    neg_G = discriminator(fusion_image)  # 由训练后的判别器对伪造输入的判别结果
                    g_param_1 = torch.autograd.Variable((torch.randn(self.batch_size, 1) * 0.5 + 0.7).cuda())
                    g_loss_1 = loss_fn(neg_G, g_param_1)
                    # 利用卷积网络计算图像梯度
                    image_grad = gradient().cuda()
                    g_loss_2 = loss_fn(fusion_image, batch_labels_ir) + 5 * loss_fn(
                        image_grad(fusion_image), image_grad(batch_labels_vi))
                    g_loss_total = g_loss_1 + 100 * g_loss_2
                    g_loss_total.backward()
                    optimizerG.step()

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss_d: [%.8f],loss_g:[%.8f]" \
                              % ((ep + 1), counter, time.time() - start_time, d_loss, g_loss_total))
        torch.save(discriminator.state_dict(), 'dcgan_netd.pth')
        torch.save(fusion_model.state_dict(), 'dcgan_netg.pth')

        # # 模型测试
        # else:
        #     print("Testing...")
        #
        #     result = self.fusion_image.eval(
        #         feed_dict={self.images_ir: train_data_ir, self.labels_ir: train_label_ir, self.images_vi: train_data_vi,
        #                    self.labels_vi: train_label_vi})
        #     result = result * 127.5 + 127.5
        #     result = merge(result, [nx_ir, ny_ir])
        #     result = result.squeeze()
        #     image_path = os.path.join(os.getcwd(), opt.sample_dir)
        #     image_path = os.path.join(image_path, "test_image.png")
        #     imsave(result, image_path)
