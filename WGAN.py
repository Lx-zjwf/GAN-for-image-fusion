from utils import read_data, input_setup
import scipy.misc
from sporco.util import tikhonov_filter
import torch
import numpy as np
import os
import visdom
from torchnet import meter
from torch import nn
from torch.optim import Adam
from model import stand_loss
import torch.nn.functional as F

# begin training
print('begin training, be patient')

# 定义可视化
class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_many_stack(self, d):
        '''
        self.plot('loss',1.00)
        '''
        name = list(d.keys())
        name_total = " ".join(name)
        x = self.index.get(name_total, 0)
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        # print(x)
        self.vis.line(Y=y, X=np.ones(y.shape)*x, win=str(name_total),  # unicode
                      opts=dict(legend=name, title=name_total), update=None if x == 0 else 'append')
        self.index[name_total] = x + 1

def imsave(image, path):
    return scipy.misc.imsave(path, image)

# 通过滤波将图像分解为低频图像和高频图像
def lowpass(image_batch, lda, npad):
    image = image_batch[0]
    image = np.squeeze(image)  # 去掉第一个维度，得到可直接处理的二维数据
    # 注意数组操作reshape和transpose的区别，transpose慎用（转置，改变原有数据顺序）
    # 定义存放高频信号和低频信号的batch
    low_filt_batch, high_filt_batch = tikhonov_filter(image, lda, npad)
    # 将low_filt_batch和high_filt_batch增加第一个维度
    low_filt_batch = np.expand_dims(low_filt_batch, axis=0)
    high_filt_batch = np.expand_dims(high_filt_batch, axis=0)
    # 将每个batch中的图像依次提取出来并分解，并各自合并成一个新的bacth
    for i in range(1, image_batch.shape[0]):
        image = image_batch[i]
        image = np.squeeze(image)
        low_filt_image, high_filt_image = tikhonov_filter(image, lda, npad)
        low_filt_image = np.expand_dims(low_filt_image, axis=0)
        high_filt_image = np.expand_dims(high_filt_image, axis=0)
        # 对第一个维度进行合并
        low_filt_batch = np.concatenate((low_filt_batch, low_filt_image), axis=0)
        high_filt_batch = np.concatenate((high_filt_batch, high_filt_image), axis=0)
    # 增加第二个维度
    low_filt_batch = np.expand_dims(low_filt_batch, axis=1)
    high_filt_batch = np.expand_dims(high_filt_batch, axis=1)

    return low_filt_batch, high_filt_batch


# 将红外图像与可见光图像的高频部分进行合并
def combine_high_image(ir_high, vi_high):
    combine_high = ir_high
    _, _, row, col = ir_high.shape
    for i in range(ir_high.shape[0]):
        # 对第i幅图像进行分析
        for m in range(row):
            for n in range(col):
                if abs(ir_high[i][0][m][n]) > abs(vi_high[i][0][m][n]):
                    combine_high[i][0][m][n] = ir_high[i][0][m][n]
                else:
                    combine_high[i][0][m][n] = vi_high[i][0][m][n]

    return combine_high


class FusionGAN(object):
    def __init__(self,
                 image_size=132,  # 训练图像的尺寸
                 label_size=132,  # 生成图像的尺寸
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
        if opt.is_train:
            input_setup(opt, "Train_ir")
            input_setup(opt, "Train_vi")
        else:
            nx_ir, ny_ir = input_setup(opt, "Test_ir")
            nx_vi, ny_vi = input_setup(opt, "Test_vi")

        if opt.is_train:
            data_dir_ir = os.path.join('./{}'.format(opt.checkpoint_dir), "Train_ir", "train.h5")
            data_dir_vi = os.path.join('./{}'.format(opt.checkpoint_dir), "Train_vi", "train.h5")
        else:
            data_dir_ir = os.path.join('./{}'.format(opt.checkpoint_dir), "Test_ir", "test.h5")
            data_dir_vi = os.path.join('./{}'.format(opt.checkpoint_dir), "Test_vi", "test.h5")

        # 读取可见光与红外图像的训练数据与标签数据
        train_data_ir, train_label_ir = read_data(data_dir_ir)
        train_data_vi, train_label_vi = read_data(data_dir_vi)

        lda = 5
        npad = 16
        counter = 0

        # 定义参数的更新方式（D和G分开训练）,需要用到的损失函数
        fusion_model = netG()
        discriminator = netD()
        optimizerG = Adam(fusion_model.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))
        optimizerD = Adam(discriminator.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))
        # criterion
        criterion = nn.BCELoss(reduce=True, size_average=True)  # 定义损失函数：交叉熵
        dis_loss = torch.nn.MSELoss(reduce=True, size_average=True)  # 计算均方损失（用于判别器）
        low_loss = nn.L1Loss()

        if opt.gpu:
            fusion_model.cuda()
            discriminator.cuda()
            criterion.cuda()  # it's a good habit
            dis_loss.cuda()

        # 可视化定义
        vis = Visualizer(env='WGAN_fusion')  # 为了可视化增加的内容
        loss_meter = meter.AverageValueMeter()  # 为了可视化增加的内容

        # if self.load(self.checkpoint_dir):
        # print(" [*] Load SUCCESS")
        # else:
        # print(" [!] Load failed...")

        # 加载预训练模型
        net_path = os.path.join(os.getcwd(), 'lowloss_sum_weight', 'epoch0')
        netG_path = os.path.join(net_path, 'netG.pth')
        netD_path = os.path.join(net_path, 'netD.pth')
        fusion_model.load_state_dict(torch.load(netG_path))
        discriminator.load_state_dict(torch.load(netD_path))

        # 模型训练
        if opt.is_train:
            print("Training...")

            for ep in range(opt.epoch):
                # Run by batch images
                loss_meter.reset()

                batch_idxs = len(train_data_ir) // opt.batch_size  # 计算batch的个数作为索引
                for idx in range(0, batch_idxs):
                    batch_images_ir = train_data_ir[idx * opt.batch_size: (idx + 1) * opt.batch_size]
                    batch_labels_ir = train_label_ir[idx * opt.batch_size: (idx + 1) * opt.batch_size]
                    batch_images_vi = train_data_vi[idx * opt.batch_size: (idx + 1) * opt.batch_size]
                    batch_labels_vi = train_label_vi[idx * opt.batch_size: (idx + 1) * opt.batch_size]
                    # 注意此时的数据涉及到维度转换的问题而非
                    # 对标签图像进行分解，用于之后损失函数的计算
                    label_ir_low, label_ir_high = lowpass(batch_labels_ir, lda, npad)
                    label_vi_low, label_vi_high = lowpass(batch_labels_vi, lda, npad)
                    # 将vi和ir的高频图像进行合并
                    combine_label_high = combine_high_image(label_ir_high, label_vi_high)

                    if opt.gpu:
                        batch_images_ir = torch.autograd.Variable(torch.Tensor(batch_images_ir).cuda())
                        batch_labels_ir = torch.autograd.Variable(torch.Tensor(batch_labels_ir).cuda())
                        label_ir_low = torch.autograd.Variable(torch.Tensor(label_ir_low).cuda())
                        batch_images_vi = torch.autograd.Variable(torch.Tensor(batch_images_vi).cuda())
                        batch_labels_vi = torch.autograd.Variable(torch.Tensor(batch_labels_vi).cuda())
                        label_vi_low = torch.autograd.Variable(torch.Tensor(label_vi_low).cuda())
                        combine_label_high = torch.autograd.Variable(torch.Tensor(combine_label_high).cuda())

                    # 将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
                    input_image = torch.cat((batch_images_ir, batch_images_vi), -1)
                    # 根据网络的卷积方法对待训练图像进行维度变换
                    input_image = input_image.permute(0, 3, 1, 2)
                    # 计算标签图像的灰度平均值
                    average_labels = (batch_labels_ir + batch_labels_vi) / 2.0
                    label_average_low = (label_ir_low + label_vi_low) / 2.0
                    batch_labels_ir = batch_labels_ir.permute(0, 3, 1, 2)
                    batch_labels_vi = batch_labels_vi.permute(0, 3, 1, 2)
                    average_labels = average_labels.permute(0, 3, 1, 2)
                    # imsave((label_average_low[0][0]*127.5+127.5).cpu().numpy().astype(np.uint8), 'average_low.bmp')

                    counter += 1  # 用于统计与显示
                    # 生成器每训练一次，判别器训练两次
                    for i in range(1):
                        # ----- train netd -----
                        # discriminator.zero_grad()
                        ## train netd with real img
                        # 分别计算正样本和负样本的分类损失
                        pos = discriminator(combine_label_high)
                        # 使用detach()截断梯度流,更新判别器参数不对生成器产生影响
                        # 生成器生成的结果
                        fusion_high = fusion_model(input_image).detach()
                        # 判别网络对高频图谱进行分类
                        neg = discriminator(fusion_high)
                        pos_param = torch.autograd.Variable((torch.ones(self.batch_size, 1)).cuda())  # 定义一组随机数
                        pos_loss = criterion(pos, pos_param)
                        neg_param = torch.autograd.Variable((torch.zeros(self.batch_size, 1)).cuda())  # 定义一组随机数
                        neg_loss = criterion(neg, neg_param)
                        d_loss = neg_loss + pos_loss
                        optimizerD.zero_grad()
                        d_loss.backward()
                        optimizerD.step()

                    # train netd with fake img
                    # fusion_model.zero_grad()
                    fusion_high = fusion_model(input_image)
                    # 将融合后的图像分解成高频和低频
                    fusion_image = fusion_model.fusion_res
                    fusion_low = fusion_image - fusion_high
                    neg_G = discriminator(fusion_high)  # 由训练后的判别器对伪造输入的判别结果
                    g_param_1 = torch.autograd.Variable((torch.ones(self.batch_size, 1)).cuda())
                    # 生成器损失一：判别器对生成结果的判别损失
                    g_loss_1 = criterion(neg_G, g_param_1)
                    # 生成器损失二：生成的高频区域与高频融合图像的相似度
                    g_loss_2 = dis_loss(fusion_high, combine_label_high)
                    # 生成器损失三：生成图像的低频部分在原始图像的低频分布内（L1损失）
                    low_fuir_diss = abs(fusion_low - label_ir_low)
                    low_fuvi_diss = abs(fusion_low - label_vi_low)
                    low_diss = low_fuir_diss + low_fuvi_diss
                    low_irvi_diss = abs(label_ir_low - label_vi_low)
                    g_loss_3 = low_loss(low_diss, low_irvi_diss)
                    # 生成器损失四：计算图像标准差损失
                    loss_4 = stand_loss()
                    g_loss_4 = loss_4(fusion_image) / 4.0
                    # 生成器损失五：生成低频区域与低频均值的差异  # 生成图像灰度与平均图像灰度的差异
                    g_loss_5 = dis_loss(fusion_low, label_ir_low) + dis_loss(fusion_low, label_vi_low)
                    g_loss_total = g_loss_1 + 1e3 * g_loss_2 + g_loss_5 + 0.2 * g_loss_4  # + 1e4 * g_loss_3
                    loss_meter.add(g_loss_total.data.cpu())  # loss可视化
                    optimizerG.zero_grad()
                    g_loss_total.backward()
                    optimizerG.step()
                    # 查看生成模型的网络参数，同 fusion_model.block5[0].weight.grad
                    params = list(fusion_model.parameters())
                    # print(params[16].grad)

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], loss_d: [%.8f],loss_g:[%.8f]"
                              % ((ep + 1), counter, d_loss, g_loss_total))

                vis.plot_many_stack({'train_loss': loss_meter.value()[0]})  # 为了可视化增加的内容
                model_path = os.path.join(os.getcwd(), 'WGAN_weight_0504', 'epoch' + str(ep))
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                netD_path = os.path.join(model_path, 'netD.pth')
                netG_path = os.path.join(model_path, 'netG.pth')
                torch.save(discriminator.state_dict(), netD_path)
                torch.save(fusion_model.state_dict(), netG_path)
