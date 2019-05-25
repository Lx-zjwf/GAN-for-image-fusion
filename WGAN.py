from utils import read_data, input_setup, gradient
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
from tensorboardX import SummaryWriter

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
                 image_size=136,  # 训练图像的尺寸
                 label_size=136,  # 生成图像的尺寸
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
        # input_setup(opt, "Train_data")

        data_dir = os.path.join('./{}'.format(opt.checkpoint_dir), "Train_data", "train.h5")

        # 读取可见光与红外图像的训练数据与标签数据
        train_data = read_data(data_dir)

        lda = 3
        npad = 16
        counter = 0

        # 定义参数的更新方式（D和G分开训练）,需要用到的损失函数
        recon_model = netG()
        discriminator = netD()
        optimizerG = Adam(recon_model.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))
        optimizerD = Adam(discriminator.parameters(), lr=opt.learning_rate, betas=(opt.beta1, 0.999))
        # 定义损失函数，通过reduce和size_average控制均值的求取
        criterion = nn.BCELoss(reduce=True, size_average=True)  # 定义损失函数：交叉熵
        dis_loss = torch.nn.MSELoss(reduce=True, size_average=True)  # 计算均方损失（用于判别器）
        class_loss = nn.L1Loss(reduce=True, size_average=True)

        if opt.gpu:
            recon_model.cuda()
            discriminator.cuda()
            criterion.cuda()  # it's a good habit
            dis_loss.cuda()
            class_loss.cuda()

        # 使用visdom进行可视化
        # vis = Visualizer(env='WGAN_fusion')  # 为了可视化增加的内容
        # loss_meter = meter.AverageValueMeter()  # 为了可视化增加的内容
        # 使用tensorboard进行可视化
        writer = SummaryWriter(log_dir='logs')

        # if self.load(self.checkpoint_dir):
        # print(" [*] Load SUCCESS")
        # else:
        # print(" [!] Load failed...")

        # 加载预训练模型
        # net_path = os.path.join(os.getcwd(), 'weight_0507', 'epoch0')
        # netG_path = os.path.join(net_path, 'netG.pth')
        # netD_path = os.path.join(net_path, 'netD.pth')
        # recon_model.load_state_dict(torch.load(netG_path))
        # discriminator.load_state_dict(torch.load(netD_path))

        # 阶段内loss的平均值
        g_loss_sum = 0
        d_loss_sum = 0

        # 模型训练
        if opt.is_train:
            print("Training...")

            for ep in range(opt.epoch):
                # Run by batch images
                # loss_meter.reset()

                batch_idxs = len(train_data) // opt.batch_size  # 计算batch的个数作为索引
                for idx in range(0, batch_idxs):
                    batch_images = train_data[idx * opt.batch_size: (idx + 1) * opt.batch_size]
                    # 注意此时的数据涉及到维度转换的问题而非
                    # 对标签图像进行分解，用于之后损失函数的计算
                    images_low, images_high = lowpass(batch_images, lda, npad)

                    if opt.gpu:
                        batch_images = torch.autograd.Variable(torch.Tensor(batch_images).cuda())
                        images_low = torch.autograd.Variable(torch.Tensor(images_low).cuda())
                        images_high = torch.autograd.Variable(torch.Tensor(images_high).cuda())

                    # 将红外和可见光图像在通道方向连起来，第一通道是红外图像，第二通道是可见光图像
                    input_image = images_low
                    batch_labels = batch_images
                    # 根据网络的卷积方法对待训练图像进行维度变换
                    batch_labels = batch_labels.permute(0, 3, 1, 2)

                    counter += 1  # 用于统计与显示
                    # 生成器每训练一次，判别器训练两次
                    for i in range(3):
                        # ----- train netd -----
                        # discriminator.zero_grad()
                        ## train netd with real img
                        # 分别计算正样本和负样本的分类损失
                        pos = discriminator(batch_labels)
                        # 使用detach()截断梯度流,更新判别器参数不对生成器产生影响
                        # 生成器生成的结果
                        recon_res = recon_model(input_image).detach()
                        # 判别网络对高频图谱进行分类
                        neg = discriminator(recon_res)
                        pos_param = torch.autograd.Variable((torch.ones(self.batch_size, 1)).cuda())  # 定义一组随机数
                        pos_loss = class_loss(pos, pos_param)
                        neg_param = torch.autograd.Variable((torch.zeros(self.batch_size, 1)).cuda())  # 定义一组随机数
                        neg_loss = class_loss(neg, neg_param)
                        d_loss = neg_loss + pos_loss
                        optimizerD.zero_grad()
                        d_loss.backward()
                        optimizerD.step()

                    # train netd with fake img
                    # fusion_model.zero_grad()
                    recon_res = recon_model(input_image)
                    neg_G = discriminator(recon_res)  # 由训练后的判别器对伪造输入的判别结果
                    g_param_1 = torch.autograd.Variable((torch.ones(self.batch_size, 1)).cuda())
                    # 生成器损失一：判别器对生成结果的判别损失
                    g_loss_1 = class_loss(neg_G, g_param_1)
                    # 生成器损失二：生成的重建结果与原始图像的差异
                    g_loss_2 = dis_loss(recon_res, batch_labels)
                    # 生成器损失三：生成图像的梯度损失
                    grad_filter = gradient().cuda()
                    label_grad = grad_filter(batch_labels)
                    recon_grad = grad_filter(recon_res)
                    g_loss_3 = dis_loss(label_grad, recon_grad)
                    # 计算总损失
                    g_loss_total = g_loss_1 + 1e3 * g_loss_2 + 1e6 * g_loss_3
                    # loss_meter.add(g_loss_total.data.cpu())  # loss可视化
                    optimizerG.zero_grad()
                    g_loss_total.backward()
                    optimizerG.step()
                    # 查看生成模型的网络参数，同 fusion_model.block5[0].weight.grad
                    params = list(recon_model.parameters())
                    # print(params[16].grad)

                    # 对阶段内的损失进行累加
                    g_loss_sum += g_loss_total.data.cpu()
                    d_loss_sum += d_loss.data.cpu()

                    if counter % 10 == 0:
                        print("Epoch: [%2d], step: [%2d], loss_d: [%.8f],loss_g:[%.8f]"
                              % (ep, counter, d_loss, g_loss_total))
                    # 损失可视化
                    if counter % 100 == 0:
                        # vis.plot_many_stack({'train_loss': loss_meter.value()[0]})
                        writer.add_scalar('generator loss', g_loss_sum / 100.0, counter / 100)
                        writer.add_scalar('discriminator loss', d_loss_sum / 100.0, counter / 100)
                        g_loss_sum = 0
                        d_loss_sum = 0

                model_path = os.path.join(os.getcwd(), 'weight_0518', 'epoch' + str(ep + 1))
                if not os.path.exists(model_path):
                    os.makedirs(model_path)
                netD_path = os.path.join(model_path, 'netD.pth')
                netG_path = os.path.join(model_path, 'netG.pth')
                torch.save(discriminator.state_dict(), netD_path)
                torch.save(recon_model.state_dict(), netG_path)

        writer.close()
