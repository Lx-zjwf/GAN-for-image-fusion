# -*- coding: utf-8 -*-
import numpy as np
from model import netG, netD
import scipy.misc
import time
import os
import glob
import torch
from torch import nn

def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """
    if is_grayscale:
        # flatten=True 以灰度图的形式读取
        return scipy.misc.imread(path, flatten=True, mode='YCbCr').astype(np.float)
    else:
        return scipy.misc.imread(path, mode='YCbCr').astype(np.float)

def imsave(image, path):
    return scipy.misc.imsave(path, image)

def prepare_data(dataset):
    data_dir = os.path.join(os.sep, (os.path.join(os.getcwd(), dataset)))
    data = glob.glob(os.path.join(data_dir, "*.jpg"))
    data.extend(glob.glob(os.path.join(data_dir, "*.bmp")))
    data.sort(key=lambda x: int(x[len(data_dir)+1:-4]))
    return data

def input_setup(index):
    padding = 6
    sub_ir_sequence = []
    sub_vi_sequence = []
    input_ir = (imread(data_ir[index])-127.5)/127.5  # 将该幅图像的数据归一化
    input_ir = np.lib.pad(input_ir, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_ir.shape
    input_ir = input_ir.reshape([w, h, 1])
    input_vi = (imread(data_vi[index])-127.5)/127.5
    input_vi = np.lib.pad(input_vi, ((padding, padding), (padding, padding)), 'edge')
    w, h = input_vi.shape
    input_vi = input_vi.reshape([w, h, 1])
    sub_ir_sequence.append(input_ir)
    sub_vi_sequence.append(input_vi)
    train_data_ir = np.asarray(sub_ir_sequence)
    train_data_vi = np.asarray(sub_vi_sequence)
    return train_data_ir, train_data_vi


# 加载模型参数进行图像融合测试
fusion_model = netG().cuda().eval()
# discriminator = netD().cuda()
ep = 0
model_path = os.path.join(os.getcwd(), 'WGAN_weight_0312', 'epoch' + str(ep))
netG_path = os.path.join(model_path, 'WGAN_netG.pth')
netD_path = os.path.join(model_path, 'WGAN_netD.pth')
fusion_model.load_state_dict(torch.load(netG_path))
# discriminator.load_state_dict(torch.load(netD_path))
data_ir = prepare_data('Test_ir')
data_vi = prepare_data('Test_vi')
for i in range(15, len(data_ir)):
    start = time.time()
    train_data_ir, train_data_vi = input_setup(i)
    # from_numpy得到的是DoubleTensor类型的，需要转成FloatTensor
    train_data_ir = torch.from_numpy(train_data_ir).float()
    train_data_vi = torch.from_numpy(train_data_vi).float()
    input_image = torch.cat((train_data_ir, train_data_vi), -1)
    input_image = input_image.permute(0, 3, 1, 2)
    input_image = torch.autograd.Variable(input_image.cuda(), volatile=True)
    result_low = fusion_model(input_image)
    result = fusion_model.feature
    result = result * 127.5 + 127.5
    # 将生成的variable数据转成numpy类型
    result = result.squeeze().cpu().detach().numpy()
    image_path = os.path.join(os.getcwd(), 'WGAN_test_result', 'epoch' + str(ep))
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if i <= 9:
        image_path = os.path.join(image_path, '0'+str(i)+".bmp")
    else:
        image_path = os.path.join(image_path, str(i)+".bmp")
    end = time.time()
    # print(out.shape)
    imsave(result, image_path)
    print("Testing [%d] success, Testing time is [%f]" % (i, end-start))
