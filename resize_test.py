# -*- coding: utf-8 -*-
import numpy as np
from model import netG
from sporco.util import tikhonov_filter
import scipy.misc
import time
import os
import glob
import cv2
import torch

def imread(path, is_grayscale=True):
    """
    Read image using its path.
    Default value is gray-scale, and image is read by YCbCr format as the paper said.
    """

    if is_grayscale:
        # 将图像转换为灰度图
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

# 分解得到图像的低频和高频区域
# 通过滤波将图像分解为低频图像和高频图像
def decomp_combine_image(ir_image, vi_image):
    # 转换为float类型 .astype(np.float)
    ir_low, ir_high = tikhonov_filter((ir_image-127.5)/127.5, 3, 16)
    vi_low, vi_high = tikhonov_filter((vi_image-127.5)/127.5, 3, 16)

    # 对红外和可见光图像的高频区域进行融合
    combine_high = ir_high
    row, col = ir_high.shape
    for m in range(row):
        for n in range(col):
            if abs(ir_high[m][n]) > abs(vi_high[m][n]):
                combine_high[m][n] = ir_high[m][n]
            else:
                combine_high[m][n] = vi_high[m][n]

    # 利用主成分分析的方法对低频区域进行融合
    # ir_low_flat = ir_low.flatten()
    # vi_low_flat = vi_low.flatten()
    # cov_array = np.cov(ir_low_flat, vi_low_flat)  # 计算协方差矩阵
    # 计算特征值与特征向量
    # fea_val, fea_vec = np.linalg.eig(cov_array)
    # scale = int(abs(fea_vec[0][0] / (fea_vec[1][0])))
    # lambda1 = abs(fea_vec[0][0] / scale) / (abs(fea_vec[0][0] / scale) + abs(fea_vec[1][0]))
    # lambda2 = abs(fea_vec[1][0]) / (abs(fea_vec[0][0] / scale) + abs(fea_vec[1][0]))
    # combine_low = lambda1 * ir_low + lambda2 * vi_low

    # 基于热源集中度的低频图像融合方法
    ir_low = ir_low * 127.5 + 127.5
    vi_low = vi_low * 127.5 + 127.5
    combine_low = np.zeros((row, col))
    energy_ir = np.mean(ir_low * ir_low)
    energy_vi = np.mean(vi_low * vi_low)
    # 定义分块的尺寸
    block = 20
    ht_stride = int(row / block)
    wd_stride = int(col / block)
    # 用于主成分分析的矩阵（能量，空间频率）
    ir_feature = []
    vi_feature = []
    for i in range(0, block):
        for j in range(0, block):
            # 将每一块取出进行处理
            if (i != block-1) and (j != block-1):
                region_ir = ir_low[ht_stride * i:ht_stride * (i + 1), wd_stride * j:wd_stride * (j + 1)]
                region_vi = vi_low[ht_stride * i:ht_stride * (i + 1), wd_stride * j:wd_stride * (j + 1)]
            elif (i == block-1) and (j != block-1):
                region_ir = ir_low[ht_stride * i:row, wd_stride * j:wd_stride * (j + 1)]
                region_vi = vi_low[ht_stride * i:row, wd_stride * j:wd_stride * (j + 1)]
            elif (i != block-1) and (j == block-1):
                region_ir = ir_low[ht_stride * i:ht_stride * (i + 1), wd_stride * j:col]
                region_vi = vi_low[ht_stride * i:ht_stride * (i + 1), wd_stride * j:col]
            elif (i == block-1) and (j == block-1):
                region_ir = ir_low[ht_stride * i:row, wd_stride * j:col]
                region_vi = vi_low[ht_stride * i:row, wd_stride * j:col]

            energy_ir_region = np.mean(region_ir * region_ir)
            energy_vi_region = np.mean(region_vi * region_vi)
            energy_ir_region_norm = energy_ir_region / (energy_ir_region + energy_vi_region)
            energy_vi_region_norm = energy_vi_region / (energy_ir_region + energy_vi_region)
            ir_feature.append(1.5 * energy_ir_region_norm)
            vi_feature.append(1.5 * energy_vi_region_norm)
            hr_ir = (energy_ir_region - energy_ir) / energy_ir
            hr_vi = (energy_vi_region - energy_vi) / energy_vi
            hscr = abs(hr_ir / hr_vi)

            grad_ir_y = cv2.Sobel(region_ir, cv2.CV_64F, dx=0, dy=1)
            fr_ir = np.mean(abs(grad_ir_y))
            grad_ir_x = cv2.Sobel(region_ir, cv2.CV_64F, dx=1, dy=0)
            fc_ir = np.mean(abs(grad_ir_x))
            sf_ir = np.sqrt(pow(fc_ir, 2) + pow(fr_ir, 2))

            grad_vi_y = cv2.Sobel(region_vi, cv2.CV_64F, dx=0, dy=1)
            fr_vi = np.mean(abs(grad_vi_y))
            grad_vi_x = cv2.Sobel(region_vi, cv2.CV_64F, dx=1, dy=0)
            fc_vi = np.mean(abs(grad_vi_x))
            sf_vi = np.sqrt(pow(fc_vi, 2) + pow(fr_vi, 2))

            ratio = sf_ir / (sf_vi + sf_ir)
            sf_ir_norm = sf_ir / (sf_vi + sf_ir)
            sf_vi_norm = sf_vi / (sf_vi + sf_ir)
            ir_feature.append(sf_ir_norm)
            vi_feature.append(sf_vi_norm)

            region_fusion = ratio * region_ir + (1 - ratio) * region_vi

            # if (hscr > 1.5):
            #     region_fusion = region_ir
            # else:
            #     region_fusion = ratio * region_ir + (1 - ratio) * region_vi

            # if (i != block-1) and (j != block-1):
            #     combine_low[ht_stride * i:ht_stride * (i + 1), wd_stride * j:wd_stride * (j + 1)] = region_fusion
            # elif (i == block-1) and (j != block-1):
            #     combine_low[ht_stride * i:row, wd_stride * j:wd_stride * (j + 1)] = region_fusion
            # elif (i != block-1) and (j == block-1):
            #     combine_low[ht_stride * i:ht_stride * (i + 1), wd_stride * j:col] = region_fusion
            # elif (i == block-1) and (j == block-1):
            #     combine_low[ht_stride * i:row, wd_stride * j:col] = region_fusion

    ir_feature = np.array(ir_feature)
    vi_feature = np.array(vi_feature)

    # 利用主成分分析的方法对低频区域进行融合
    cov_array = np.cov(ir_feature, vi_feature)  # 计算协方差矩阵
    # 计算特征值与特征向量
    fea_val, fea_vec = np.linalg.eig(cov_array)
    lambda1 = abs(fea_vec[0][0]) / (abs(fea_vec[0][0]) + abs(fea_vec[1][0]))
    lambda2 = abs(fea_vec[1][0]) / (abs(fea_vec[0][0]) + abs(fea_vec[1][0]))
    combine_low = lambda1 * ir_low + lambda2 * vi_low

    combine_low = (combine_low - 127.5) / 127.5

    # 将计算得到的图像数据通过uint8的方式显示出来
    # combine_average = (ir_image + vi_image) / 2
    # cv2.imshow('ir_low', (ir_low*127.5 + 127.5).astype(np.uint8))
    # cv2.imshow('ir_high', (ir_high*127.5 + 127.5).astype(np.uint8))
    # cv2.imshow('vi_low', (vi_low*127.5 + 127.5).astype(np.uint8))
    # cv2.imshow('vi_high', (vi_high*127.5 + 127.5).astype(np.uint8))
    # cv2.imshow('combine_high', (combine_high * 127.5 + 127.5).astype(np.uint8))
    # cv2.imshow('combine', ((combine_high + combine_low) * 127.5 + 127.5).astype(np.uint8))
    # cv2.imshow('combine_average', (combine_average.astype(np.uint8)))
    # cv2.imshow('calc_low', ((combine_average - combine_high * 127.5 + 127.5).astype(np.uint8)))

    # cv2.imshow('combine_low', (combine_low * 127.5 + 127.5).astype(np.uint8))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return combine_low

# 加载模型参数进行图像融合测试
recon_model = netG().cuda().eval()
ep = 19
model_path = os.path.join(os.getcwd(), 'weight_0507', 'epoch' + str(ep))
netG_path = os.path.join(model_path, 'netG.pth')
netD_path = os.path.join(model_path, 'netD.pth')
recon_model.load_state_dict(torch.load(netG_path))
data_ir = prepare_data('IR')
data_vi = prepare_data('VIS')
for i in range(0, len(data_ir)):
    start = time.time()

    ir_image = imread(data_ir[i], True)
    vi_image = imread(data_vi[i], True)
    # 得到合并后的低频图像
    combine_low = decomp_combine_image(ir_image, vi_image)
    height, width = combine_low.shape
    size = (int(width * 0.35), int(height * 0.35))
    input_image = cv2.resize(combine_low, size)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = np.expand_dims(input_image, axis=3)
    input_image = torch.FloatTensor(input_image)
    input_image = input_image.permute(0, 3, 1, 2)
    input_image = torch.autograd.Variable(input_image.cuda(), volatile=True)

    # # 防止显存溢出
    # try:
    #     result_low = fusion_model(input_image)
    # except RuntimeError as exception:
    #     if "out of memory" in str(exception):
    #         print("WARNING: out of memory")
    #         if hasattr(torch.cuda, 'empty_cache'):
    #             torch.cuda.empty_cache()
    #     else:
    #         raise exception

    result = recon_model(input_image)
    result = result * 127.5 + 127.5
    # 将生成的variable数据转成numpy类型
    # 查看生成的图片以及低频、高频区域
    result = result.squeeze().cpu().detach().numpy()
    image_path = os.path.join(os.getcwd(), 'result_0509', 'epoch' + str(ep))

    if not os.path.exists(image_path):
        os.makedirs(image_path)
    if i <= 9:
        result_path = os.path.join(image_path, 'result_0'+str(i)+".bmp")
    else:
        result_path = os.path.join(image_path, 'result_'+str(i)+".bmp")
    end = time.time()
    # print(out.shape)
    imsave(result.astype(np.uint8), result_path)
    print(end - start)
    # print("Testing [%d] success, Testing time is [%f]" % (i, end-start))
