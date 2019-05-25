# -*- coding: utf-8 -*-
"""
Scipy version > 0.18 is needed, due to 'mode' option from scipy.misc.imread function
"""

import os
import glob
import h5py
import scipy.misc
import scipy.ndimage
import argparse
import numpy as np
import torch
from torch import nn
import cv2

def read_data(path):
    """
    Read h5 format data file

    Args:
      path: file path of desired file
      data: '.h5' file format that contains train data values
      label: '.h5' file format that contains train label values
    """
    with h5py.File(path, 'r') as hf:
        data = np.array(hf.get('data'))
        return data


def preprocess(path, scale=3):
    """
    Preprocess single image file
      (1) Read original image as YCbCr format (and grayscale as default)
      (2) Normalize
      (3) Apply image file with bicubic interpolation

    Args:
      path: file path of desired file
      input_: image applied bicubic interpolation (low-resolution)
      label_: image with original resolution (high-resolution)
    """
    # 读到图片
    image = imread(path, is_grayscale=True)
    # 将图片label裁剪为scale的倍数
    label_ = modcrop(image, scale)

    # Must be normalized
    image = (image - 127.5) / 127.5
    label_ = (image - 127.5) / 127.5
    # 下采样之后再插值
    input_ = scipy.ndimage.interpolation.zoom(label_, (1. / scale), prefilter=False)
    input_ = scipy.ndimage.interpolation.zoom(input_, (scale / 1.), prefilter=False)

    return input_, label_


# 提取图片地址
def prepare_data(opt, dataset):
    """
    Args:
      dataset: choose train dataset or test dataset

      For train dataset, output data would be ['.../t1.bmp', '.../t2.bmp', ..., '.../t99.bmp']
    """
    filenames = os.listdir(dataset)
    data_dir = os.path.join(os.getcwd(), dataset)
    data = glob.glob(os.path.join(data_dir, "*.bmp"))  # 查找bmp文件，存入列表中
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    # 将图片按序号排序
    data.sort(key=lambda x: int(x[len(data_dir) + 1:-4]))
    # print(data)
    return data

# 将文件以h5py的形式保存
def make_data(opt, data, data_dir):
    """
    Make input data as h5 file format
    Depending on 'is_train' (flag value), savepath would be changed.
    """
    savepath = os.path.join('.', os.path.join('checkpoint', data_dir, 'train.h5'))
    if not os.path.exists(os.path.join('.', os.path.join('checkpoint', data_dir))):
        os.makedirs(os.path.join('.', os.path.join('checkpoint', data_dir)))
    with h5py.File(savepath, 'w') as hf:
        hf.create_dataset('data', data=data)


# 将图像数据保存为文件
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


# 根据尺度的倍数对图像进行裁剪，保证缩放后没有余数
def modcrop(image, scale=3):
    """
    To scale down and up the original image, first thing to do is to have no remainder while scaling operation.

    We need to find modulo of height (and width) and scale factor.
    Then, subtract the modulo from height (and width) of original image size.
    There would be no remainder even after scaling operation.
    """
    if len(image.shape) == 3:
        h, w, _ = image.shape
        h = h - np.mod(h, scale)
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w, :]
    else:
        h, w = image.shape
        h = h - np.mod(h, scale)  # 去除尺度整数倍多余的部分
        w = w - np.mod(w, scale)
        image = image[0:h, 0:w]
    return image


def input_setup(opt, data_dir, index=0):
    """
    Read image files and make their sub-images and saved them as a h5 file format.
    """
    # Load data path
    data = prepare_data(opt, dataset=data_dir)

    sub_input_sequence = []
    padding = int(abs(opt.image_size - opt.label_size) / 2)  # 6

    for i in range(len(data)):
        # input_, label_ = preprocess(data[i], opt.scale)
        input_ = (imread(data[i]) - 127.5) / 127.5

        if len(input_.shape) == 3:
            h, w, _ = input_.shape
        else:
            h, w = input_.shape
        # 按stride步长采样小patch
        for x in range(0, h - opt.image_size + 1, opt.stride):
            for y in range(0, w - opt.image_size + 1, opt.stride):
                sub_input = input_[x:x + opt.image_size, y:y + opt.image_size]
                # Make channel value
                sub_input = sub_input.reshape([opt.image_size, opt.image_size, 1])
                sub_input_sequence.append(sub_input)

    """
        len(sub_input_sequence) : the number of sub_input (33 x 33 x ch) in one image
        (sub_input_sequence[0]).shape : (33, 33, 1)
        """
    # Make list to numpy array. With this transform
    arrdata = np.asarray(sub_input_sequence)
    # print(arrdata.shape)
    make_data(opt, arrdata, data_dir)  # 保存数据

def imsave(image, path):
    return scipy.misc.imsave(path, image)


# 将一组图片进行组合
def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 1))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image
    return (img * 127.5 + 127.5)


# 通过滤波器滤波的方式求得梯度值
class gradient(nn.Module):
    def __init__(self):
        super(gradient, self).__init__()
        x_kernel = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        y_kernel = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        x_kernel = torch.FloatTensor(x_kernel).unsqueeze(0).unsqueeze(0)
        y_kernel = torch.FloatTensor(y_kernel).unsqueeze(0).unsqueeze(0)
        self.x_weight = nn.Parameter(data=x_kernel, requires_grad=False)
        self.y_weight = nn.Parameter(data=y_kernel, requires_grad=False)

    def forward(self, input):
        x_grad = torch.nn.functional.conv2d(input, self.x_weight, padding=1)
        y_grad = torch.nn.functional.conv2d(input, self.y_weight, padding=1)
        gradRes = torch.mean((x_grad + y_grad).float())
        return gradRes


# 根据输入数据的二范数对其进行归一化
# def l2_norm(input_x, epsilon=1e-12):
#     input_x_norm = input_x / (tf.reduce_sum(input_x ** 2) ** 0.5 + epsilon)
#     return input_x_norm
