# -*- coding: utf-8 -*-
from model import netD, netG
from WGAN import FusionGAN
import argparse
import os

parse = argparse.ArgumentParser()
parse.add_argument('--epoch', type=int, default=20)
parse.add_argument('--batch_size', type=int, default=4)
parse.add_argument('--image_size', type=int, default=132)
parse.add_argument('--label_size', type=int, default=132)
parse.add_argument('--learning_rate', type=float, default=2e-4)  # or 1e-4
parse.add_argument('--c_dim', type=int, default=1)
parse.add_argument('--scale', type=int, default=3)
parse.add_argument('--stride', type=int, default=14)
parse.add_argument('--checkpoint_dir', type=str, default="checkpoint")
parse.add_argument('--sample_dir', type=str, default="sample")
parse.add_argument('--summary_dir', type=str, default="log")
parse.add_argument('--is_train', type=bool, default=True)
parse.add_argument('--beta1', type=float, default=0.5)
parse.add_argument('--workers', type=int, default=2)
parse.add_argument('--gpu', type=bool, default=True)
opt = parse.parse_args()

if not os.path.exists(opt.checkpoint_dir):
    os.makedirs(opt.checkpoint_dir)
if not os.path.exists(opt.sample_dir):
    os.makedirs(opt.sample_dir)

srcnn = FusionGAN(image_size=132, label_size=132, batch_size=4,  c_dim=1,
                  checkpoint_dir="checkpoint", sample_dir="sample")

srcnn.train(opt, netD, netG)
