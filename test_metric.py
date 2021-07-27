
import glob
import cv2
import numpy as np
import torch
import os
from tqdm import tqdm
import time


from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from metrics.niqe import calculate_niqe

path_o = "/home/jin0545/tmp/RCAN/result/"
path_gt = "/ssd/div2k/DIV2K_train_HR/"

o_list = sorted(os.listdir(path_o))
gt_list = sorted(os.listdir(path_gt))

assert len(o_list) == len(gt_list)


metric_psnr = []
metric_ssim = []
metric_niqe = []

for i in tqdm(range(len(o_list)), total = len(o_list)):  
  img_o = cv2.imread(path_o + '/' + o_list[i], cv2.IMREAD_COLOR)
  img_gt = cv2.imread(path_gt + '/' + gt_list[i], cv2.IMREAD_COLOR)
  metric_psnr.append(calculate_psnr(img_gt, img_o, 0))
  metric_ssim.append(calculate_ssim(img_gt, img_o, 0))
  metric_niqe.append(calculate_niqe(img_o, 0))

psnr = sum(metric_psnr) / len(metric_psnr)
ssim = sum(metric_ssim) / len(metric_ssim)
niqe = sum(metric_niqe) / len(metric_niqe)

  
print(psnr, ssim, niqe)

