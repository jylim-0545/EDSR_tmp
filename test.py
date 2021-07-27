import torch
import numpy as np
import torch.nn as nn
from model.model import SingleNetwork
import data.dataset as ds
from tqdm import tqdm
from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from metrics.niqe import calculate_niqe
import torch.optim.lr_scheduler as lrs
import os
import imageio
import time

n_b = 24
n_f = 64
reduction = 16
scale = 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
train_dir = "/ssd/div2k/test_LR_X2/"
gt_dir = "/ssd/div2k/test_HR/"
train_dir = "/ssd/div2k/overfitting/LR_X4_1"
gt_dir = "/ssd/div2k/overfitting/HR_1"

datasets = ds.DS(train_dir, gt_dir, scale=scale, test=True)

data_loader = torch.utils.data.DataLoader(dataset = datasets, batch_size = 1, shuffle = False, pin_memory=True, num_workers = 0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


save_path = 'result/tmp'


model = SingleNetwork(num_block=n_b, num_feature=n_f, num_channel=3, scale=scale, bias=True)

model_path = 'tmp.pth'
model.load_state_dict(torch.load(model_path), strict=True)
model.eval()
model.cuda()


metric_psnr = []
metric_ssim = []

for idx, (img, gt, name) in tqdm(enumerate(data_loader), total = len(data_loader)):
 with torch.no_grad():
   img = img.to(device)
   gt = gt.data.squeeze().float().cpu().clamp_(0, 255.0).numpy()
   output = model(img)
   output = output.data.squeeze().float().cpu().clamp_(0, 1.0).numpy()
   output = output*255.0
 metric_psnr.append(calculate_psnr(gt, output, 0, input_order = 'CHW'))
 metric_ssim.append(calculate_ssim(gt, output, 0, input_order = 'CHW'))
 #output = np.transpose(output, (1,2,0))
 #idx = idx + 1
 #idx = str(idx).zfill(4)
 #imageio.imwrite('{}/{}.png'.format(save_path, idx), output.astype(np.uint8))

psnr = sum(metric_psnr) / len(metric_psnr)
ssim = sum(metric_ssim) / len(metric_ssim)


 
print(psnr, ssim)
