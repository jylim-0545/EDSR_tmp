import torch
import numpy as np
import torch.nn as nn
import data.dataset as ds
from tqdm import tqdm
from metrics.psnr_ssim import calculate_psnr, calculate_ssim
from metrics.niqe import calculate_niqe
import torch.optim.lr_scheduler as lrs
import os
from model.model import SingleNetwork


n_b = 24
n_f = 64
scale = 4
bs = 1
learning_rate = 1e-4
betas = [0.9, 0.999]
num_epochs = 100
num_val = 1
decay = 1000000
gamma = 0.5
num_iterations_per_epoch = 4000
weight_decay = 0

train_dir = "/ssd/div2k/overfitting/LR_X4_1"
gt_dir = "/ssd/div2k/overfitting/HR_1"

loss = nn.L1Loss()


net = SingleNetwork(num_block=n_b, num_feature=n_f, num_channel=3, scale=scale, bias=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)



#datasets = ds.DS(train_dir, gt_dir, scale=scale)
datasets = ds.DS(train_dir, gt_dir, scale=scale, length = bs*num_iterations_per_epoch)

opt = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
#opt.load_state_dict(ck['opt'])

scheduler = lrs.StepLR(opt, step_size = decay, gamma = gamma)
#scheduler = ck['scheduler']
data_loader = torch.utils.data.DataLoader(dataset = datasets, batch_size = bs, shuffle = True, pin_memory=True, num_workers = 0)



for epoch in range(num_epochs):
  loss_total = []

  net.train()
  for idx, (img, gt) in tqdm(enumerate(data_loader), total=len(data_loader)):

    img, gt =  img.to(device), gt.to(device)
    opt.zero_grad()
    output = net(img)
    ls = loss(output, gt)

    loss_total.append(ls.item())
    ls.backward()
    opt.step()
    scheduler.step()

  print("[Epoch %d] loss %f" % (epoch, sum(loss_total)/len(loss_total)))
  net.eval()
  total_psnr = []
  total_ssim = []
  
  with torch.no_grad():
    for i in range(num_val):
      img, gt = datasets.getRanditem()
      img = img.unsqueeze(0)
      img = img.to(device='cuda')
      output = net(img)
      output = output.data.squeeze().float().cpu().clamp_(0, 1.0).numpy()
      output = output*255.0   
      total_psnr.append(calculate_psnr(output, gt, 0, input_order = 'CHW'))
      total_ssim.append(calculate_ssim(output, gt, 0, input_order = 'CHW'))
  print("[val] psnr: %f / ssim: %f" % (sum(total_psnr)/num_val, sum(total_ssim)/num_val))

 # if epoch%save_period == (save_period-1):
  #torch.save(net.state_dict(), os.path.join('result', '{:d}.pth'.format(epoch)))

torch.save(net.state_dict(), os.path.join('PRE_0_DS_1.pth'))

#torch.save({'epoch' :epoch, 'model':net.state_dict(), 'opt': opt.state_dict(), 'loss' : loss, 'scheduler': scheduler}, os.path.join('result', 'final.pth'.format(epoch)))  


