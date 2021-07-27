import torch
import torch.nn.functional as F
import os
import cv2
import numpy as np
import random
from torch.utils.data import Dataset
from tqdm import tqdm
from data.common import augment, get_patch
from torchvision.transforms import Compose, ToTensor

from PIL import Image


device = torch.device('cuda')
class DS(Dataset):
  def __init__(self, root_dir, gt_dir = None, patch_size = 48, scale = 1.0, length = None, test=False):
  
    self.img_list = sorted(os.listdir(root_dir))
    self.root_dir = root_dir
    self.gt_dir = gt_dir
    if gt_dir is not None:
      self.gt_list = sorted(os.listdir(gt_dir))
    self.test = test
    self.patch_size = patch_size
    self.scale = scale
    self.length = len(self.img_list)
    self._setup()
    if length == None:
      self.len_iteration = self.length
    else:
      self.len_iteration = length
    self.input_transform = Compose([ToTensor(),])
    
  def _setup(self):
      self.gt = []
      self.img = []
      print("Preparing dataset... ")
      for idx in tqdm(range(self.length)):
        img_name = self.root_dir + '/' + self.img_list[idx]
        img = Image.open(img_name)
        img.load()
        
        gt_name = self.gt_dir + '/' + self.gt_list[idx]
        gt = Image.open(gt_name)
        gt.load()
        
        self.img.append(img)
        self.gt.append(gt)

  def __len__(self):
    if self.test:
      return self.length
    return self.len_iteration

  def getRanditem(self):

    idx = random.randrange(0, self.length)

    img = self.img[idx]
    gt = self.gt[idx]
    img = self.input_transform(img)
    gt = np.array(gt) 
    gt = np.transpose(gt, (2, 0, 1))
    return img, gt
    
  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()
    idx = idx%self.length
    img = self.img[idx]

    if self.test:
      img = self.input_transform(img)
      gt = self.gt[idx]
      gt = np.array(gt) 
      gt = np.transpose(gt, (2, 0, 1))
      return img, gt, os.path.splitext(self.img_list[idx])[0]
    else:
      gt = self.gt[idx]

      img, gt = get_patch(img, gt, self.patch_size, self.scale)
      [img, gt] = augment([img, gt])
      img = self.input_transform(img)
      gt = self.input_transform(gt)     

      return img, gt