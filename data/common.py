import random

import numpy as np
import cv2
import torch
from torchvision import transforms
from PIL import Image

def get_patch(img_in, img_tar, patch_size, scale, multi_scale=False):
    width, height = img_in.size
    height_ = random.randrange(0, height - patch_size + 1)
    width_ = random.randrange(0, width - patch_size + 1)
        
    img_in = img_in.crop((width_ , height_, width_ + patch_size, height_ + patch_size))
    img_tar = img_tar.crop((width_ * scale, height_ * scale, (width_ + patch_size) * scale, (height_ + patch_size) * scale))
        
    return img_in, img_tar


def augment(l, hflip=True, rot=True):
    hflip = hflip and random.random() < 0.5
    vflip = rot and random.random() < 0.5
    rot90 = rot and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            img = img.transpose(method = Image.FLIP_LEFT_RIGHT)
        if vflip:  # vertical
            img = img.transpose(method = Image.FLIP_TOP_BOTTOM)
        if rot90:
            img = img.rotate(90, expand=True)
        
        return img

    return [_augment(_l) for _l in l]
