import random, sys, os
import torch
import torch.nn as nn
import torch.utils.data as data

import model.ops as ops

#(Multi-resolution)Nas_Net
class MultiNetwork(nn.Module):
    def __init__(self, config, act=nn.ReLU(True)):
        super(MultiNetwork, self).__init__()

        self.networks = nn.ModuleList()
        self.scale_dict = {}

        #set model parameter (e.g., layer, channel)
        for iteration, scale in enumerate(config):
            self.networks.append(SingleNetwork(num_block=config[scale]['block'], num_feature=config[scale]['feature'], num_channel=3, scale=scale, bias=True, act=act))
            self.scale_dict[scale] = iteration

        self.target_scale = None

    def setTargetScale(self, scale):
        assert scale in self.scale_dict.keys()
        self.target_scale= scale

    def forward(self, x):
        assert self.target_scale != None
        x = self.networks[self.scale_dict[self.target_scale]].forward(x)
        return x

#(Single-resolution)NAS_Net
class SingleNetwork(nn.Module):
    def __init__(self, num_block, num_feature, num_channel, scale, bias=True, act=nn.ReLU(True)):
        super(SingleNetwork, self).__init__()
        self.num_block = num_block
        self.num_feature = num_feature
        self.num_channel = num_channel
        self.scale = scale

        assert self.scale in [1,2,3,4,5]

        self.head = nn.Sequential(*[nn.Conv2d(in_channels=self.num_channel, out_channels=self.num_feature, kernel_size=3, stride=1, padding=1, bias=bias)])

        self.body = nn.ModuleList()
        for _ in range(self.num_block):
            modules_body = [ops.ResBlock(self.num_feature, bias=bias, act=act)]
            self.body.append(nn.Sequential(*modules_body))

        body_end = []
        body_end.append(nn.Conv2d(in_channels=self.num_feature, out_channels=self.num_feature, kernel_size=3, stride=1, padding=1, bias=bias))
        self.body_end = nn.Sequential(*body_end)

        if self.scale > 1:
            self.upscale= nn.Sequential(*ops.Upsampler(self.scale, self.num_feature, bias=bias))

        self.tail = nn.Sequential(*[nn.Conv2d(in_channels=self.num_feature, out_channels=self.num_channel, kernel_size=3, stride=1, padding=1, bias=bias)])

    def forward(self, x):
        idx = self.num_block

        #feed-forward part
        x = self.head(x)
        res = x

        for i in range(idx):
            res = self.body[i](res)
        res = self.body_end(res)
        res += x

        if self.scale > 1:
            x = self.upscale(res)
        else:
            x = res

        x = self.tail(x)

        return x
