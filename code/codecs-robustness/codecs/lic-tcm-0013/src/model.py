"""
Module: CodecModel

A PyTorch wrapper for the Transformer-CNN Mixture (TCM) learned image compression model,
which integrates parallel CNN and transformer blocks with a channel-wise entropy model
for state-of-the-art rate–distortion performance across multiple resolutions.

The source code: 
https://github.com/jmliu206/LIC_TCM
The paper: Jinming Liu, Heming Sun, Jiro Katto, 
"Learned Image Compression with Mixed Transformer-CNN Architectures," CVPR, 2023
"""

import math
import io
import torch
from torchvision import transforms
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from src.models import TCM

def pad(x, p):
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)

def crop(x, padding):
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )

class CodecModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.device = device
        self.net = TCM(config=[2,2,2,2,2,2], head_dim=[8, 16, 32, 32, 16, 8], drop_path_rate=0.0, N=64, M=320)
        self.net = self.net.to(device)
        self.net.eval()

        dictory = {}
        checkpoint = torch.load('LIC-TCM-64-0.013.pth.tar', map_location=device)
        for k, v in checkpoint["state_dict"].items():
            dictory[k.replace("module.", "")] = v
        self.net.load_state_dict(dictory)
        self.net.update()
    
    def forward(self, image, return_bpp=False):
        self.eval()
        p = 128
        x_padded, padding = pad(image, p)
        out_dec = self.net.forward(x_padded)
        compressed = crop(out_dec["x_hat"], padding)
        num_pixels = image.size(0) * image.size(2) * image.size(3)
        bpp_loss = torch.sum( -torch.log2(out_dec['likelihoods']['y']) )/num_pixels
        return {'x_hat': compressed, 'likelihoods':out_dec["likelihoods"], 'bpp':bpp_loss}