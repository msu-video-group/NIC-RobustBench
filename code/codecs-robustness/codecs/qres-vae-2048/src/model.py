"""
Module: CodecModel

A PyTorch wrapper for the Quantized Hierarchical VAE (QRes-VAE) learned image compression model.

The source code: 
https://github.com/duanzhiihao/lossy-vae

The paper: Zhihao Duan, Ming Lu, Zhan Ma, Fengqing Zhu, 
"Lossy Image Compression with Quantized Hierarchical VAEs," WACV, 2023
"""

import math
import io
import torch
from torchvision import transforms
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
from lvae import get_model

class CodecModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.device = device
        self.model = get_model('qres34m', lmb=2048, pretrained=True) # weights are downloaded automatically
        # self.model.eval()
        self.model.compress_mode(True) # initialize entropy coding
        self.model.to(device)
    
    def forward(self, image, return_bpp=False):
        #string = self.model.forward(image)
        #print(string)
        # self.eval()
        out = self.model.forward(image, return_rec=True)
        return {'x_hat': out['im_hat'], 'likelihoods':None, 'bpp':out['bppix']}