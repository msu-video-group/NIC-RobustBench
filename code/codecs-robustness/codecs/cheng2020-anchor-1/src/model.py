"""
Module: CodecModel

This module defines `CodecModel`, a PyTorch `nn.Module` wrapper for a
denoising-diffusion–based neural image codec. It constructs a U-Net denoiser,
a ResNet compressor for context, and a Gaussian diffusion process with EMA
(Exponential Moving Average) weight loading from a pretrained checkpoint.
This variant uses only L2 reconstruction loss (LPIPS auxiliary weight set to 0).

The source code:
https://github.com/InterDigitalInc/CompressAI

The paper:
Yang R., Mandt S. "Lossy image compression with conditional diffusion models"
Advances in Neural Information Processing Systems. – 2023. – Т. 36. – С. 64971-64995.
"""

import math
import io
import torch
from torchvision import transforms
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
from compressai.zoo import cheng2020_anchor

class CodecModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = cheng2020_anchor(quality=1, pretrained=True).train().to(device)
    
    def calc_real_bpp(self, images):
        assert len(images.shape) == 4 # [b,c,h,w]
        data = self.model.compress(images)['strings']
        s = 0
        for string_arr in data:
            for encoded_str in string_arr:
                s+= len(encoded_str)
        num_pixels = images.shape[-1] * images.shape[-2] 
        num_pixels *= images.shape[0] # batch size
        return (float(s) * 8) / num_pixels
    
    def forward(self, image, return_bpp=False):
        out = self.model(image)
        num_pixels = image.shape[-1] * image.shape[-2]
        num_pixels *= image.shape[0] # batch size
        bpp_loss = torch.sum( -torch.log2(out['likelihoods']['y']) ) / num_pixels
        out['bpp'] = bpp_loss
        if return_bpp:
            with torch.no_grad():
                real_bpp = self.calc_real_bpp(image)
                out['real_bpp'] = real_bpp
        return out