"""
Module: CodecModel

A PyTorch wrapper for the Joint Autoregressive and Hierarchical Priors (mbt2018)
model, which combines autoregressive and hierarchical entropy models for
state-of-the-art learned image compression via CompressAI.

The source code: 
https://github.com/InterDigitalInc/CompressAI
The paper: 
David Minnen, Johannes Ballé, and George Toderici, 
"Joint Autoregressive and Hierarchical Priors for Learned Image Compression," Advances in Neural Information Processing Systems, 2018
"""

import math
import io
import torch
from torchvision import transforms
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
from compressai.zoo import mbt2018_mean

class CodecModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = mbt2018_mean(quality=2, pretrained=True).train().to(device)
    
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