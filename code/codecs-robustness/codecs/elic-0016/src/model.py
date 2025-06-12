"""
Module: CodecModel

This module defines `CodecModel`, a PyTorch `nn.Module` wrapper for a
custom ELIC-based neural image codec. The wrapper initializes entropy coding,
moves the model to the specified device, and provides a simple interface
to compress and reconstruct images while computing bits-per-pixel (bpp).

The source code:
https://github.com/VincentChandelier/ELiC-ReImplemetation

The paper:
Dailan He, Ziming Yang, Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang:
"Elic: Efficient learned image compression with unevenly grouped space-
channel contextual adaptive coding," In Proceedings of the CVPR, 2022. 5718–5727.

"""

import math
import io
import torch
from torchvision import transforms
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
from src.Network import TestModel
from compressai.zoo import load_state_dict

class CodecModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.device = device
        model_path = 'ELIC_0016_ft_3980_Plateau.pth.tar'

        model_cls = TestModel()
        state_dict = load_state_dict(torch.load(model_path))

        self.model = model_cls.from_state_dict(state_dict).eval()

        self.model.update(True) # initialize entropy coding
        self.model.to(device)
    
    def forward(self, image, return_bpp=False):
        #string = self.model.forward(image)
        #print(string)
        self.eval()
        out = self.model(image)
        num_pixels = image.shape[-1] * image.shape[-2]
        bpp_loss = torch.sum( -torch.log2(out['likelihoods']['y']) ) / num_pixels
        out['bpp'] = bpp_loss
        return out