"""
Module: CodecModel

This module defines `CodecModel`, a PyTorch `nn.Module` wrapper for a CODEC_NAME  neural image codec. 
The source code:
https://github.com/InterDigitalInc/CompressAI

The paper:
https://interdigitalinc.github.io/CompressAI/zoo.html#bmshj2018
Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang, N. Johnston: 
“Variational Image Compression with a Scale Hyperprior”, Int Conf. on Learning Representations (ICLR), 2018.

"""

import math
import io
import torch
from torchvision import transforms
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt
from compressai.zoo import bmshj2018_hyperprior

class CodecModel(torch.nn.Module):
    """
    Neural image codec based on the bmshj2018 hyperprior prior model.

    This class loads a pretrained CompressAI model at quality level 1, moves it
    to the specified device, and exposes a uniform interface for compression
    and decompression. It also provides utilities to compute both the model’s
    internal bpp estimate and the empirical bpp from the encoded bitstream.

    Args:
        device (torch.device):  Target device (CPU or CUDA) for model execution.
    """
    
    def __init__(self, device):
        super().__init__()
        self.device = device
        self.model = bmshj2018_hyperprior(quality=2, pretrained=True).train().to(device)
    
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
        """
        Compress and reconstruct an image via denoising diffusion.

        Args:
            image (torch.Tensor): Input image tensor of shape (B, C, H, W),
                normalized to [0, 1].
            return_bpp (bool, optional): If True, include bits-per-pixel in output.

        Returns:
            dict: {
                'x_hat' (torch.Tensor): Reconstructed image tensor in [0, 1].
                'likelihoods': None (placeholder for compatibility).
                'real_bpp' (torch.Tensor or float): Estimated bits-per-pixel.
            }
        """
        
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