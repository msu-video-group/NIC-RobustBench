"""
Module: CodecModel

A PyTorch wrapper for the High-Fidelity Generative Image Compression (HiFiC)
model. Loads the pretrained 'hific_low.pt' checkpoint via `prepare_model`,
and performs image compression and reconstruction in a single forward pass,
returning the reconstructed image and estimated bits-per-pixel.

The source code:
https://github.com/Justin-Tan/high-fidelity-generative-compression 

The paper:
Mentzer, Fabian, et al. "High-fidelity generative image compression," Advances in neural information processing systems 33, 2020
https://arxiv.org/abs/2006.09965
"""

import torch
import numpy as np
import torch
from src.compress import prepare_model, make_deterministic
import os

class CodecModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.device = device
        model_path = 'hific_med.pt'
        self.model, args = prepare_model(model_path)
        self.model.eval()
    
    def forward(self, image, return_bpp=False):
        self.model.eval()
        #image = 2*image-1
        #compressed_output = self.model.compress(image)
        #reconstruction = self.model.decompress(compressed_output)
        #q_bpp = compressed_output.total_bpp
        compressed_output = self.model.compression_forward(image)[0]
        return {'x_hat': compressed_output.reconstruction, 'likelihoods':None, 'bpp': compressed_output.n_bpp}