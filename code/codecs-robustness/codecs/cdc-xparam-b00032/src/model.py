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
from src.modules.denoising_diffusion import GaussianDiffusion
from src.modules.unet import Unet
from src.modules.compress_modules import ResnetCompressor
from ema_pytorch import EMA

class CodecModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        
        self.device = device
        self.n_denoise_step = 5
        self.lpips_weight = 0.0
        self.path = 'image-l2-use_weight5-vimeo-d64-t8193-b0.0032-x-cosine-01-float32-aux0.0_2.pt'
        self.gamma = 0.8


        denoise_model = Unet(
            dim=64,
            channels=3,
            context_channels=64,
            dim_mults=[1,2,3,4,5,6],
            context_dim_mults=[1,2,3,4],
            embd_type="01",
        )

        context_model = ResnetCompressor(
            dim=64,
            dim_mults=[1,2,3,4],
            reverse_dim_mults=[4,3,2,1],
            hyper_dims_mults=[4,4,4],
            channels=3,
            out_channels=64,
        )

        diffusion = GaussianDiffusion(
            denoise_fn=denoise_model,
            context_fn=context_model,
            ae_fn=None,
            num_timesteps=8193,
            loss_type="l2",
            lagrangian=0.0032,
            pred_mode="x",
            aux_loss_weight=self.lpips_weight,
            aux_loss_type="lpips",
            var_schedule="cosine",
            use_loss_weight=True,
            loss_weight_min=5,
            use_aux_loss_weight_schedule=False,
        )
        loaded_param = torch.load(
            self.path,
            map_location=lambda storage, loc: storage,
        )
        ema = EMA(diffusion, beta=0.999, update_every=10, power=0.75, update_after_step=100)
        ema.load_state_dict(loaded_param["ema"])
        self.diffusion = ema.ema_model
        self.diffusion.to(self.device)
        self.diffusion.eval()
    
    def forward(self, image, return_bpp=False):
        self.eval()
        compressed, bpp = self.diffusion.compress(
                image * 2.0 - 1.0,
                sample_steps=self.n_denoise_step,
                bpp_return_mean=True,
                init=torch.randn_like(image) * self.gamma
        )
        compressed = compressed.clamp(-1, 1) / 2.0 + 0.5
        return {'x_hat': compressed, 'likelihoods':None, 'bpp':bpp}