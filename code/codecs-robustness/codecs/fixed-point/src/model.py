"""
Module: CodecModel

A PyTorch wrapper for the LRS Fixed-Inference Super-Resolution model, which uses a pretrained float network and a fixed-inference pipeline to upscale images by a factor of 1.5 via learned residual mapping.

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

import src.scripts.model as model
from src.scripts.test_single import LRS_Fixed_Inference

class CodecModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        model_path = 'models/lrs_float.pth'        
        self.device = device
        self.sr_model = model.SuperResolution().to(device)
        self.sr_model.load_state_dict(torch.load(model_path))
        self.sr_model.eval()

    
    def forward(self, image, return_bpp=False):
        output = torch.nn.functional.interpolate(image, (int(image.shape[2]/1.5), int(image.shape[3]/1.5)))
        output = LRS_Fixed_Inference(output, self.sr_model)
        output[output >= 1] = 1
        output[output <= 0] = 0
        output = torch.nn.functional.interpolate(output, (int(image.shape[2]), int(image.shape[3])))
        return {'x_hat': output, 'likelihoods':None, 'bpp':1}