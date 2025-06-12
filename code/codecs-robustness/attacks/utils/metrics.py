from skimage.metrics import peak_signal_noise_ratio, structural_similarity, mean_squared_error
import numpy as np
from read_dataset import to_numpy, to_torch
from pytorch_msssim import ms_ssim
import os
from torchvision.utils import save_image
import shlex
import subprocess
import torch
import pyiqa
import pandas as pd
import re, os, shutil, json

def PSNR(x, y):
    return peak_signal_noise_ratio(to_numpy(x), to_numpy(y), data_range=1)


def SSIM(x, y):
    ssim_sum = 0
    for img1, img2 in zip(to_numpy(x), to_numpy(y)):
        ssim = structural_similarity(img1, img2, channel_axis=2, data_range=1)
        ssim_sum += ssim
    return ssim_sum / len(x)

def MSE(x, y):
    return mean_squared_error(to_numpy(x), to_numpy(y))

def L_inf_dist(x, y):
    return np.max(np.abs(to_numpy(x).astype(np.float32) - to_numpy(y).astype(np.float32)))

def MAE(x, y):
    return np.abs(to_numpy(x).astype(np.float32) - to_numpy(y).astype(np.float32)).mean()

def MSSSIM(x, y):
    return ms_ssim(to_torch(x), to_torch(y), data_range=1.).item()

def vmaf(ref, dist):
    orig_path = os.path.join(os.getcwd(), "orig.png")
    save_image(ref, orig_path)
    dist_path = os.path.join(os.getcwd(), "dist.png")
    save_image(dist, dist_path)
    
    env = os.environ.copy()
    cmd = [
        "ffmpeg",
        "-hide_banner", 
        "-loglevel",
        "error",
        "-i", orig_path,
        "-i", dist_path,
        "-lavfi",
        "libvmaf="
        "model=version=vmaf_v0.6.1:log_fmt=json:log_path=tmp.json",
        "-f", "null",
        "-"
    ]

    proc = subprocess.run(
        cmd,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False
    )
    with open("tmp.json", 'r') as f:
        j = json.load(f)

    score = j['frames'][-1]['metrics']['vmaf']

    return torch.tensor(float(score))

class niqe(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.model = pyiqa.create_metric(
            'niqe',
            as_loss=True,
            device=device
        )
        self.lower_better = self.model.lower_better
        self.full_reference = False

    def forward(self, image, inference=False):
        try:
            return self.model(image)
        except torch._C._LinAlgError:
            return None