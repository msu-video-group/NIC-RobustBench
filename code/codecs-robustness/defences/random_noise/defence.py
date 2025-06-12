import torch
from torchvision import transforms
import numpy as np

class Defense:
    def __init__(self, sigma=0.01):
        self.defence_name = 'random-noise'
        self.sigma = sigma

    def __call__(self, image):
        return torch.clamp(image + self.sigma*torch.randn(image.shape, device=image.device), 0, 1)