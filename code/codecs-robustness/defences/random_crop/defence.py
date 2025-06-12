import torch
from torchvision import transforms


class Defense:
    def __init__(self, size=128):
        self.defence_name = 'random-crop'
        self.size = size

    def __call__(self, image):
        return transforms.RandomCrop(self.size)(image)
