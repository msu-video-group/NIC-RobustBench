import torch
from torchvision import transforms


class Defense:
    def __init__(self):
        self.defence_name = 'no-defence'
        pass

    def __call__(self, image):
        return image
