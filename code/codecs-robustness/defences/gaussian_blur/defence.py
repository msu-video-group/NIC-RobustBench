import torch
from torchvision.transforms import GaussianBlur

class Defense:
    def __init__(self, kernel_size=3):
        self.defence_name = 'gaussian-blur'
        self.kernel_size = kernel_size
        self.transform = GaussianBlur(kernel_size, 0.3*((kernel_size-1)*0.5 - 1) + 0.8)

    def __call__(self, image):
        image = self.transform(image)
        return image
   
