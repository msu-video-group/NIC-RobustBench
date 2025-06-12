import torch
from torchvision import transforms

class Defense:
    def __init__(self, scale=0.5):
        self.defence_name = 'resize'
        self.scale = scale

    def __call__(self, image):
        new_size = (torch.tensor(image.shape[-2:]) * self.scale)
        image = transforms.Resize((int(new_size[0]), int(new_size[1])))(image)
        return image
   
