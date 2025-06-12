import torch
from torchvision import transforms

class Defense:
    def __init__(self, scale=0.5, mode='bilinear'):
        self.defence_name = 'upscale'
        self.scale = scale
        self.mode = mode

    def __call__(self, image):
        new_size = (torch.tensor(image.shape[-2:]) * self.scale)
        image = transforms.Resize((int(new_size[0]), int(new_size[1])))(image)
        resized_image = torch.nn.Upsample(size=(299, 299), mode=self.mode)(image)
        return resized_image
   
