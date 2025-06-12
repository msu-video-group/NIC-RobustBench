import torch
from torchvision import transforms

class Defense:
    def __init__(self, angle_limit=15):
        self.defence_name = 'random-rotate'
        self.angle_limit = angle_limit


    def __call__(self, image):
        return transforms.RandomRotation(self.angle_limit)(image)
   
