import torch
import random

import math
from torchvision.transforms.functional import rotate


class Defense:
    def __init__(self):
        self.defence_name = 'reversible_random_rotate'

    def preprocess(self, image):
        self.angle = random.randint(0, 359)
        self.h, self.w = image.shape[-2:]
        image = torch.nn.functional.pad(image, [self.w//2, self.w//2, self.h//2, self.h//2])
        rotated_image_tensor = rotate(image, self.angle)
        return rotated_image_tensor

    def postprocess(self, image):
        reversed_rot = rotate(image, -self.angle)
        reversed_rot = reversed_rot[..., self.h//2:-self.h//2, self.w//2:-self.w//2]
        return reversed_rot
