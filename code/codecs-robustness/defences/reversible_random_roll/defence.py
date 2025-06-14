import random

import torch


class Defense:
    def __init__(self):
        self.defence_name = 'reversible_random_roll'

    def preprocess(self, image):
        self.roll_axis = random.choice([2, 3])
        self.roll_size = random.randint(0, image.shape[self.roll_axis] - 1)
        return torch.roll(image, self.roll_size, dims=self.roll_axis)

    def postprocess(self, image):
        return torch.roll(image, -self.roll_size, dims=self.roll_axis)
