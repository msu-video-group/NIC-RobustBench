import random

import torch


class Defense:
    def __init__(self):
        self.defence_name = 'reversible_random_color_order'

    def preprocess(self, image):
        self.permute = [0, 1, 2]
        while self.permute == [0, 1, 2]:
            random.shuffle(self.permute)
        return image[:, self.permute]

    def postprocess(self, image):
        _, indices = torch.sort(torch.tensor(self.permute))
        return image[:, indices]
