import random
import functools

import torch
from torchvision.transforms.functional import rotate


def rotates(x, reverse=-1):
    if reverse == -1:
        x0 = torch.flip(x, [2])
        x1 = torch.flip(x, [3])
        x2 = torch.flip(x0, [3])

        x3 = torch.rot90(x, 1, [2,3])
        x4 = torch.flip(x3, [2])
        x5 = torch.flip(x3, [3])
        x6 = torch.flip(x4, [3])
        return x, x0, x1, x2, x3, x4, x5, x6
    else:
        cases = {
            0: x,
            1: torch.flip(x, [2]),
            2: torch.flip(x, [3]),
            3: torch.flip(torch.flip(x, [3]), [2]),
            4: torch.rot90(x, -1, [2,3]),
            5: torch.rot90(torch.flip(x, [2]), -1, [2,3]),
            6: torch.rot90(torch.flip(x, [3]), -1, [2,3]),
            7: torch.rot90(torch.flip(torch.flip(x, [3]), [2]), -1, [2,3]),
        }
        return cases[reverse]


class Defense:
    def __init__(self, num=10):
        self.defence_name = 'self_ensemble'
        self.codec = None
        self.best_idx = -1

    def preprocess(self, image):
        xs = rotates(image)
        best_x = image

        best_mse, self.best_idx = float("inf"), 0
        for i, x in enumerate(xs):
            x_hat = self.codec(x)['x_hat']
            if x_hat is not None:
                mse = torch.mean((x - x_hat) ** 2)
                if mse < best_mse:
                    self.best_idx, best_mse, best_x, best_x_hat = i, mse, x, x_hat
        return best_x

    def postprocess(self, image):
        return torch.clamp(rotates(image, reverse=self.best_idx), min=0., max=1.)
