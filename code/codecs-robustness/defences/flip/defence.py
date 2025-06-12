import torch


class Defense:
    def __init__(self, axes=[2, 3]):
        self.defence_name = 'flip'
        self.axes = axes

    def __call__(self, image):
        return torch.flip(image, self.axes)
   
