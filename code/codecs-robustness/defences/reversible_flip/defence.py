import torch


class Defense:
    def __init__(self, axes=[2, 3]):
        self.defence_name = 'reversible_flip'
        self.axes = axes

    def preprocess(self, image):
        return torch.flip(image, self.axes)

    def postprocess(self, image):
        return torch.flip(image, self.axes)
