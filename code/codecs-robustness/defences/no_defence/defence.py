import torch


class Defense:
    def __init__(self):
        self.defence_name = 'no-defence'

    def preprocess(self, image):
        return image

    def postprocess(self, image):
        return image
