import torch


class Defense:
    def __init__(self, npp=16):
        self.defence_name = 'color-quantization'
        self.npp = npp

    def __call__(self, image):
        npp_int = self.npp - 1
        if image.max() <= 1:
            npp_int = npp_int*255
            
        x_int = torch.round(image * npp_int)
        x_float = x_int / npp_int
        return x_float
   
