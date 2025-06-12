import torch

from torchvision.io import decode_jpeg, encode_jpeg

def apply_jpeg(x: torch.Tensor, quality: int) -> torch.Tensor:
    return decode_jpeg(encode_jpeg(x, quality))

class Defense:
    def __init__(self, q=50):
        self.defence_name = 'jpeg'
        self.q = q

    def __call__(self, image):
        res = []
        device = image.device
        for img in image:
            res.append(apply_jpeg((img.cpu()*255).type(torch.uint8), self.q)[None, :])
        image = torch.cat(res)
        image = (image/255).float().to(device)
        return image
   
