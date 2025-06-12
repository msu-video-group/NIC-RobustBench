"""
Baseline method. This module adds random noise to input images.


"""


import torch

from torch.autograd import Variable
from fgsm_evaluate import test_main 


def attack(compress_image, model=None, device='cpu', is_jpegai=False, loss_func=None, loss_func_name='undefined', eps=7/255, var=7/255):

    input_range = 1

    if hasattr(model, 'input_range'):
        input_range = model.input_range

    if is_jpegai:
        print(f'Attacking JPEGAI model, input_range: ', input_range)

    return torch.clamp(compress_image + torch.clamp(torch.randn_like(compress_image) * var * input_range, -eps * input_range, eps * input_range), 0, input_range)

if __name__ == "__main__":
    test_main(attack)