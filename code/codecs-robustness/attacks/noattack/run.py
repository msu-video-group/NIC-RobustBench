import torch
from torch.autograd import Variable
from fgsm_evaluate import test_main 

def attack(compress_image, model=None, device='cpu', is_jpegai=False, loss_func=None, loss_func_name='undefined',
            iters = 10):
    return compress_image
    
if __name__ == "__main__":
    test_main(attack)


