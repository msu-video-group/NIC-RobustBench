"""
Summary
-------
Implements a randomized, gradient-based adversarial attack for learned image compression models.  
An initial random perturbation (p) within the ε-ball is iteratively refined by computing the attack loss gradient w.r.t. p and stepping in its signed direction, 
while clipping to ensure the total perturbation stays within the allowed bound. 
The loss combines a user-provided distortion metric on the decompressed outputs and a regularization term on the perturbation magnitude.

Source code:
    https://github.com/MadryLab/mnist_challenge 
     
Paper:
Aleksander Madry, Aleksandar Makelov, Ludwig Schmidt, Dimitris Tsipras, and Adrian Vladu: 
"Towards Deep Learning Models Resistant to Adversarial Attacks," In ICLR, 208
"""


import torch

from torch.autograd import Variable
from fgsm_evaluate import test_main 

def attack(compress_image, model=None, device='cpu', is_jpegai=False, loss_func=None, loss_func_name='undefined',
           alpha = 2/255, eps = 7 / 255, iters = 12):

    input_range = 1

    if hasattr(model, 'input_range'):
        input_range = model.input_range

    if is_jpegai:
        print(f'Attacking JPEGAI model, input_range: ', input_range)

    src_image = compress_image.clone().to(device)
    src_image = torch.autograd.Variable(src_image, requires_grad=True)
    decompr_src = model(src_image.to(device))['x_hat']
    decompr_src = torch.autograd.Variable(decompr_src.clone().to(device), requires_grad=True)
    compress_image = torch.autograd.Variable(compress_image.clone().to(device), requires_grad=True)
    
    p = torch.randn_like(compress_image).to(device) * eps * input_range
    p = torch.clamp(p, -eps * input_range, eps * input_range)
    p = torch.autograd.Variable(p, requires_grad=True)

    for i in range(iters):
        res = compress_image + p
        res.data.clamp_(0., input_range)

        compression_results = model(res.to(device))
        decompr, likelihoods, bpp_loss = compression_results['x_hat'], compression_results['likelihoods'], compression_results['bpp']

        loss = loss_func(src_image, res, decompr_src, decompr, bpp_loss, is_jpegai)
        loss_dist = torch.nn.MSELoss(reduction='mean')(p, torch.zeros_like(p).to(device)) / (input_range * input_range)
        loss.backward() 

        g = p.grad
        g = torch.sign(g)
        p.data -= alpha * g * input_range # * 255 for jpeg
        p.data.clamp_(-eps * input_range, +eps * input_range)  # * 255 for jpeg
        p.grad.zero_()

    res_image = compress_image + p
    res_image = (res_image).data.clamp_(min=0, max=input_range)

    return res_image

if __name__ == "__main__":
    test_main(attack)
