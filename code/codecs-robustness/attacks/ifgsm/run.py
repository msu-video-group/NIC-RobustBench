"""

This module implements the Basic Iterative Method (BIM), a multi-step, L_inf-bounded adversarial attack
against learned image compression models. At each iteration, a small step is taken in the direction of the sign
of the gradient of a loss, and the total perturbation is clipped to lie within an epsilon-ball.

Paper
-----
Alexey Kurakin, Ian J. Goodfellow, and Samy Bengio.  
“Adversarial Examples in the Physical World.”  
5th International Conference on Learning Representations (ICLR) Workshop Track, 2017.  
https://arxiv.org/abs/1607.02533

"""


import torch

from torch.autograd import Variable
from fgsm_evaluate import test_main 

def attack(compress_image, model=None, device='cpu', is_jpegai=False, loss_func=None, loss_func_name='undefined',
           alpha = 2/255, eps = 7 / 255, iters = 12):

    print(f'LOSS {loss_func_name}')
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
    mult = 1

    if input_range == 1:
        mult = 1/255.0

    p = torch.ones_like(compress_image).to(device) * mult
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
