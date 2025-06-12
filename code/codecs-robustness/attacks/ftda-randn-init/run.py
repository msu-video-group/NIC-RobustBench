"""

This module implements an iterative, distortion-constrained adversarial attack against learned image
compression models with random initialisation. By maintaining a per-pixel perturbation tensor `p`, it alternates between
maximizing a user-supplied adversarial loss and minimizing an L₂ distortion up to a hard threshold.
Two Adam optimizers with distinct learning rates are used to step on the adversarial objective when
distortion is below a threshold, or on the distortion objective otherwise.  The attack supports both
targeted and untargeted modes via the provided `loss_func`.

Source code
https://github.com/tongxyh/ImageCompression_Adversarial

Paper
-----
Tong Chen and Zhan Ma,
"Toward Robust Neural Image Compression: Adversarial Attack and Model Finetuning,"
IEEE Transactions on Circuits and Systems for Video Technology 33, 12 (Dec. 2023), 7842–7856.
"""

import torch

from torch.autograd import Variable
from fgsm_evaluate import test_main 

def attack(compress_image, model=None, device='cpu', is_jpegai=False, loss_func=None, loss_func_name='undefined',
           lr = 0.2, THRESH = 0.0006, iters = 100, lr_d = 0.1):

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
    p = torch.clamp(torch.randn_like(compress_image).to(device) * THRESH * input_range, -2 * THRESH * input_range, 2 * THRESH * input_range)
    p = torch.autograd.Variable(p, requires_grad=True)
    opt = torch.optim.Adam([p], lr = lr)
    opt_d = torch.optim.Adam([p], lr = lr_d)

    for i in range(iters):
        opt.zero_grad()
        opt_d.zero_grad()
        res = compress_image + p
        res = torch.clamp(res, 0, input_range)

        compression_results = model(res.to(device))
        decompr, likelihoods, bpp_loss = compression_results['x_hat'], compression_results['likelihoods'], compression_results['bpp']
        loss_adv = loss_func(src_image, res, decompr_src, decompr, bpp_loss, is_jpegai)
        loss_dist = torch.nn.MSELoss(reduction='mean')(p, torch.zeros_like(p).to(device)) / (input_range * input_range)
        print(loss_dist, loss_adv, res.dtype)

        if loss_dist < THRESH:
            print("UP")
            cur_loss = loss_adv
            optim = opt
        else:
            print("DOWN")
            cur_loss = 10000 * loss_dist
            optim = opt_d
        
        cur_loss.backward() 
        optim.step()

        if i == iters - 1:
            loss_dist = 100 * torch.nn.MSELoss(reduction='mean')(p, torch.zeros_like(p).to(device)) / (input_range * input_range)
            k = 0

            while loss_dist > 100 * THRESH and k < 1000:
                k += 1
                opt_d.zero_grad()
                loss_dist.backward()
                opt_d.step()
                loss_dist = 100 * torch.nn.MSELoss(reduction='mean')(p, torch.zeros_like(p).to(device)) / (input_range * input_range)
                print("LESS", loss_dist)

    res_image = compress_image + p
    res_image = (res_image).data.clamp_(min=0, max=input_range)

    return res_image

if __name__ == "__main__":
    test_main(attack)
