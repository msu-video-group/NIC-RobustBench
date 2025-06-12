"""

This module implements an iterative, projection-based adversarial attack tailored for learned image compression models.  
Starting from a compressed input, the procedure repeatedly computes gradients of a user-supplied loss function (e.g., distortion, bit-rate, or perceptual metrics) w.r.t. the compressed representation, 
then projects each update to respect a maximal perturbation bound (ε) under the model’s input range. 
This yields an adversarially perturbed compressed image that maximizes the given loss while staying within the allowed distortion budget.


Paper:
https://pubmed.ncbi.nlm.nih.gov/18831621/
Wang, Zhou, and Eero P. Simoncelli,
"Maximum differentiation (MAD) competition: A methodology for comparing computational models of perceptual quantities," 
Journal of Vision 8.12 (2008): 8-8.
"""

import torch

from torch.autograd import Variable
from fgsm_evaluate import test_main 

def attack(compress_image, ref_image=None, model=None, metric_range=100, device='cpu', is_jpegai=False, loss_func=None, loss_func_name='undefined', 
           lr=0.2, eps = 7 / 255, iters = 25):

    input_range = 1

    if hasattr(model, 'input_range'):
        input_range = model.input_range

    if is_jpegai:
        print(f'Attacking JPEGAI model, input_range: ', input_range)

    init_image = compress_image.clone()
    compress_image = Variable(compress_image.to(device), requires_grad=True)
    init_image = Variable(init_image.to(device), requires_grad=False)

    src_image = compress_image.clone().to(device)
    src_image = torch.autograd.Variable(src_image, requires_grad=True)
    decompr_src = model(src_image.to(device))['x_hat']
    decompr_src = torch.autograd.Variable(decompr_src.clone().to(device), requires_grad=True)

    eps *= input_range # for jpeg

    for i in range(iters):
        compression_results = model(compress_image.to(device))
        decompr, likelihoods, bpp_loss = compression_results['x_hat'], compression_results['likelihoods'], compression_results['bpp']

        loss = loss_func(src_image, compress_image, decompr_src, decompr, bpp_loss, is_jpegai)
        loss.backward()

        g2 = compress_image.grad.clone()
        compress_image.grad.zero_()
    
        if (i < 1):
            pg = g2.clone()
            if (pg != 0).nonzero().shape[0] == 0:
                pg = torch.ones_like(pg).to(device)
        else:
            loss = ((compress_image - init_image) ** 2).mean() ** 0.5
            loss.backward()
            g1 = compress_image.grad.clone()
            compress_image.grad.zero_()
            pg = g2 - (g1 * g2).sum() / (g1 * g1).sum() * g1
    
    
        pg = torch.sign(pg)
        compress_image.data -=  lr * pg * input_range
        compress_image.grad.zero_()

        cur_score = ((compress_image - init_image) ** 2).mean() ** 0.5

        while cur_score > eps:
            cur_score.backward() 
            g2 = torch.sign(compress_image.grad)
            compress_image.data -= 0.0005 * g2 * input_range
            compress_image.grad.zero_()
            compress_image.data.clamp_(0., input_range)
            cur_score = ((compress_image - init_image) ** 2).mean() ** 0.5
        
        compress_image.data.clamp_(0., input_range)
        compress_image.grad.zero_()

    res_image = (compress_image).data.clamp_(min=0, max=input_range)

    return torch.clamp(res_image, 0, input_range)

if __name__ == "__main__":
    test_main(attack)

