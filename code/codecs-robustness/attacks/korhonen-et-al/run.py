"""

This module implements a spatial-activity-guided adversarial attack on learned image compression models.  
By computing a spatial activity map from the luminance channel (via Sobel filtering and dilation),  
gradient updates to the compressed image are weighted to concentrate perturbations in textured regions,  
improving imperceptibility while deceiving downstream classifiers or codecs.

Paper
-----
Jari Korhonen and Junyong You.  
“Adversarial Attacks against Blind Image Quality Assessment Models.”  
In _QoEVMA ’22: Proceedings of the 2nd Workshop on Quality of Experience  
in Visual Multimedia Applications, ACM, Lisbon, Portugal, October 10–14 2022.  
https://doi.org/10.1145/3552469.3555715
"""

import torch

from torch.autograd import Variable
from fgsm_evaluate import test_main 
from codec_losses import loss_func

import numpy as np

import cv2
from scipy import ndimage
from torchvision import transforms

def rgb2ycbcr(im_rgb):
    im_rgb = im_rgb.astype(np.float32)
    im_ycrcb = cv2.cvtColor(im_rgb, cv2.COLOR_RGB2YCR_CB)
    im_ycbcr = im_ycrcb[:,:,(0,2,1)].astype(np.float32)
    im_ycbcr[:,:,0] = (im_ycbcr[:,:,0]*(235-16)+16)/255.0
    im_ycbcr[:,:,1:] = (im_ycbcr[:,:,1:]*(240-16)+16)/255.0
    return im_ycbcr

def makeSpatialActivityMap(im):
  im = im.cpu().detach().permute(0, 2, 3, 1).numpy()[0]
  #H = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8  
  im = rgb2ycbcr(im)
  im_sob = ndimage.sobel(im[:,:,0])
  im_zero = np.zeros_like(im_sob)
  im_zero[1:-1, 1:-1] = im_sob[1:-1, 1:-1]

  maxval = im_zero.max()

  if maxval == 0:
    im_zero = im_zero + 1
    maxval = 1
  
  im_sob = im_zero /maxval

  DF = np.array([[0, 1, 1, 1, 0], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [1, 1, 1, 1, 1], 
        [0, 1, 1, 1, 0]]).astype('uint8')
  
  out_im = cv2.dilate(im_sob, DF)
  return out_im
          

def attack(compress_image, model=None, device='cpu', is_jpegai=False, loss_func=None, loss_func_name='undefined', 
          lr=0.02, n_iters = 50):

    input_range = 1

    if hasattr(model, 'input_range'):
        input_range = model.input_range

    if is_jpegai:
        print(f'Attacking JPEGAI model, input_range: ', input_range)

    scale = 255 if input_range == 1 else 1

    sp_map = makeSpatialActivityMap(compress_image * scale)
    sp_map = sp_map / 255
    sp_map = transforms.ToTensor()(sp_map.astype(np.float32))
    sp_map = sp_map.unsqueeze_(0)
    sp_map = sp_map.to(device)

    src_image = compress_image.clone().to(device)
    src_image = torch.autograd.Variable(src_image, requires_grad=True)
    decompr_src = model(src_image.to(device))['x_hat']
    decompr_src = torch.autograd.Variable(decompr_src.clone().to(device), requires_grad=True)
    
    compress_image = Variable(compress_image, requires_grad=True)
    opt = torch.optim.Adam([compress_image], lr = lr)
    
    for i in range(n_iters):
        compression_results = model(compress_image.to(device))
        decompr, likelihoods, bpp_loss = compression_results['x_hat'], compression_results['likelihoods'], compression_results['bpp']
        loss = loss_func(src_image, compress_image, decompr_src, decompr, bpp_loss, is_jpegai)
        loss.backward() 
        compress_image.grad = torch.nan_to_num(compress_image.grad)
        compress_image.grad *= sp_map
        opt.step()
        compress_image.data.clamp_(0., input_range)
        opt.zero_grad()

    res_image = (compress_image).data.clamp_(min=0, max=input_range)

    return res_image

if __name__ == "__main__":
    test_main(attack)

