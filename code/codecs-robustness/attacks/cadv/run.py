"""

This module implements a colorization-guided adversarial attack pipeline targeting learned image compression models
and downstream classifiers.  It leverages a pretrained automatic colorization network to generate sparse color hints
that, when optimized, produce imperceptible perturbations in the compressed domain.

Key components
--------------
class color_net(nn.Module):
    A U-Net–style colorization network based on VGG-16 convolutional blocks, with dilated convolutions to
    enlarge receptive fields.  Predicts a discrete color distribution (“classification”) and a continuous
    AB-regression, then upsamples via learned transpose convolutions and skip connections.

normalize(im)
    Normalize a torch tensor image using ImageNet mean/std per channel.

get_colorization_data(im, model, device)
    Convert RGB→LAB, cluster AB channels via K-Means, compute pixelwise entropy from the colorization model,
    select high-entropy clusters as hint locations, and prepare hint & mask tensors for optimization.

forward(model, classifier, opt, data)
    Run one forward pass: colorize with optimized hints, reconstruct RGB, classify, and return both.

compute_loss(opt, y, criterion)
    Compute (targeted or untargeted) classification loss against a specified target class.

compute_class(opt, y, num_labels=5)
    Return top-K softmax scores and corresponding class labels.

rgb2xyz, xyz2rgb, xyz2lab, lab2xyz, rgb2lab, lab2rgb
    Utility functions for color-space conversions between RGB, XYZ, and LAB.  

attack(compress_image, model=None, device='cpu', ...)
    Main adversarial loop: load a frozen colorization network, initialize hints and masks, step
    through gradient updates to maximize classification error (and optionally bitrate loss)
    on compressed reconstructions, and return a perturbed image tensor.


Source code
-----------
The colorization network architecture and pretrained weights are adapted from the PyTorch reimplementation
of the “Colorful Image Colorization” model:
    https://github.com/AI-secure/Big-but-Invisible-Adversarial-Attack

Paper
-----
Bhattad, Anand, Min Jin Chong, Kaizhao Liang, B. Li and David Alexander Forsyth,
“Unrestricted Adversarial Examples via Semantic Manipulation,” 
International Conference on Learning Representations (2019).
"""

import torch
import torch.nn as nn
from torch.nn import init
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np
from PIL import Image
import os
import math
from sklearn.cluster import KMeans
from scipy.ndimage.filters import median_filter, gaussian_filter
import torchvision.models as models
from torchvision.utils import save_image
from torch.nn import Module
from torch.autograd import Variable
from fgsm_evaluate import test_main 


class color_net(nn.Module):
    def __init__(self, input_nc=4, output_nc=2, norm_layer=nn.BatchNorm2d, use_tanh=True, classification=False):
        super(color_net, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.classification = classification
        use_bias = True

        # Conv1
        # model1=[nn.ReflectionPad2d(1),]
        model1=[nn.Conv2d(input_nc, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model1+=[norm_layer(64),]
        model1+=[nn.ReLU(True),]
        # model1+=[nn.ReflectionPad2d(1),]
        model1+=[nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model1+=[nn.ReLU(True),]
        model1+=[norm_layer(64),]
        # add a subsampling operation

        # Conv2
        # model2=[nn.ReflectionPad2d(1),]
        model2=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model2+=[norm_layer(128),]
        model2+=[nn.ReLU(True),]
        # model2+=[nn.ReflectionPad2d(1),]
        model2+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model2+=[nn.ReLU(True),]
        model2+=[norm_layer(128),]
        # add a subsampling layer operation

        # Conv3
        # model3=[nn.ReflectionPad2d(1),]
        model3=[nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model3+=[norm_layer(256),]
        model3+=[nn.ReLU(True),]
        # model3+=[nn.ReflectionPad2d(1),]
        model3+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model3+=[nn.ReLU(True),]
        model3+=[norm_layer(256),]
        # add a subsampling layer operation

        # Conv4
        # model47=[nn.ReflectionPad2d(1),]
        model4=[nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model4+=[norm_layer(512),]
        model4+=[nn.ReLU(True),]
        # model4+=[nn.ReflectionPad2d(1),]
        model4+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model4+=[nn.ReLU(True),]
        model4+=[norm_layer(512),]

        # Conv5
        # model47+=[nn.ReflectionPad2d(2),]
        model5=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model5+=[norm_layer(512),]
        model5+=[nn.ReLU(True),]
        # model5+=[nn.ReflectionPad2d(2),]
        model5+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model5+=[nn.ReLU(True),]
        model5+=[norm_layer(512),]

        # Conv6
        # model6+=[nn.ReflectionPad2d(2),]
        model6=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        # model6+=[norm_layer(512),]
        model6+=[nn.ReLU(True),]
        # model6+=[nn.ReflectionPad2d(2),]
        model6+=[nn.Conv2d(512, 512, kernel_size=3, dilation=2, stride=1, padding=2, bias=use_bias),]
        model6+=[nn.ReLU(True),]
        model6+=[norm_layer(512),]

        # Conv7
        # model47+=[nn.ReflectionPad2d(1),]
        model7=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model7+=[norm_layer(512),]
        model7+=[nn.ReLU(True),]
        # model7+=[nn.ReflectionPad2d(1),]
        model7+=[nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model7+=[nn.ReLU(True),]
        model7+=[norm_layer(512),]

        # Conv7
        model8up=[nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=use_bias)]

        # model3short8=[nn.ReflectionPad2d(1),]
        model3short8=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]

        # model47+=[norm_layer(256),]
        model8=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # model8+=[norm_layer(256),]
        model8+=[nn.ReLU(True),]
        # model8+=[nn.ReflectionPad2d(1),]
        model8+=[nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model8+=[nn.ReLU(True),]
        model8+=[norm_layer(256),]

        # Conv9
        model9up=[nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model2short9=[nn.ReflectionPad2d(1),]
        model2short9=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above

        # model9=[norm_layer(128),]
        model9=[nn.ReLU(True),]
        # model9+=[nn.ReflectionPad2d(1),]
        model9+=[nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        model9+=[nn.ReLU(True),]
        model9+=[norm_layer(128),]

        # Conv10
        model10up=[nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1, bias=use_bias),]

        # model1short10=[nn.ReflectionPad2d(1),]
        model1short10=[nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=use_bias),]
        # add the two feature maps above

        # model10=[norm_layer(128),]
        model10=[nn.ReLU(True),]
        # model10+=[nn.ReflectionPad2d(1),]
        model10+=[nn.Conv2d(128, 128, kernel_size=3, dilation=1, stride=1, padding=1, bias=use_bias),]
        model10+=[nn.LeakyReLU(negative_slope=.2),]

        # classification output
        model_class=[nn.Conv2d(256, 529, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]

        # regression output
        model_out=[nn.Conv2d(128, 2, kernel_size=1, padding=0, dilation=1, stride=1, bias=use_bias),]
        if(use_tanh):
            model_out+=[nn.Tanh()]

        self.model1 = nn.Sequential(*model1)
        self.model2 = nn.Sequential(*model2)
        self.model3 = nn.Sequential(*model3)
        self.model4 = nn.Sequential(*model4)
        self.model5 = nn.Sequential(*model5)
        self.model6 = nn.Sequential(*model6)
        self.model7 = nn.Sequential(*model7)
        self.model8up = nn.Sequential(*model8up)
        self.model8 = nn.Sequential(*model8)
        self.model9up = nn.Sequential(*model9up)
        self.model9 = nn.Sequential(*model9)
        self.model10up = nn.Sequential(*model10up)
        self.model10 = nn.Sequential(*model10)
        self.model3short8 = nn.Sequential(*model3short8)
        self.model2short9 = nn.Sequential(*model2short9)
        self.model1short10 = nn.Sequential(*model1short10)

        self.model_class = nn.Sequential(*model_class)
        self.model_out = nn.Sequential(*model_out)

        self.upsample4 = nn.Sequential(*[nn.Upsample(scale_factor=4, mode='nearest'),])
        self.softmax = nn.Sequential(*[nn.Softmax(dim=1),])

    def forward(self, input_A, input_B, mask_B):
        conv1_2 = self.model1(torch.cat((input_A,input_B,mask_B),dim=1))
        conv2_2 = self.model2(conv1_2[:,:,::2,::2])
        conv3_3 = self.model3(conv2_2[:,:,::2,::2])
        conv4_3 = self.model4(conv3_3[:,:,::2,::2])
        conv5_3 = self.model5(conv4_3)
        conv6_3 = self.model6(conv5_3)
        conv7_3 = self.model7(conv6_3)
        conv8_up = self.model8up(conv7_3) + self.model3short8(conv3_3)
        conv8_3 = self.model8(conv8_up)

        if(self.classification):
            out_class = self.model_class(conv8_3)

            conv9_up = self.model9up(conv8_3.detach()) + self.model2short9(conv2_2.detach())
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2.detach())
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)
        else:
            out_class = self.model_class(conv8_3.detach())

            conv9_up = self.model9up(conv8_3) + self.model2short9(conv2_2)
            conv9_3 = self.model9(conv9_up)
            conv10_up = self.model10up(conv9_3) + self.model1short10(conv1_2)
            conv10_2 = self.model10(conv10_up)
            out_reg = self.model_out(conv10_2)

        return (out_class, out_reg)


MEAN = torch.Tensor([0.485, 0.456, 0.406])
STD = torch.Tensor([0.229, 0.224, 0.225])

def normalize(im):
    '''
    Normalize a given tensor using Imagenet statistics

    '''
    mean = MEAN.cuda() if im.is_cuda else MEAN
    std = STD.cuda() if im.is_cuda else STD

    if im.dim() == 4:
        im = im.transpose(1, 3)
        im = (im - mean) / std
        im = im.transpose(1, 3)
    else:
        im = im.transpose(0, 2)
        im = (im - mean) / std
        im = im.transpose(0, 2)

    return im



def get_colorization_data(im, model, device):
    '''
    Convert image to LAB, and choose hints given clusters
    '''
    data = {}
    data_lab = rgb2lab(im)
    data['L'] = data_lab[:,[0,],:,:]
    data['AB'] = data_lab[:,1:,:,:]
    hints = data['AB'].clone()
    mask = torch.full_like(data['L'], -.5)
    n_clusters = 8
    k = 4
    hint = 50

    with torch.no_grad():
        # Get original classification
        #target = classifier(normalize(im)).argmax(1)
        #print(f'Original class: {target.item()}, {opt.idx2label[target]}')

        N,C,H,W = hints.shape
        # Convert hints to numpy for clustering
        np_hints = hints.squeeze(0).cpu().numpy()
        np_hints = gaussian_filter(np_hints, 3).reshape([2,-1]).T
        #hints = median_filter(hints, size=10).reshape([2,-1]).T
        hints_q = KMeans(n_clusters=n_clusters).fit_predict(np_hints)

        # Calculate entropy based on colorization model. Use 0 hints for this
        logits, reg = model(data['L'], torch.zeros_like(hints).to(device), torch.full_like(mask, -.5).to(device))
        hints_distr = F.softmax(logits, dim=1).clamp(1e-8).squeeze(0)
        entropy = (-1 * hints_distr * torch.log(hints_distr)).sum(0)

        # upsample to original resolution
        entropy = F.interpolate(entropy[None,None,...], scale_factor=4, mode='nearest').squeeze().view(-1)

        # Calculate mean entropy of each clusters
        entropy_cluster = np.zeros(n_clusters)
        for i in range(n_clusters):
            class_entropy = entropy[(hints_q == i).nonzero()]
            entropy_cluster[i] = class_entropy.mean()

        # Choose clusters with largest entropy
        sorted_idx = np.argsort(entropy_cluster)
        idx = sorted_idx[:k] # obtain indices of clusters we want to give hints to
        idx = np.isin(hints_q, idx).nonzero()[0]

        # Sample randomly within the chosen clusters
        idx_rand = idx[np.random.choice(idx.shape[0], hint, replace=False)]
        chosen_hints = hints.view(N,C,-1)[...,idx_rand].clone()

        # Fill in hints values and corresponding mask values
        hints.fill_(0)
        hints.view(N,C,-1)[...,idx_rand] = chosen_hints
        mask.view(N,1,-1)[...,idx_rand] = .5

    data['hints'] = hints
    data['mask'] = mask

    return data


def forward(model, classifier, opt, data):
    out_class, out_reg = model(data['L'], data['hints'].clamp(-1,1), data['mask'].clamp(-.5,.5))
    out_rgb = lab2rgb(torch.cat((data['L'], out_reg), dim=1), opt).clamp(0,1)
    y_pred = classifier(normalize(out_rgb))
    return out_rgb, y_pred

def compute_loss(opt, y, criterion):
    t = torch.LongTensor([opt.target]*y.shape[0]).cuda()
    loss = criterion(y, t)
    if not opt.targeted:
        loss *= -1
    return loss

def compute_class(opt, y, num_labels=5):
    y_softmax = F.softmax(y)
    # assume we are using batch of 1 here and no jpg loss
    val, idx = y_softmax[0].sort(descending=True)
    labels = [(opt.idx2label[idx[i]], round(val[i].item(), 3)) for i in range(num_labels)]

    return val, idx, labels

# Converts a Tensor into an image array (numpy)
# |imtype|: the desired type of the converted numpy array
# Color conversion code
def rgb2xyz(rgb): # rgb from [0,1]
    # xyz_from_rgb = np.array([[0.412453, 0.357580, 0.180423],
        # [0.212671, 0.715160, 0.072169],
        # [0.019334, 0.119193, 0.950227]])

    mask = (rgb > .04045).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (((rgb+.055)/1.055)**2.4)*mask + rgb/12.92*(1-mask)

    x = .412453*rgb[:,0,:,:]+.357580*rgb[:,1,:,:]+.180423*rgb[:,2,:,:]
    y = .212671*rgb[:,0,:,:]+.715160*rgb[:,1,:,:]+.072169*rgb[:,2,:,:]
    z = .019334*rgb[:,0,:,:]+.119193*rgb[:,1,:,:]+.950227*rgb[:,2,:,:]
    out = torch.cat((x[:,None,:,:],y[:,None,:,:],z[:,None,:,:]),dim=1)

    # if(torch.sum(torch.isnan(out))>0):
        # print('rgb2xyz')
        # embed()
    return out

def xyz2rgb(xyz):
    # array([[ 3.24048134, -1.53715152, -0.49853633],
    #        [-0.96925495,  1.87599   ,  0.04155593],
    #        [ 0.05564664, -0.20404134,  1.05731107]])

    r = 3.24048134*xyz[:,0,:,:]-1.53715152*xyz[:,1,:,:]-0.49853633*xyz[:,2,:,:]
    g = -0.96925495*xyz[:,0,:,:]+1.87599*xyz[:,1,:,:]+.04155593*xyz[:,2,:,:]
    b = .05564664*xyz[:,0,:,:]-.20404134*xyz[:,1,:,:]+1.05731107*xyz[:,2,:,:]

    rgb = torch.cat((r[:,None,:,:],g[:,None,:,:],b[:,None,:,:]),dim=1)
    rgb = torch.max(rgb,torch.zeros_like(rgb)) # sometimes reaches a small negative number, which causes NaNs

    mask = (rgb > .0031308).type(torch.FloatTensor)
    if(rgb.is_cuda):
        mask = mask.cuda()

    rgb = (1.055*(rgb**(1./2.4)) - 0.055)*mask + 12.92*rgb*(1-mask)

    # if(torch.sum(torch.isnan(rgb))>0):
        # print('xyz2rgb')
        # embed()
    return rgb

def xyz2lab(xyz):
    # 0.95047, 1., 1.08883 # white
    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    if(xyz.is_cuda):
        sc = sc.cuda()

    xyz_scale = xyz/sc

    mask = (xyz_scale > .008856).type(torch.FloatTensor)
    if(xyz_scale.is_cuda):
        mask = mask.cuda()

    xyz_int = xyz_scale**(1/3.)*mask + (7.787*xyz_scale + 16./116.)*(1-mask)

    L = 116.*xyz_int[:,1,:,:]-16.
    a = 500.*(xyz_int[:,0,:,:]-xyz_int[:,1,:,:])
    b = 200.*(xyz_int[:,1,:,:]-xyz_int[:,2,:,:])
    out = torch.cat((L[:,None,:,:],a[:,None,:,:],b[:,None,:,:]),dim=1)

    return out

def lab2xyz(lab):
    y_int = (lab[:,0,:,:]+16.)/116.
    x_int = (lab[:,1,:,:]/500.) + y_int
    z_int = y_int - (lab[:,2,:,:]/200.)
    if(z_int.is_cuda):
        z_int = torch.max(torch.Tensor((0,)).cuda(), z_int)
    else:
        z_int = torch.max(torch.Tensor((0,)), z_int)

    out = torch.cat((x_int[:,None,:,:],y_int[:,None,:,:],z_int[:,None,:,:]),dim=1)
    mask = (out > .2068966).type(torch.FloatTensor)
    if(out.is_cuda):
        mask = mask.cuda()

    out = (out**3.)*mask + (out - 16./116.)/7.787*(1-mask)

    sc = torch.Tensor((0.95047, 1., 1.08883))[None,:,None,None]
    sc = sc.to(out.device)

    out = out*sc

    return out

def rgb2lab(rgb):
    lab = xyz2lab(rgb2xyz(rgb))
    l_rs = (lab[:,[0],:,:]-50.)/100.
    ab_rs = lab[:,1:,:,:]/110.
    out = torch.cat((l_rs,ab_rs),dim=1)
    return out

def lab2rgb(lab_rs, opt=None):
    #l = lab_rs[:,[0],:,:]*opt.l_norm + opt.l_cent
    l = lab_rs[:,[0],:,:]*100. + 50.
    ab = lab_rs[:,1:,:,:]*110.
    lab = torch.cat((l,ab),dim=1)
    out = xyz2rgb(lab2xyz(lab))
    return out



def attack(compress_image, model=None, device='cpu', is_jpegai=False, loss_func=None, loss_func_name='undefined', 
           lr=0.00001, n_iters = 50):

    input_range = 1

    if hasattr(model, 'input_range'):
        input_range = model.input_range

    if is_jpegai:
        print(f'Attacking JPEGAI model, input_range: ', input_range)

    color_model = color_net().eval().to(device)
    color_model.load_state_dict(torch.load('cadv-colorization-model.pth'))
    old_compress_image = compress_image.clone()
    old_w = compress_image.shape[2]
    old_h = compress_image.shape[3]
    w = (old_w // 4) * 4
    h = (old_h // 4) * 4

    compress_image = compress_image[:,:,:w,:h]
    compress_image = compress_image.to(device)

    src_image = compress_image.clone().to(device)
    src_image = torch.autograd.Variable(src_image, requires_grad=True)
    decompr_src = model(src_image.to(device))['x_hat']
    decompr_src = torch.autograd.Variable(decompr_src.clone().to(device), requires_grad=True)

    data= get_colorization_data(compress_image / input_range, color_model, device)
    optimizer = torch.optim.Adam([data['hints'].requires_grad_(), data['mask'].requires_grad_()], lr=lr, betas=(0.9, 0.999))

    for iteration in range(n_iters):
        out_class, out_reg = color_model(data['L'], data['hints'].clamp(-1,1), data['mask'].clamp(-.5,.5))
        compress_image = lab2rgb(torch.cat((data['L'], out_reg), dim=1)).clamp(0,1) * input_range
        compression_results = model(compress_image.to(device))
        decompr, likelihoods, bpp_loss = compression_results['x_hat'], compression_results['likelihoods'], compression_results['bpp']
        loss = loss_func(src_image, compress_image, decompr_src, decompr, bpp_loss, is_jpegai)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        compress_image.data.clamp_(min=0, max=input_range)
        compress_image.data[torch.isnan(compress_image.data)] = 0

    old_compress_image[:,:,:w,:h] = compress_image
    compress_image = old_compress_image
    res_image = (compress_image).data.clamp_(min=0, max=input_range)
    return torch.clamp(res_image, 0, input_range)



if __name__ == "__main__":
    test_main(attack)
