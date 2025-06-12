import torch
import torchvision
from kornia import color
from color_transforms_255 import ycbcr_to_rgb_255, rgb_to_ycbcr_255, rgb_to_y_255
from pytorch_msssim import ms_ssim
# inputs of loss funcs are assumed to be in rgb with a range of [0,1] for all models except JPEGAI, 
# for JPEGAI RGB [0,255] for x, x_hat; YCbCr [0,255] for y, y_hat
loss_color_space = 'YCbCr'
def process_colorspace(x, x_hat, y, y_hat, is_jpegai):
    if loss_color_space == 'rgb':
        if is_jpegai:
            y = ycbcr_to_rgb_255(y)
            y_hat = ycbcr_to_rgb_255(y_hat)
        else:
            pass
    elif loss_color_space == 'YCbCr':
        if is_jpegai:
            print('jpegai ycbcr loss color conversion')
            x = rgb_to_ycbcr_255(x)
            x_hat = rgb_to_ycbcr_255(x_hat)
            print(f'x, x_hat, y, y_hat mins: {x.min()}, {x_hat.min()}, {y.min()}, {y_hat.min()}')
            print(f'x, x_hat, y, y_hat maxs: {x.max()}, {x_hat.max()}, {y.max()}, {y_hat.max()}')
        else:
            x = color.rgb_to_ycbcr(x)
            x_hat = color.rgb_to_ycbcr(x_hat)
            y = color.rgb_to_ycbcr(y)
            y_hat = color.rgb_to_ycbcr(y_hat)
    return x, x_hat, y, y_hat

def added_noises_loss(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False): # x - before compression, y - after, hat - attacked
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.square(y_hat - y - (x_hat - x)).mean()

def added_noises_loss_Y(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False): 
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.square(y_hat[:,0,:,:] - y[:,0,:,:] - (x_hat[:,0,:,:] - x[:,0,:,:])).mean()

def reconstr_loss(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False):
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.nn.functional.mse_loss(x_hat, y_hat)

def reconstr_loss_Y(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False):
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.nn.functional.mse_loss(x_hat[:,0,:,:], y_hat[:,0,:,:])

def src_reconstr_loss_Y(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False): 
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.nn.functional.mse_loss(x[:,0,:,:], y_hat[:,0,:,:])

def ftda_default_loss(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False): 
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.nn.functional.mse_loss(y, y_hat)

def ftda_default_loss_Y(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False): 
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return -torch.nn.functional.mse_loss(y[:,0,:,:], y_hat[:,0,:,:])

def ftda_msssim_loss(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False):
    data_range = 1
    if is_jpegai:
        data_range = 255.0
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return ms_ssim(y, y_hat, data_range=data_range)

def reconstruction_msssim_loss(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False):
    data_range = 1
    if is_jpegai:
        data_range = 255.0
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat, is_jpegai)
    return ms_ssim(x_hat, y_hat, data_range=data_range)

def bpp_increase_loss(x, x_hat, y, y_hat, bpp_loss, is_jpegai=False):
    return 1 - bpp_loss

loss_func_name = 'added_noises_loss'
loss_name_2_func = {
    'added_noises_loss':added_noises_loss,
    'reconstr_loss':reconstr_loss,
    'ftda_default_loss':ftda_default_loss,
    'ftda_msssim_loss':ftda_msssim_loss,
    'reconstruction_msssim_loss':reconstruction_msssim_loss,
    'bpp_increase_loss':bpp_increase_loss,
    'added_noises_loss_Y':added_noises_loss_Y,
    'reconstr_loss_Y':reconstr_loss_Y,
    'src_reconstr_loss_Y':src_reconstr_loss_Y,
    'ftda_default_loss_Y':ftda_default_loss_Y
}
# experimental
def pointwise_added_noises_loss(x, x_hat, y, y_hat, h_pos=250, w_pos=250, kernel_size=19, sigma=3): # x - before compression, y - after, hat - attacked
    # construct a mask based on the position of the pixel: create aa gaussian mask (with center in h_pos, w_pos) and apply it to the loss
    # it should work for dimensions B x C x H x W
    x, x_hat, y, y_hat = process_colorspace(x, x_hat, y, y_hat)
    mask = torch.zeros_like(x)
    mask[:, :, h_pos, w_pos] = 1
    blur_op = torchvision.transforms.GaussianBlur(kernel_size, sigma)
    mask = blur_op(mask)
    
    #return torch.nn.functional.mse_loss(x_hat, y_hat)
    return -torch.square((y_hat - y - (x_hat - x)) * mask).mean()


loss_func = loss_name_2_func[loss_func_name]