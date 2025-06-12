import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import model

np.set_printoptions(suppress=True)


def Get_PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def Handle_Parameter(conv, b_W):
    weight = conv.weight.data
    bias = conv.bias.data
    bit_range = 2 ** (b_W - 1)

    range_float = max(weight.abs().max(), bias.abs().max())
    range_float = torch.clamp(range_float, 0, 1)
    range_int01 = torch.round(range_float * bit_range) / bit_range

    weight_01 = torch.clamp(weight / range_int01, -1, 1)
    bias_01 = torch.clamp(bias / range_int01, -1, 1)

    weight_int = torch.round(weight_01 * bit_range)
    bias_int = torch.round(bias_01 * bit_range)

    conv.weight.data = weight_int
    conv.bias.data = bias_int

    #print(range_int01.item())
    return conv, range_int01


# Converting activations to fixed-point representation.
# @x:           input tensor.
# @a_l:         number of bits to the left of the decimal point.
# @a_r:         number of bits to the right of the decimal point.
def Handle_Activation(x, a_l=0, a_r=8, be_ReLU=False):
    if be_ReLU:
        a_low = 0
        a_high = 2 ** a_l
    else:
        a_low = -2 ** (a_l - 1)
        a_high = 2 ** (a_l - 1)
    a_mult = 2 ** a_r

    out = torch.clamp(x, a_low, a_high)
    out = torch.round(out * a_mult) / a_mult
    return out


# Fix-point convolution
# @x:           input tensor.
# @conv:        convolution parameter (including weight and bias).
# @stride:      nn.functional.conv2d parameter
# @a_l:         number of bits to the left of the decimal point.
# @a_r:         number of bits to the right of the decimal point.
# @b_W:         bit-width of weight.
# @be_Last:     whether the current convolution layer is the last one.
# @be_ReLU:     whether ReLU is connected in series behind the current convolution layer.
def QConv(x, conv, stride=1, a_l=0, a_r=8, b_W=8, be_Last=False, be_ReLU=False):
    conv, range_factor = Handle_Parameter(conv, b_W)
    out = nn.functional.conv2d(x, conv.weight, conv.bias, stride, 1)
    out *= range_factor / (2 ** (b_W - 1))

    if not be_Last:
        out = Handle_Activation(out, be_ReLU=be_ReLU)
    return out, conv, range_factor * (2 ** (b_W - 1))


def LRS_Fixed_Inference(img, sr_model, be_Save=False):
    out1, sr_model.conv1, range_1 = QConv(img, sr_model.conv1, stride=1, be_ReLU=True)
    #print(sr_model.conv1.weight)

    print(sr_model.SU_Res1.conv1.weight[33, 33].cpu().numpy())
    out2_1, sr_model.SU_Res1.conv1, range_2 = QConv(out1, sr_model.SU_Res1.conv1, be_ReLU=True)
    print(sr_model.SU_Res1.conv1.weight[33, 33].cpu().numpy())
    print(range_2)

    out2_2, sr_model.SU_Res1.conv2, range_3 = QConv(out2_1, sr_model.SU_Res1.conv2)
    out2 = Handle_Activation(out1 + out2_2)

    out3_1, sr_model.SU_Res2.conv1, range_4 = QConv(out2, sr_model.SU_Res2.conv1, be_ReLU=True)
    out3_2, sr_model.SU_Res2.conv2, range_5 = QConv(out3_1, sr_model.SU_Res2.conv2)
    out3 = Handle_Activation(out2 + out3_2)

    out4_1, sr_model.SU_Res3.conv1, range_6 = QConv(out3, sr_model.SU_Res3.conv1, be_ReLU=True)
    out4_2, sr_model.SU_Res3.conv2, range_7 = QConv(out4_1, sr_model.SU_Res3.conv2)
    out4 = Handle_Activation(out3 + out4_2)

    out5 = Handle_Activation(out4 + out1)
    out6, sr_model.conv2, range_8 = QConv(out5, sr_model.conv2, be_ReLU=True)

    out7 = sr_model.PixelShuffle(out6)
    out8, sr_model.conv3, range_9 = QConv(out7, sr_model.conv3, stride=2, be_Last=True)

    range_set = np.array([range_1, range_2, range_3, range_4, range_5, range_6, range_7, range_8, range_9])
    np.save("../models/range.npy", range_set)

    if be_Save:
        torch.save(sr_model.state_dict(), "../models/lrs_integer.pth")

    return out8


def Single_Test(model_path, image_path):
    sr_model = model.SuperResolution().cuda()
    sr_model.load_state_dict(torch.load(model_path))
    sr_model.eval()

    ori_img = Image.open(image_path)
    img = ori_img.resize((int(ori_img.size[0] / 1.5), int(ori_img.size[1] / 1.5)), Image.BICUBIC)
    img = np.array(img) / 255.0
    img = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0).cuda()

    with torch.no_grad():
        output = LRS_Fixed_Inference(img, sr_model, True)

        output[output >= 1] = 1
        output[output <= 0] = 0
        output = output.data[0] * 255.0
        output = output.cpu().numpy().transpose(1, 2, 0).astype("int")

    psnr = Get_PSNR(np.array(ori_img), output)
    print(image_path, psnr)


model_path = "../models/lrs_fixed.pth"
image_path = "../Kodak_Cli/kodim05.png"

Single_Test(model_path, image_path)
