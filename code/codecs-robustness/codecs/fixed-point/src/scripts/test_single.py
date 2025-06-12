import math
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import src.model as model

np.set_printoptions(suppress=True)


def Get_PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def Check(x, a_mult, be_ReLU=False):
    x = x.detach().cpu().numpy()
    diff = np.sum((x * a_mult - np.round(x * a_mult)) ** 2)
    assert(diff < 0.000001 and diff > -0.000001)


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
def QConv(x, conv, range_factor, stride=1, a_l=0, a_r=8, b_W=8, be_Last=False, be_ReLU=False):
    bit_range = 2 ** (b_W - 1)

    out = nn.functional.conv2d(x, conv.weight, conv.bias, stride, 1)
    out *= range_factor / (bit_range ** 2)

    if not be_Last:
        out = Handle_Activation(out, be_ReLU=be_ReLU)
    return out, conv


def LRS_Fixed_Inference(img, sr_model):
    range_set = np.load("../models/range.npy", allow_pickle=True)
    # print(range_set)

    out1, sr_model.conv1 = QConv(
        img, sr_model.conv1, range_set[0], be_ReLU=True)
    Check(out1, 2 ** 8)  # a_r = 8, thus a_mult = 2 ** 8

    out2_1, sr_model.SU_Res1.conv1 = QConv(
        out1, sr_model.SU_Res1.conv1, range_set[1], be_ReLU=True)
    out2_2, sr_model.SU_Res1.conv2 = QConv(
        out2_1, sr_model.SU_Res1.conv2, range_set[2])
    out2 = Handle_Activation(out1 + out2_2)
    Check(out2_1, 2 ** 8)
    Check(out2_2, 2 ** 8)
    Check(out2, 2 ** 8)

    out3_1, sr_model.SU_Res2.conv1 = QConv(
        out2, sr_model.SU_Res2.conv1, range_set[3], be_ReLU=True)
    out3_2, sr_model.SU_Res2.conv2 = QConv(
        out3_1, sr_model.SU_Res2.conv2, range_set[4])
    out3 = Handle_Activation(out2 + out3_2)
    Check(out3_1, 2 ** 8)
    Check(out3_2, 2 ** 8)
    Check(out3, 2 ** 8)

    out4_1, sr_model.SU_Res3.conv1 = QConv(
        out3, sr_model.SU_Res3.conv1, range_set[5], be_ReLU=True)
    out4_2, sr_model.SU_Res3.conv2 = QConv(
        out4_1, sr_model.SU_Res3.conv2, range_set[6])
    out4 = Handle_Activation(out3 + out4_2)
    Check(out4_1, 2 ** 8)
    Check(out4_2, 2 ** 8)
    Check(out4, 2 ** 8)

    out5 = Handle_Activation(out4 + out1)
    out6, sr_model.conv2 = QConv(
        out5, sr_model.conv2, range_set[7], be_ReLU=True)
    Check(out5, 2 ** 8)
    Check(out6, 2 ** 8)

    out7 = sr_model.PixelShuffle(out6)
    out8, sr_model.conv3 = QConv(
        out7, sr_model.conv3, range_set[8], stride=2, be_Last=True)
    Check(out7, 2 ** 8)

    return out8


def Single_Test(model_path, image_path):
    sr_model = model.SuperResolution().cuda()
    sr_model.load_state_dict(torch.load(model_path))
    sr_model.eval()

    ori_img = Image.open(image_path)
    img = ori_img.resize(
        (int(ori_img.size[0] / 1.5), int(ori_img.size[1] / 1.5)), Image.BICUBIC)
    img = np.array(img) / 255.0
    img = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0).cuda()

    with torch.no_grad():
        output = LRS_Fixed_Inference(img, sr_model)

        output[output >= 1] = 1
        output[output <= 0] = 0
        output = output.data[0] * 255.0
        output = output.cpu().numpy().transpose(1, 2, 0).astype("int")

    img = Image.fromarray(output.astype(np.uint8))
    img.save("rec_kodim08.png")

    psnr = Get_PSNR(np.array(ori_img), output)
    print(image_path, psnr)
    return psnr


def main():
    model_path = "../models/lrs_integer.pth"
    image_path = "../Kodak_Cli/kodim08.png"

    Single_Test(model_path, image_path)


if __name__ == "__main__":
    main()
