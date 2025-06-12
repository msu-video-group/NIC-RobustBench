import os
import math
import numpy as np
import torch
import torch.nn as nn
#import torch.utils.data as Data
from PIL import Image
from torch.autograd import Variable
import model
from test_single import Single_Test


def Get_PSNR(img1, img2):
    mse = np.mean((img1 - img2) ** 2)
    if mse < 1.0e-10:
        return 100
    return 10 * math.log10(255.0 ** 2 / mse)


def Kodak_Test(model_path, be_Fixed=False):
    psnr_sum = 0.
    img_dir = "../Kodak_Cli/"

    if be_Fixed:
        for img_name in os.listdir(img_dir):
            psnr = Single_Test(model_path, img_dir + img_name)
            psnr_sum += psnr
    else:
        for img_name in os.listdir(img_dir):
            sr_model = model.SuperResolution().cuda()
            sr_model.load_state_dict(torch.load(model_path))
            sr_model.eval()

            ori_img = Image.open(img_dir + img_name)
            img = ori_img.resize(
                (int(ori_img.size[0] / 1.5), int(ori_img.size[1] / 1.5)), Image.BICUBIC)
            img = np.array(img) / 255.0
            img = torch.FloatTensor(img.transpose(2, 0, 1)).unsqueeze(0).cuda()

            with torch.no_grad():
                output = sr_model(img)

                output[output >= 1] = 1
                output[output <= 0] = 0
                output = output.data[0] * 255.0
                output = output.cpu().numpy().transpose(1, 2, 0).astype("int")

            psnr = Get_PSNR(np.array(ori_img), output)
            print(img_name, psnr)
            psnr_sum += psnr

    print(model_path, psnr_sum / 24)


np.set_printoptions(suppress=True)
Kodak_Test("../models/lrs_integer.pth", be_Fixed=True)
Kodak_Test("../models/lrs_float.pth", be_Fixed=False)
