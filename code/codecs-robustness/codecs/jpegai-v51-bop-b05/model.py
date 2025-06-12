"""
Module: CodecModel

A PyTorch wrapper for the JPEG AI v5 neural image compression codec, integrating the JPEG AI Part 1 Core Coding Engine with β-rate control and optional main-run bitstream pipeline.

The source code: 
https://gitlab.com/wg1/jpeg-ai/jpeg-ai-reference-software

"""

import math
import io
import torch
from torchvision import transforms
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sys, os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPT_DIR += "/JPEG_AIv5_main"
sys.path.append(SCRIPT_DIR)
from JPEG_AIv5_main import *
from kornia import color

class CodecModel(torch.nn.Module):
    def __init__(self, device, beta=0.5, type_model='bop', is_main=False):
        super().__init__()
        self.device = device
        self.beta = beta
        self.type_model = type_model

        self.input_range = 255
        self.output_range = 255
        self.output_cspace = 'YCbCr'
        
        if not is_main:
            self.conf = set_conf(beta=beta, type_model=type_model)
            self.is_main = False
        else:
            # Create a main codec when changing the configuration file in main_run runs (change hop/bop). Due to the specifics of the main codec, the result is affected by the number of images compressed by the codec with the same bpp
            self.main = create_main()
            self.cfg = "JPEG_AIv5_main/" + self.type_model + '.json'
            BRM = {0.002 : ['12'], 0.012 : ['25'], 0.075 : ['75'], 0.5 : ['100']}
            self.bpps = BRM[beta]
            self.is_main = True

    def forward(self, image, bits_pathes = None, rec_pathes = None, return_bpp=False):
            if self.is_main:
                if len(image.shape) < 4:
                    image = image.unsqueeze(0)
                # for v6.1 there isn't profile argument
                img = Image_jpeg.create_from_tensor(tensor=image,
                                   data_range=[0.0, 255.0],
                                   bit_depth=8,
                                   profile=None,
                                   color_space='rgb')
                
                ans = []

                for i in range(len(self.bpps)):  
                    try:
                        out = main_run(self.main, img, self.bpps[i], path_for_bin=bits_pathes[i], path_for_reco=rec_pathes[i], path_to_cfg=self.cfg)
                        info = os.stat(bits_pathes[i])
                        bpp = info.st_size * 8 / (image.shape[-1] * image.shape[-2])
                        out = torch.nan_to_num(out.type(torch.float32), nan=0, posinf=0, neginf=0)
                        ret = {'x_hat': out.unsqueeze(0).type(torch.float32), 'likelihoods':None, 'bpp':bpp}
                    except:
                        ret = {'x_hat': None, 'likelihoods':None, 'bpp': np.nan}
                    ans.append(ret)
                
                return ans[0]

            image = image.to('cuda:0')
            image = image.unsqueeze(0)
            self.conf[0].eval()
            #out = jpeg_run(image * 255.0, 0.5, self.conf)
            bpp_loss, out = jpeg_run_bitrate(image, self.beta, self.conf, type_atack='y')
            #print('JPEGAI out shape: ', out.shape)
            #print(f'JPEGAI img max and min values: {torch.min(out)} {torch.max(out)}')
            if bpp_loss is None:
                bpp_loss = torch.tensor([torch.nan]).to(image.device)
            #return {'x_hat': color.ycbcr_to_rgb(out / 255.0).type(torch.float32), 'likelihoods':None, 'bpp':bpp_loss}
            return {'x_hat': out.type(torch.float32), 'likelihoods':None, 'bpp':bpp_loss}