"""
This module defines `CodecModel`, a PyTorch `nn.Module` wrapper for the EVC_LL
neural image codec. It loads a pretrained checkpoint (`EVC_LL.pth.tar`), reads
the model’s quantization scales, interpolates them over a fixed number of rate
levels, and selects the scale corresponding to a given rate index.

The source code:
https://github.com/microsoft/DCVC

The paper:
Guo-Hua Wang, Jiahao Li, Bin Li, and Yan Lu:
"Evc: Towards real-time neural image compression with mask decay," ICLR 2023

"""

import math
import io
import torch
from torchvision import transforms
import numpy as np

from PIL import Image

import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn.functional as F
from src.src.utils.common import str2bool, interpolate_log, create_folder, dump_json
from src.src.models import build_model
from src.src.utils.stream_helper import get_padding_size, get_state_dict

class CodecModel(torch.nn.Module):
    def __init__(self, device):
        super().__init__()

        self.args = {
            'i_frame_model_path':'EVC_LL.pth.tar',
            'i_frame_model':"EVC_LL",
            'ec_thread':1,
            'rate_idx':1,
        }

        self.rate_num = 6
        
        self.device = device
        i_state_dict = get_state_dict(self.args['i_frame_model_path'])
        
        q_scales_list = []
        if "q_scale" in i_state_dict:
            q_scales = i_state_dict["q_scale"]
        elif "student.q_scale" in i_state_dict:
            q_scales = i_state_dict["student.q_scale"]
        elif "teacher.q_scale" in i_state_dict:
            q_scales = i_state_dict["teacher.q_scale"]
        q_scales_list.append(q_scales.reshape(-1))

        if q_scales_list:
            i_frame_q_scales = torch.cat(q_scales_list)
        else:
            i_frame_q_scales = []
        
        if len(i_frame_q_scales) > 0:
            max_q_scale = i_frame_q_scales[0]
            min_q_scale = i_frame_q_scales[-1]
            i_frame_q_scales = interpolate_log(min_q_scale, max_q_scale, self.rate_num)
            i_frame_q_scales = torch.tensor(i_frame_q_scales)
    
        if len(i_frame_q_scales) > 0:
            self.args['i_frame_q_scale'] = i_frame_q_scales[self.args['rate_idx']].to(torch.float32)
        else:
            self.args['i_frame_q_scale'] = []

        self.i_frame_net = build_model(self.args['i_frame_model'], ec_thread=self.args['ec_thread'])
        self.i_frame_net.load_state_dict(i_state_dict, verbose=False)
        if hasattr(self.i_frame_net, 'set_rate'):
            self.i_frame_net.set_rate(self.args['rate_idx'])
        self.i_frame_net = self.i_frame_net.to(device)
        self.i_frame_net.eval()
    
    def forward(self, image, return_bpp=False):
        self.eval()
        pic_height = image.shape[2]
        pic_width = image.shape[3]

        padding_l, padding_r, padding_t, padding_b = get_padding_size(pic_height, pic_width)
        x_padded = torch.nn.functional.pad(
            image,
            (padding_l, padding_r, padding_t, padding_b),
            mode="constant",
            value=0,
        )

        result = self.i_frame_net(x_padded, self.args['i_frame_q_scale'])

        recon_frame = result["x_hat"]
        recon_frame = recon_frame.clamp_(0, 1)
        compressed = F.pad(recon_frame, (-padding_l, -padding_r, -padding_t, -padding_b))
        
        return {'x_hat': compressed, 'likelihoods':None, 'bpp': result['bpp']}