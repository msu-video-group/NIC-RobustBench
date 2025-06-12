import torch
import torch.nn as nn

from runpy import run_path
from collections import OrderedDict
from defence.eval_sde_adv import SDE_Adv_Model

import torch.nn.functional as F
import subprocess
import argparse
import yaml
import os
import defence.utils as utils
    
class Defense:
    def __init__(self, weights_path=1):
        args = {'config':'imagenet.yml',
                'data_seed':0,
                'seed':1234,
                'verbose':'info',
                'sample_step':1,
                't':5,
                't_delta':15,
                'rand_t':False,
                'diffusion_type':'ddpm',
                'score_type':'guided_diffusion',
                'eot_iter':20,
                'sigma2':1e-3,
                'lambda_ld':1e-2,
                'eta':5.,
                'step_size':1e-3,
                'domain':'celebahq',
                'classifier_name':'Eyeglasses',
                'partition':'val',
                'adv_batch_size':64,
                'attack_type':'square',
                'lp_norm':'Linf',
                'attack_version':'custom',
                'num_sub':1000,
                'adv_eps':0.07
                }
        args = argparse.Namespace(**args)
        
        with open(os.path.join('defence', 'configs', args.config), 'r') as f:
            config = yaml.safe_load(f)
        """
        config = {'image_size': 256, 'num_channels': 256, 'num_res_blocks': 2,
                   'num_heads': 4, 'num_heads_upsample': -1, 'num_head_channels': 64,
                     'attention_resolutions': '32,16,8', 'channel_mult': '', 'dropout': 0.0,
                       'class_cond': False, 'use_checkpoint': False, 'use_scale_shift_norm': True,
                         'resblock_updown': True, 'use_fp16': True, 'use_new_attention_order': False,
                           'learn_sigma': True, 'diffusion_steps': 1000, 'noise_schedule': 'linear',
                             'timestep_respacing': '1000', 'use_kl': False, 'predict_xstart': False,
                               'rescale_timesteps': True, 'rescale_learned_sigmas': False, 'device':'cuda'}
        """
        new_config = utils.dict2namespace(config)
        new_config.device = torch.device('cuda')
        self.defence_model = SDE_Adv_Model(args, new_config)
        self.defence_model.eval()
        self.defence_name = 'diffpure'

    def __call__(self, image):
        self.defence_model.to(image.device)
        res = self.defence_model(image)
        return res.clamp(0.0, 1.0)
   