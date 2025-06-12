import torch
import cv2
import os
import csv
import json
import importlib
import time
import numpy as np
from read_dataset import to_numpy, iter_images, get_batch
from evaluate import compress, predict, write_log, eval_encoded_video, Encoder
from metrics import PSNR, SSIM, MSE, L_inf_dist, MAE, MSSSIM, vmaf, niqe
from traditional_reference_codec import jpeg2k_compress, jpeg2k_compress_fix_bpp
from frozendict import frozendict
from functools import partial
import defended_model as dfnd
import subprocess
import pandas as pd
import torchvision
from pathlib import Path
from codec_scoring_methods import calc_scores_codec
from torchvision import transforms
from codec_losses import loss_name_2_func
from color_transforms_255 import ycbcr_to_rgb_255
import glymur

np.random.seed(int(time.time()))
# fr cols
FR_COLS = ['msssim', 'ssim', 'mse', 'psnr', 'l_inf', 'mae', 'vmaf']

NR_COLS = ['niqe']

# Columns in resulting dataframe with raw values
RAW_RESULTS_COLS = ['image_name','codec_name','defence_name','test_dataset','loss_name'] + \
    ['bpp_defended-clear', 'bpp_undefended-clear', 'bpp_defended-attacked', 'bpp_undefended-attacked'] + \
    ['real-bpp_defended-clear', 'real-bpp_undefended-clear', 'real-bpp_defended-attacked', 'real-bpp_undefended-attacked'] + \
    ['bpp_jpeg-clear', 'bpp_jpeg-attacked', 'bpp_jpeg-clear-fix', 'bpp_jpeg-attacked-fix'] + \
    [f'{x}_clear_defended-rec-clear' for x in FR_COLS] + \
    [f'{x}_clear_attacked' for x in FR_COLS] + \
    [f'{x}_attacked_defended-rec-attacked' for x in FR_COLS] + \
    [f'{x}_clear_rec-clear' for x in FR_COLS] + \
    [f'{x}_attacked_rec-attacked' for x in FR_COLS] + \
    [f'{x}_clear' for x in NR_COLS ] + \
    [f'{x}_attacked' for x in NR_COLS ] + \
    [f'{x}_rec-attacked' for x in NR_COLS ] + \
    [f'{x}_rec-clear' for x in NR_COLS ] + \
    [f'{x}_defended-rec-clear' for x in NR_COLS ] + \
    [f'{x}_defended-rec-attacked' for x in NR_COLS ] + \
    [f'{x}_rec-clear_rec-attacked' for x in FR_COLS] + \
    [f'{x}_defended-rec-clear_defended-rec-attacked' for x in FR_COLS] + \
    [f'{x}_clear_rec-clear_jpeg' for x in FR_COLS] + \
    [f'{x}_attacked_rec-attacked_jpeg' for x in FR_COLS] + \
    [f'{x}_rec-clear_rec-attacked_jpeg' for x in FR_COLS] + \
    [f'{x}_rec-attacked_jpeg' for x in NR_COLS] + [f'{x}_rec-clear_jpeg' for x in NR_COLS] + \
    [f'{x}_clear_rec-clear_jpeg-fix' for x in FR_COLS] + \
    [f'{x}_attacked_rec-attacked_jpeg-fix' for x in FR_COLS] + \
    [f'{x}_rec-clear_rec-attacked_jpeg-fix' for x in FR_COLS] + \
    [f'{x}_rec-attacked_jpeg-fix' for x in NR_COLS] + [f'{x}_rec-clear_jpeg-fix' for x in NR_COLS]

def calc_frs(src, dest, device='cuda:0'):
    fr_vals = {
                'mse':[],
                'mae':[],
                'ssim':[],
                'psnr':[],
                'l_inf':[],
                'msssim':[],
                'vmaf':[]
            }
    for i in range(len(src)):
        fr_vals['mse'].append(MSE(src[i].clone().unsqueeze(0).to(device), dest[i].clone().unsqueeze(0).to(device)))
        fr_vals['mae'].append(MAE(src[i].clone().unsqueeze(0).to(device), dest[i].clone().unsqueeze(0).to(device)))
        fr_vals['ssim'].append(SSIM(src[i].clone().unsqueeze(0).to(device), dest[i].clone().unsqueeze(0).to(device)))
        fr_vals['psnr'].append(PSNR(src[i].clone().unsqueeze(0).to(device), dest[i].clone().unsqueeze(0).to(device)))
        fr_vals['l_inf'].append(L_inf_dist(src[i].clone().unsqueeze(0).to(device), dest[i].clone().unsqueeze(0).to(device)))
        fr_vals['msssim'].append(MSSSIM(src[i].clone().unsqueeze(0).to(device), dest[i].clone().unsqueeze(0).to(device)))
        fr_vals['vmaf'].append(vmaf(src[i].clone().unsqueeze(0).to(device), dest[i].clone().unsqueeze(0).to(device)))
    return fr_vals

def calc_nrs(src, nr_models, device='cuda:0'):
    '''
    src : Tensor of shape [1,3,h,w]
    nr_models : Dict  NR metric name : Torch model
    '''
    nr_vals = {k:[] for k in NR_COLS}
    for i in range(len(src)):
        for k in NR_COLS:
            val = nr_models[k](src[i].clone().unsqueeze(0).to(device))
            if val is not None:
                val = val.cpu().item()
            else:
                val = np.nan
            nr_vals[k].append(val)
    return nr_vals

def apply_attack(model, attack_callback, dist_images, device='cpu', variable_params={}, seed=42, is_jpegai=False, loss_func=None, loss_func_name='undefined'):
    model.train()
    torch.manual_seed(seed)
    t0 = time.time()
    attacked_images = attack_callback(dist_images.clone(), model=model, device=device, is_jpegai=is_jpegai, loss_func=loss_func, loss_func_name=loss_func_name, **variable_params)

    attack_time = time.time() - t0
    model.eval()
    #print('Attacked images shape: ', attacked_images.shape)
    if attacked_images is None:
        #print('Attacked image is None')
        #print(attacked_images)
        return None
    return attacked_images, attack_time
    
def create_row(model, undefended_model, images, attacked_images, device, torch_seed, nr_models,
               video_name, codec_name, test_dataset, preset, defence_preset, defence_name,
               dump_path=None, batch_size=None, global_i=None, dump_freq=None, fn=None, is_main=False, input_range=255, output_range=255,
            is_jpegai=True, loss_func_name='undefined', reconstructed_save_path=None, save_freq=10):
    with torch.no_grad():
        attacked_images = attacked_images.to(device)
        images = images.to(device)
        time_st = time.time()
        torch.manual_seed(torch_seed)
        if is_main:
            def_clear_outs = model.forward(images, bits_pathes=[f'./{Path(fn).stem}_clear.bits'], rec_pathes=[f'./{Path(fn).stem}_clear.png'] )
        else:
            def_clear_outs = model(images)
        rec_defended_images = def_clear_outs['x_hat']
        if rec_defended_images is not None:
            rec_defended_images = torch.clamp(rec_defended_images, 0, output_range)
        def_clear_bpp = float(def_clear_outs['bpp'])

        torch.manual_seed(torch_seed)
        if is_main:
            def_attacked_outs = model.forward(attacked_images, bits_pathes=[f'./{Path(fn).stem}_attacked.bits'], rec_pathes=[f'./{Path(fn).stem}_attacked.png'] )
        else:
            def_attacked_outs = model(attacked_images)
        rec_defended_attacked_images = def_attacked_outs['x_hat']
        if rec_defended_attacked_images is not None:
            rec_defended_attacked_images = torch.clamp(rec_defended_attacked_images, 0, output_range)
        def_attacked_bpp = float(def_attacked_outs['bpp'])
        delta_time = time.time() - time_st

        torch.manual_seed(torch_seed)
        if is_main:
            undef_clear_outs = undefended_model.forward(images, bits_pathes=[f'./{Path(fn).stem}_undef_clear.bits'], rec_pathes=[f'./{Path(fn).stem}_undef_clear.png'] )
        else:
            undef_clear_outs = undefended_model(images)
        rec_undefended_images = undef_clear_outs['x_hat']
        if rec_undefended_images is not None:
            rec_undefended_images = torch.clamp(rec_undefended_images, 0, output_range)
        undef_clear_bpp = float(undef_clear_outs['bpp'])

        torch.manual_seed(torch_seed)
        if is_main:
            undef_attacked_outs = undefended_model.forward(attacked_images, bits_pathes=[f'./{Path(fn).stem}_undef_attacked.bits'], rec_pathes=[f'./{Path(fn).stem}_undef_attacked.png'] )
        else:
            undef_attacked_outs = undefended_model(attacked_images)
        rec_undefended_attacked_images = undef_attacked_outs['x_hat']
        if rec_undefended_attacked_images is not None:
            rec_undefended_attacked_images = torch.clamp(rec_undefended_attacked_images, 0, output_range)
        undef_attacked_bpp = float(undef_attacked_outs['bpp'])
        # Calculate all fr metrics
        # clear source images vs reconstructed clear (with defence as preprocessing, if provided)
        images = images.to(device)

        # convert everything to rgb [0,1]
        images = images / input_range
        attacked_images = attacked_images / input_range
        def _maincodec_output_2_rgb_01(x):
            return torch.clamp(x / output_range, 0,1) if x is not None else None
        if is_jpegai:
            rec_defended_images = _maincodec_output_2_rgb_01(rec_defended_images)
            rec_defended_attacked_images = _maincodec_output_2_rgb_01(rec_defended_attacked_images)
            rec_undefended_images = _maincodec_output_2_rgb_01(rec_undefended_images)
            rec_undefended_attacked_images = _maincodec_output_2_rgb_01(rec_undefended_attacked_images)
        
        target_quality = None

        if images is not None and rec_defended_images is not None:
            rec_defended_images = rec_defended_images.to(device)
            fr_clear_rec_def_clear = calc_frs(images, rec_defended_images)
            target_quality = fr_clear_rec_def_clear["psnr"]
            print(f'MAIN CODEC PSNR CLEAR REC CLEAR: {fr_clear_rec_def_clear["psnr"]}')
        else:
            print('MAIN CODEC FAILURE ON CLEAR')
            fr_clear_rec_def_clear = { x: np.nan for x in FR_COLS}

        if rec_defended_attacked_images is not None:
            # attacked images vs reconstructed attacked (with defence as preprocessing, if provided)
            fr_attacked_rec_def_attacked = calc_frs(attacked_images, rec_defended_attacked_images)
        else:
            fr_attacked_rec_def_attacked = { x: np.nan for x in FR_COLS}
        
        # clear vs attacked, wo reconstruction
        fr_clear_attacked = calc_frs(images, attacked_images)

        if rec_undefended_images is not None:
            # clear vs reconstructed clear (WITHOUT defence)
            fr_clear_rec_clear = calc_frs(images, rec_undefended_images)
        else:
            fr_clear_rec_clear = { x: np.nan for x in FR_COLS}
        
        if rec_undefended_attacked_images is not None:
            # clear vs reconstructed clear (WITHOUT defence)
            fr_attacked_rec_attacked = calc_frs(attacked_images, rec_undefended_attacked_images)
        else:
            fr_attacked_rec_attacked = { x: np.nan for x in FR_COLS}
        
        # reconstructed clear vs reconstructed attacked
        if rec_defended_attacked_images is not None and rec_defended_images is not None:
        
            fr_rec_def_clear_rec_def_attacked = calc_frs(rec_defended_images, rec_defended_attacked_images)
        else:
            fr_rec_def_clear_rec_def_attacked = { x: np.nan for x in FR_COLS}
        
        # reconstructed clear vs reconstructed attacked (WITHOUT defence)
        if rec_undefended_attacked_images is not None and rec_undefended_images is not None:
        
            fr_rec_clear_rec_attacked = calc_frs(rec_undefended_images, rec_undefended_attacked_images)
        else:
            fr_rec_clear_rec_attacked = { x: np.nan for x in FR_COLS}

        # Calculate NR metrics
        nr_clear = calc_nrs(images, nr_models)
        nr_attacked = calc_nrs(attacked_images, nr_models)

        if rec_undefended_attacked_images is not None:
            nr_rec_undefended_attacked = calc_nrs(rec_undefended_attacked_images, nr_models)
        else:
            nr_rec_undefended_attacked = { x: np.nan for x in NR_COLS}
        if rec_undefended_images is not None:
            nr_rec_undefended = calc_nrs(rec_undefended_images, nr_models)
        else:
            nr_rec_undefended = { x: np.nan for x in NR_COLS}
        if rec_defended_images is not None:
            nr_rec_defended = calc_nrs(rec_defended_images, nr_models)
        else:
            nr_rec_defended = { x: np.nan for x in NR_COLS}
        if rec_defended_attacked_images is not None:
            nr_rec_defended_attacked = calc_nrs(rec_defended_attacked_images, nr_models)
        else:
            nr_rec_defended_attacked = { x: np.nan for x in NR_COLS}

        if target_quality is not None:
            rec_clear_jpeg, jpeg_clear_bpp = jpeg2k_compress(images, dump_path, target_quality, device)
            rec_attacked_jpeg, jpeg_attacked_bpp = jpeg2k_compress(attacked_images, dump_path, target_quality, device)
            jpeg_clear_bpp = jpeg_clear_bpp[0]
            jpeg_attacked_bpp = jpeg_attacked_bpp[0]
            # real_bpps['real-bpp_jpeg-clear']  = float(jpeg_clear_bpp)
            # real_bpps['real-bpp_jpeg-attacked']  = float(jpeg_attacked_bpp)

            fr_clear_rec_clear_jpeg = calc_frs(images, rec_clear_jpeg)
            fr_attacked_rec_attacked_jpeg = calc_frs(attacked_images, rec_attacked_jpeg)
            fr_rec_clear_jpeg_rec_attacked_jpeg = calc_frs(rec_clear_jpeg, rec_attacked_jpeg)
            
            nr_rec_attacked_jpeg = calc_nrs(rec_attacked_jpeg, nr_models)
            nr_rec_clear_jpeg = calc_nrs(rec_clear_jpeg, nr_models)

            rec_clear_jpeg_fix, jpeg_clear_bpp_fix = jpeg2k_compress_fix_bpp(images, dump_path, [float(def_clear_bpp)], device)
            rec_attacked_jpeg_fix, jpeg_attacked_bpp_fix = jpeg2k_compress_fix_bpp(attacked_images, dump_path, [float(def_clear_bpp)], device)
            jpeg_clear_bpp_fix = jpeg_clear_bpp_fix[0]
            jpeg_attacked_bpp_fix = jpeg_attacked_bpp_fix[0]
            # real_bpps['real-bpp_jpeg-clear']  = float(jpeg_clear_bpp_fix)
            # real_bpps['real-bpp_jpeg-attacked']  = float(jpeg_attacked_bpp_fix)

            fr_clear_rec_clear_jpeg_fix = calc_frs(images, rec_clear_jpeg_fix)
            fr_attacked_rec_attacked_jpeg_fix = calc_frs(attacked_images, rec_attacked_jpeg_fix)
            fr_rec_clear_jpeg_rec_attacked_jpeg_fix = calc_frs(rec_clear_jpeg_fix, rec_attacked_jpeg_fix)
            
            nr_rec_attacked_jpeg_fix = calc_nrs(rec_attacked_jpeg_fix, nr_models)
            nr_rec_clear_jpeg_fix = calc_nrs(rec_clear_jpeg_fix, nr_models)
        else:
            fr_clear_rec_clear_jpeg = { x: np.nan for x in FR_COLS}
            fr_attacked_rec_attacked_jpeg = { x: np.nan for x in FR_COLS}
            fr_rec_clear_jpeg_rec_attacked_jpeg = { x: np.nan for x in FR_COLS}
            
            nr_rec_attacked_jpeg = { x: np.nan for x in NR_COLS}
            nr_rec_clear_jpeg = { x: np.nan for x in NR_COLS}

            jpeg_clear_bpp = np.nan
            jpeg_attacked_bpp = np.nan

            fr_clear_rec_clear_jpeg_fix = { x: np.nan for x in FR_COLS}
            fr_attacked_rec_attacked_jpeg_fix = { x: np.nan for x in FR_COLS}
            fr_rec_clear_jpeg_rec_attacked_jpeg_fix = { x: np.nan for x in FR_COLS}
            
            nr_rec_attacked_jpeg_fix = { x: np.nan for x in NR_COLS}
            nr_rec_clear_jpeg_fix = { x: np.nan for x in NR_COLS}

            jpeg_clear_bpp = np.nan
            jpeg_attacked_bpp = np.nan

    row =  {
        'image_name': Path(video_name).name,
        'codec_name':codec_name,
        'test_dataset': test_dataset,
        'loss_name':loss_func_name,
        'preset' : preset,
        'defence_preset': defence_preset,
        'defence_name':defence_name,
        # BPPs
        'bpp_defended-clear':float(def_clear_bpp),
        'bpp_undefended-clear':float(undef_clear_bpp),
        'bpp_defended-attacked':float(def_attacked_bpp),
        'bpp_undefended-attacked':float(undef_attacked_bpp),
        'bpp_jpeg-clear':float(jpeg_clear_bpp),
        'bpp_jpeg-attacked':float(jpeg_attacked_bpp),
        'bpp_jpeg-clear-fix':float(jpeg_clear_bpp_fix),
        'bpp_jpeg-attacked-fix':float(jpeg_attacked_bpp_fix),
        }       
    for col in FR_COLS:
        row[f'{col}_clear_defended-rec-clear'] = np.nan if np.isnan(fr_clear_rec_def_clear[col]).all() else np.nanmean(fr_clear_rec_def_clear[col])
        row[f'{col}_attacked_defended-rec-attacked'] = np.nan if np.isnan(fr_attacked_rec_def_attacked[col]).all() else np.nanmean(fr_attacked_rec_def_attacked[col])
        row[f'{col}_clear_attacked'] = np.nan if np.isnan(fr_clear_attacked[col]).all() else np.nanmean(fr_clear_attacked[col])
        row[f'{col}_clear_rec-clear'] = np.nan if np.isnan(fr_clear_rec_clear[col]).all() else np.nanmean(fr_clear_rec_clear[col])
        row[f'{col}_attacked_rec-attacked'] = np.nan if np.isnan(fr_attacked_rec_attacked[col]).all() else np.nanmean(fr_attacked_rec_attacked[col])    

        row[f'{col}_rec-clear_rec-attacked'] = np.mean(fr_rec_clear_rec_attacked[col])     
        row[f'{col}_defended-rec-clear_defended-rec-attacked'] = np.mean(fr_rec_def_clear_rec_def_attacked[col])

        row[f'{col}_clear_rec-clear_jpeg'] = np.mean(fr_clear_rec_clear_jpeg[col])
        row[f'{col}_attacked_rec-attacked_jpeg'] = np.mean(fr_attacked_rec_attacked_jpeg[col])
        row[f'{col}_rec-clear_rec-attacked_jpeg'] = np.mean(fr_rec_clear_jpeg_rec_attacked_jpeg[col])

        row[f'{col}_clear_rec-clear_jpeg-fix'] = np.mean(fr_clear_rec_clear_jpeg_fix[col])
        row[f'{col}_attacked_rec-attacked_jpeg-fix'] = np.mean(fr_attacked_rec_attacked_jpeg_fix[col])
        row[f'{col}_rec-clear_rec-attacked_jpeg-fix'] = np.mean(fr_rec_clear_jpeg_rec_attacked_jpeg_fix[col])

    for col in NR_COLS:
        row[f'{col}_clear'] = np.nan if np.isnan(nr_clear[col]).all() else np.nanmean(nr_clear[col])
        row[f'{col}_attacked'] = np.nan if np.isnan(nr_attacked[col]).all() else np.nanmean(nr_attacked[col])
        row[f'{col}_rec-attacked'] = np.nan if np.isnan(nr_rec_undefended_attacked[col]).all() else np.nanmean(nr_rec_undefended_attacked[col])
        row[f'{col}_rec-clear'] = np.nan if np.isnan(nr_rec_undefended[col]).all() else np.nanmean(nr_rec_undefended[col])
        row[f'{col}_defended-rec-clear'] = np.nan if np.isnan(nr_rec_defended[col]).all() else np.nanmean(nr_rec_defended[col])      
        row[f'{col}_defended-rec-attacked'] = np.nan if np.isnan(nr_rec_defended_attacked[col]).all() else np.nanmean(nr_rec_defended_attacked[col]) 

        row[f'{col}_rec-attacked_jpeg'] = np.nan if np.isnan(nr_rec_attacked_jpeg[col]).all() else np.nanmean(nr_rec_attacked_jpeg[col])
        row[f'{col}_rec-clear_jpeg'] = np.nan if np.isnan(nr_rec_clear_jpeg[col]).all() else np.nanmean(nr_rec_clear_jpeg[col])

        row[f'{col}_rec-attacked_jpeg-fix'] = np.nan if np.isnan(nr_rec_attacked_jpeg_fix[col]).all() else np.nanmean(nr_rec_attacked_jpeg_fix[col])
        row[f'{col}_rec-clear_jpeg-fix'] = np.nan if np.isnan(nr_rec_clear_jpeg_fix[col]).all() else np.nanmean(nr_rec_clear_jpeg_fix[col]) 

    if dump_path is not None and batch_size == 1 and global_i % dump_freq == 0 and rec_defended_attacked_images is not None:
        #fn = Path(img_names[0]).stem
        cv2.imwrite(os.path.join(dump_path, f'main_rec_att_{Path(fn).stem}.png'), cv2.cvtColor(to_numpy(torch.clamp(rec_defended_attacked_images[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
    if reconstructed_save_path is not None and global_i % save_freq == 0 and rec_defended_attacked_images is not None:
        cv2.imwrite(os.path.join(reconstructed_save_path, f'main_rec_att_{Path(fn).stem}.png'), cv2.cvtColor(to_numpy(torch.clamp(rec_defended_attacked_images[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
    return row, delta_time 


def run(model, undefended_model, defence, dataset_path, test_dataset, attack_callback, save_path='res.csv', batch_size=1, device='cpu', dump_path=None, dump_freq=500,
        variable_params={}, preset=-1, defence_preset=-1, defence_name='no-defence', codec_name='', save_freq=10, attacked_save_path=None, main_codec=None, undef_main_codec=None, 
        is_jpegai=False, loss_func=None, loss_func_name='undefined', reconstructed_save_path=None):
    nr_models = {}
    nr_models['niqe'] = niqe(device)

    # 255 for JPEGAI, 1 for others
    input_range = 1
    if hasattr(model, 'input_range'):
        input_range = model.input_range
    output_range = 1
    if hasattr(model, 'output_range'):
        output_range = model.output_range
    output_cspace = 'rgb' # YCbCr for JPEGAI
    if hasattr(model, 'output_cspace'):
        output_cspace = model.output_cspace
    print(f'INPUT RANGE: {input_range}, OUTPUT RANGE: {output_range}, OUTPUT COLOR SPACE: {output_cspace}')
    time_sum = 0
    attack_num = 0
    total_time = 0
    cur_result_df = pd.DataFrame(columns=RAW_RESULTS_COLS)
    main_codec_result_df = None
    main_codec_total_time = None
    if main_codec is not None:
        main_codec_result_df = pd.DataFrame(columns=RAW_RESULTS_COLS)
        main_codec_total_time = 0
    video_iter = iter_images(dataset_path)
    prev_path = None
    prev_video_name = None
    is_video = None
    global_i = 0
    while True:
        images, video_name, fn, video_path, video_iter, received_video = get_batch(video_iter, batch_size) 
        if is_video is None:
            is_video = received_video
        if video_name != prev_video_name:
            if is_video:
                raise NotImplementedError('Video processing not implemented yet.')
            local_i = 0  
        if images is None:
            break
        images = np.stack(images)
        images = torch.from_numpy(images.astype(np.float32)).permute(0, 3, 1, 2)
        images = images.to(device)

        MULTIPLE = 64
        # TODO: how to transform images to the right shape (multiple of 64)? Crop or resize? or pad?
        method = 'resize'
        if method == 'crop':
            images = images[:, :, :images.size(2) // MULTIPLE * MULTIPLE, :images.size(3) // MULTIPLE * MULTIPLE]
        elif method == 'resize':
            images = torch.nn.functional.interpolate(images, size=(images.size(2) // MULTIPLE * MULTIPLE, images.size(3) // MULTIPLE * MULTIPLE), mode='bilinear')
        elif method == 'pad':
            # pad to the right and bottom so that both dimensions are divisible by 32
            pad_h = (images.size(2) // MULTIPLE + 1) * MULTIPLE - images.size(2)
            pad_w = (images.size(3) // MULTIPLE + 1) * MULTIPLE - images.size(3)
            images = torch.nn.functional.pad(images, (0, pad_w, 0, pad_h), mode='constant', value=0)    
        success_attack = True

        # random seed equal for both clear and attacked
        torch_seed = np.random.randint(low=0, high=999999)

        #print(images.shape)
        # images and attacked_images will have [0,255] range for JPEGAI, [0,1] otherwise
        images = images * input_range

        attack_result = apply_attack(
            model, # it is defended model
            attack_callback,
            images.contiguous(),
            device=device,
            variable_params=variable_params,
            seed=torch_seed,
            is_jpegai=is_jpegai,
            loss_func=loss_func,
            loss_func_name=loss_func_name,
            )
        if attack_result is not None:
            attacked_images, attack_time = attack_result
        else:
            success_attack = False
        if success_attack:
            time_sum += attack_time
            attack_num += 1
            with torch.no_grad():
                attacked_images = attacked_images.to(device)
                images = images.to(device)
                real_bpps = {}
                time_st = time.time()
                torch.manual_seed(torch_seed)
                def_clear_outs = model(images, return_bpp=True)
                rec_defended_images = def_clear_outs['x_hat']
                rec_defended_images = torch.clamp(rec_defended_images, 0, output_range)
                def_clear_bpp = float(def_clear_outs['bpp'])
                real_bpps['real-bpp_defended-clear'] = float(def_clear_outs['real_bpp']) if 'real_bpp' in def_clear_outs.keys() else np.nan

                torch.manual_seed(torch_seed)
                def_attacked_outs = model(attacked_images, return_bpp=True)
                rec_defended_attacked_images = def_attacked_outs['x_hat']
                rec_defended_attacked_images = torch.clamp(rec_defended_attacked_images, 0, output_range)
                def_attacked_bpp = float(def_attacked_outs['bpp'])
                real_bpps['real-bpp_defended-attacked'] = float(def_attacked_outs['real_bpp']) if 'real_bpp' in def_attacked_outs.keys() else np.nan
                total_time += time.time() - time_st

                torch.manual_seed(torch_seed)
                undef_clear_outs = undefended_model(images, return_bpp=True)
                rec_undefended_images = undef_clear_outs['x_hat']
                rec_undefended_images = torch.clamp(rec_undefended_images, 0, output_range)
                undef_clear_bpp = float(undef_clear_outs['bpp'])
                real_bpps['real-bpp_undefended-clear']  = float(undef_clear_outs['real_bpp']) if 'real_bpp' in undef_clear_outs.keys() else np.nan

                torch.manual_seed(torch_seed)
                undef_attacked_outs = undefended_model(attacked_images, return_bpp=True)
                rec_undefended_attacked_images = undef_attacked_outs['x_hat']
                rec_undefended_attacked_images = torch.clamp(rec_undefended_attacked_images, 0, output_range)
                undef_attacked_bpp = float(undef_attacked_outs['bpp'])
                real_bpps['real-bpp_undefended-attacked']  = float(undef_attacked_outs['real_bpp']) if 'real_bpp' in undef_attacked_outs.keys() else np.nan

                print(f'DATA RANGE BEFORE images: {images.min()}, {images.max()}')
                print(f'DATA RANGE BEFORE attacked_images: {attacked_images.min()}, {attacked_images.max()}')
                print(f'DATA RANGE BEFORE rec_defended_images: {rec_defended_images.min()}, {rec_defended_images.max()}')
                print(f'DATA RANGE BEFORE rec_defended_attacked_images: {rec_defended_attacked_images.min()}, {rec_defended_attacked_images.max()}')
                print(f'DATA RANGE BEFORE rec_undefended_images: {rec_undefended_images.min()}, {rec_undefended_images.max()}')
                print(f'DATA RANGE BEFORE rec_undefended_attacked_images: {rec_undefended_attacked_images.min()}, {rec_undefended_attacked_images.max()}')
                # transform images to rgb [0,1] if needed
                images = images / input_range
                attacked_images = attacked_images / input_range
                if is_jpegai:
                    def ycbcr255_to_rgb_01(x):
                        return torch.clamp(ycbcr_to_rgb_255(x) / output_range, 0,1)
                    rec_defended_images = ycbcr255_to_rgb_01(rec_defended_images)
                    rec_defended_attacked_images = ycbcr255_to_rgb_01(rec_defended_attacked_images)
                    rec_undefended_images = ycbcr255_to_rgb_01(rec_undefended_images)
                    rec_undefended_attacked_images = ycbcr255_to_rgb_01(rec_undefended_attacked_images)
                
                print(f'DATA RANGE images: {images.min()}, {images.max()}')
                print(f'DATA RANGE attacked_images: {attacked_images.min()}, {attacked_images.max()}')
                print(f'DATA RANGE rec_defended_images: {rec_defended_images.min()}, {rec_defended_images.max()}')
                print(f'DATA RANGE rec_defended_attacked_images: {rec_defended_attacked_images.min()}, {rec_defended_attacked_images.max()}')
                print(f'DATA RANGE rec_undefended_images: {rec_undefended_images.min()}, {rec_undefended_images.max()}')
                print(f'DATA RANGE rec_undefended_attacked_images: {rec_undefended_attacked_images.min()}, {rec_undefended_attacked_images.max()}')
                # Calculate all fr metrics
                # clear source images vs reconstructed clear (with defence as preprocessing, if provided)
                fr_clear_rec_def_clear = calc_frs(images, rec_defended_images)

                print(f'PSNR CLEAR REC CLEAR: {fr_clear_rec_def_clear["psnr"]}')
                # attacked images vs reconstructed attacked (with defence as preprocessing, if provided)
                fr_attacked_rec_def_attacked = calc_frs(attacked_images, rec_defended_attacked_images)

                # clear vs attacked, wo reconstruction
                fr_clear_attacked = calc_frs(images, attacked_images)

                # reconstructed clear vs reconstructed attacked
                fr_rec_def_clear_rec_def_attacked = calc_frs(rec_defended_images, rec_defended_attacked_images)

                # clear vs reconstructed clear (WITHOUT defence)
                fr_clear_rec_clear = calc_frs(images, rec_undefended_images)

                # clear vs reconstructed clear (WITHOUT defence)
                fr_attacked_rec_attacked = calc_frs(attacked_images, rec_undefended_attacked_images)

                # reconstructed clear vs reconstructed attacked (WITHOUT defence)
                fr_rec_clear_rec_attacked = calc_frs(rec_undefended_images, rec_undefended_attacked_images)

                # Calculate NR metrics
                nr_clear = calc_nrs(images, nr_models)
                nr_attacked = calc_nrs(attacked_images, nr_models)

                nr_rec_undefended_attacked = calc_nrs(rec_undefended_attacked_images, nr_models)
                nr_rec_undefended = calc_nrs(rec_undefended_images, nr_models)

                nr_rec_defended = calc_nrs(rec_defended_images, nr_models)
                nr_rec_defended_attacked = calc_nrs(rec_defended_attacked_images, nr_models)
                
                rec_clear_jpeg, jpeg_clear_bpp = jpeg2k_compress(images, dump_path, fr_clear_rec_def_clear["psnr"], device)
                rec_attacked_jpeg, jpeg_attacked_bpp = jpeg2k_compress(attacked_images, dump_path, fr_clear_rec_def_clear["psnr"], device)
                jpeg_clear_bpp = jpeg_clear_bpp[0]
                jpeg_attacked_bpp = jpeg_attacked_bpp[0]
                # real_bpps['real-bpp_jpeg-clear']  = float(jpeg_clear_bpp)
                # real_bpps['real-bpp_jpeg-attacked']  = float(jpeg_attacked_bpp)

                fr_clear_rec_clear_jpeg = calc_frs(images, rec_clear_jpeg)
                fr_attacked_rec_attacked_jpeg = calc_frs(attacked_images, rec_attacked_jpeg)
                fr_rec_clear_jpeg_rec_attacked_jpeg = calc_frs(rec_clear_jpeg, rec_attacked_jpeg)
                
                nr_rec_attacked_jpeg = calc_nrs(rec_attacked_jpeg, nr_models)
                nr_rec_clear_jpeg = calc_nrs(rec_clear_jpeg, nr_models)

                rec_clear_jpeg_fix, jpeg_clear_bpp_fix = jpeg2k_compress_fix_bpp(images, dump_path, [float(def_clear_bpp)], device)
                rec_attacked_jpeg_fix, jpeg_attacked_bpp_fix = jpeg2k_compress_fix_bpp(attacked_images, dump_path, [float(def_clear_bpp)], device)
                jpeg_clear_bpp_fix = jpeg_clear_bpp_fix[0]
                jpeg_attacked_bpp_fix = jpeg_attacked_bpp_fix[0]
                # real_bpps['real-bpp_jpeg-clear']  = float(jpeg_clear_bpp_fix)
                # real_bpps['real-bpp_jpeg-attacked']  = float(jpeg_attacked_bpp_fix)

                fr_clear_rec_clear_jpeg_fix = calc_frs(images, rec_clear_jpeg_fix)
                fr_attacked_rec_attacked_jpeg_fix = calc_frs(attacked_images, rec_attacked_jpeg_fix)
                fr_rec_clear_jpeg_rec_attacked_jpeg_fix = calc_frs(rec_clear_jpeg_fix, rec_attacked_jpeg_fix)
                
                nr_rec_attacked_jpeg_fix = calc_nrs(rec_attacked_jpeg_fix, nr_models)
                nr_rec_clear_jpeg_fix = calc_nrs(rec_clear_jpeg_fix, nr_models)

            row =  {
                'image_name': Path(video_name).name,
                'codec_name':codec_name,
                'test_dataset': test_dataset,
                'loss_name':loss_func_name,
                'preset' : preset,
                'defence_preset': defence_preset,
                'defence_name':defence_name,
                # BPPs
                'bpp_defended-clear':float(def_clear_bpp),
                'bpp_undefended-clear':float(undef_clear_bpp),
                'bpp_defended-attacked':float(def_attacked_bpp),
                'bpp_undefended-attacked':float(undef_attacked_bpp),
                'bpp_jpeg-clear':float(jpeg_clear_bpp),
                'bpp_jpeg-attacked':float(jpeg_attacked_bpp),
                'bpp_jpeg-clear-fix':float(jpeg_clear_bpp_fix),
                'bpp_jpeg-attacked-fix':float(jpeg_attacked_bpp_fix),
                } 
            row.update(real_bpps)
            for col in FR_COLS:
                row[f'{col}_clear_defended-rec-clear'] = np.mean(fr_clear_rec_def_clear[col])
                row[f'{col}_attacked_defended-rec-attacked'] = np.mean(fr_attacked_rec_def_attacked[col])
                row[f'{col}_clear_attacked'] = np.mean(fr_clear_attacked[col])
                row[f'{col}_clear_rec-clear'] = np.mean(fr_clear_rec_clear[col])
                row[f'{col}_attacked_rec-attacked'] = np.mean(fr_attacked_rec_attacked[col])  

                row[f'{col}_rec-clear_rec-attacked'] = np.mean(fr_rec_clear_rec_attacked[col])     
                row[f'{col}_defended-rec-clear_defended-rec-attacked'] = np.mean(fr_rec_def_clear_rec_def_attacked[col])

                row[f'{col}_clear_rec-clear_jpeg'] = np.mean(fr_clear_rec_clear_jpeg[col])
                row[f'{col}_attacked_rec-attacked_jpeg'] = np.mean(fr_attacked_rec_attacked_jpeg[col])
                row[f'{col}_rec-clear_rec-attacked_jpeg'] = np.mean(fr_rec_clear_jpeg_rec_attacked_jpeg[col])

                row[f'{col}_clear_rec-clear_jpeg-fix'] = np.mean(fr_clear_rec_clear_jpeg_fix[col])
                row[f'{col}_attacked_rec-attacked_jpeg-fix'] = np.mean(fr_attacked_rec_attacked_jpeg_fix[col])
                row[f'{col}_rec-clear_rec-attacked_jpeg-fix'] = np.mean(fr_rec_clear_jpeg_rec_attacked_jpeg_fix[col])

            for col in NR_COLS:
                row[f'{col}_clear'] = np.nan if np.isnan(nr_clear[col]).all() else np.nanmean(nr_clear[col])
                row[f'{col}_attacked'] = np.nan if np.isnan(nr_attacked[col]).all() else np.nanmean(nr_attacked[col])
                row[f'{col}_rec-attacked'] = np.nan if np.isnan(nr_rec_undefended_attacked[col]).all() else np.nanmean(nr_rec_undefended_attacked[col])
                row[f'{col}_rec-clear'] = np.nan if np.isnan(nr_rec_undefended[col]).all() else np.nanmean(nr_rec_undefended[col])
                row[f'{col}_defended-rec-clear'] = np.nan if np.isnan(nr_rec_defended[col]).all() else np.nanmean(nr_rec_defended[col])      
                row[f'{col}_defended-rec-attacked'] = np.nan if np.isnan(nr_rec_defended_attacked[col]).all() else np.nanmean(nr_rec_defended_attacked[col]) 

                row[f'{col}_rec-attacked_jpeg'] = np.nan if np.isnan(nr_rec_attacked_jpeg[col]).all() else np.nanmean(nr_rec_attacked_jpeg[col])
                row[f'{col}_rec-clear_jpeg'] = np.nan if np.isnan(nr_rec_clear_jpeg[col]).all() else np.nanmean(nr_rec_clear_jpeg[col])

                row[f'{col}_rec-attacked_jpeg-fix'] = np.nan if np.isnan(nr_rec_attacked_jpeg_fix[col]).all() else np.nanmean(nr_rec_attacked_jpeg_fix[col])
                row[f'{col}_rec-clear_jpeg-fix'] = np.nan if np.isnan(nr_rec_clear_jpeg_fix[col]).all() else np.nanmean(nr_rec_clear_jpeg_fix[col])

            cur_result_df.loc[len(cur_result_df)] = row

            if main_codec is not None:
                main_codec_row, delta_time_main = create_row(model=main_codec, undefended_model=undef_main_codec, images=images * input_range, attacked_images=attacked_images * input_range,
                                            device=device, torch_seed=torch_seed, nr_models=nr_models, video_name=video_name, codec_name=codec_name, 
                                            test_dataset=test_dataset, preset=preset, defence_preset=defence_preset, defence_name=defence_name,
                                            dump_path=dump_path, batch_size=batch_size, global_i=global_i, dump_freq=dump_freq, fn=fn, is_main=True,
                                            input_range=input_range, output_range=output_range, is_jpegai=is_jpegai, loss_func_name=loss_func_name,
                                            reconstructed_save_path=reconstructed_save_path, save_freq=save_freq)
                main_codec_result_df.loc[len(main_codec_result_df)] = main_codec_row
                main_codec_total_time += delta_time_main
            
            if dump_path is not None and batch_size == 1 and global_i % dump_freq == 0:
                #fn = Path(img_names[0]).stem
                cv2.imwrite(os.path.join(dump_path, f'att_{Path(fn).stem}.png'), cv2.cvtColor(to_numpy(torch.clamp(attacked_images[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(dump_path, f'rec_att_{Path(fn).stem}.png'), cv2.cvtColor(to_numpy(torch.clamp(rec_defended_attacked_images[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(dump_path, f'rec_att_jpeg_{Path(fn).stem}.png'), cv2.cvtColor(to_numpy(torch.clamp(rec_attacked_jpeg[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(dump_path, f'rec_att_jpeg_fix_{Path(fn).stem}.png'), cv2.cvtColor(to_numpy(torch.clamp(rec_attacked_jpeg_fix[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))

            if attacked_save_path is not None and global_i % save_freq == 0:
                cv2.imwrite(os.path.join(attacked_save_path, f'att_{Path(fn).stem}.png'), cv2.cvtColor(to_numpy(torch.clamp(attacked_images[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
                #cv2.imwrite(os.path.join(dump_path, f'{test_dataset}_{fn}.png'), to_numpy(attacked_images).squeeze(0) * 255)

            if reconstructed_save_path is not None and global_i % save_freq == 0:
                cv2.imwrite(os.path.join(reconstructed_save_path, f'rec_att_{Path(fn).stem}.png'), cv2.cvtColor(to_numpy(torch.clamp(rec_defended_attacked_images[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(reconstructed_save_path, f'rec_att_jpeg_{Path(fn).stem}.png'), cv2.cvtColor(to_numpy(torch.clamp(rec_attacked_jpeg[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(reconstructed_save_path, f'rec_att_jpeg_fix_{Path(fn).stem}.png'), cv2.cvtColor(to_numpy(torch.clamp(rec_attacked_jpeg_fix[0],0,1).unsqueeze(0)).squeeze(0) * 255, cv2.COLOR_RGB2BGR))

        local_i += batch_size
        global_i += batch_size

    if attack_num == 0:
        return None
    return cur_result_df, total_time, time_sum, main_codec_result_df, main_codec_total_time


def load_attack_params_csv(attack_name, metric_name, preset_name, defence_name, presets_path='picked_presets.csv'):
    '''
    Used to load params for attack from CSV file with already picked parameters and their values for all bounds
    '''
    assert preset_name >= 0
    if defence_name == 'diff_jpeg':
        defence_name = 'diffjpeg'
    
    presets = pd.read_csv(presets_path)
    cur_attack_preset = presets[presets.attack == attack_name]
    cur_attack_preset = cur_attack_preset[cur_attack_preset.codec == metric_name]
    cur_attack_preset = cur_attack_preset[cur_attack_preset.category == preset_name]
    cur_attack_preset = cur_attack_preset[cur_attack_preset.defence == defence_name]
    if len(cur_attack_preset) == 0:
        raise ValueError(f'Attack: {attack_name}; Metric: {metric_name}; preset: {preset_name}, defence {defence_name} -- NOT FOUND IN CSV {presets_path}')
    
    if len(cur_attack_preset) > 1:
        print(f'[Warning] More than one entry is found for: Attack: {attack_name}; Metric: {metric_name}; preset: {preset_name}, defence {defence_name}. \
               Using first entry.')
    param_val = cur_attack_preset['param_val'].values[0]
    param_name = cur_attack_preset['param_name'].values[0]
    print('Loaded entry:')
    print(cur_attack_preset)

    return {param_name:param_val}
  
def load_defence_params_json(preset_name, presets_path='defence/defence_presets.json'):
    res = {}
    if preset_name != -1:        
        with open(presets_path) as json_file:
            presets = json.load(json_file)
            for param_name in presets:
                res[param_name] = presets[param_name]['presets'][f'{preset_name}']
        return res
    else:
        print(f'[Warning] defence: Preset == -1 was passed: ignoring presets, using global default params')
        return {}

def load_attack_params_json(preset_name, attack_name, presets_path='attack_presets_codecs.json'):
    if preset_name != -1:        
        with open(presets_path) as json_file:
            presets = json.load(json_file)
            cur_preset = presets[attack_name][int(preset_name)]
        return cur_preset
    else:
        print(f'[Warning] defence: Preset == -1 was passed: ignoring presets, using global default params')
        return {}
    
def test_main(attack_callback):
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cpu')
    parser.add_argument("--codec", type=str, required=True)
    parser.add_argument("--test-dataset", type=str, nargs='+')
    parser.add_argument("--dataset-path", type=str, nargs='+')
    parser.add_argument("--dump-path", type=str, default=None)
    parser.add_argument("--dump-freq", type=int, default=500)
    parser.add_argument("--save-path", type=str, default='res.csv')
    parser.add_argument("--log-file", type=str, default=None)
    parser.add_argument("--preset", type=int, default=-1)
    parser.add_argument("--attack", type=str, default='undefined')
    parser.add_argument('--defence-preset', type=int, default=-1)
    #parser.add_argument('--mos-path', type=str, default='/source-dataset/koniq10k_scores.csv')
    parser.add_argument('--run-all-presets', type=int, default=0, help='If ==1, ignore preset argument and run on all presets.')
    parser.add_argument('--only-default-preset', type=int, default=0, help='If ==1, ignore --preset and --run-all-presets args \
                         and only run attack with default parameter vals')
    parser.add_argument("--presets-csv", type=str, default='./picked_presets_all_v1.csv', help='Path to CSV file with preset info.')
    parser.add_argument("--attacked-dataset-path", type=str, default='./attacked-dataset')
    parser.add_argument("--reconstructed-dataset-path", type=str, default=None)
    parser.add_argument("--save-freq", type=int, default=10)
    parser.add_argument("--mainc-save-path", type=str, default="/artifacts")
    parser.add_argument("--mainc-log-file", type=str, default="/artifacts")
    parser.add_argument("--loss_name", type=str, default='undefined')
    args = parser.parse_args()


    loss_func_name = args.loss_name
    loss_func = loss_name_2_func[loss_func_name]

    dir_list = os.listdir('./')
    # prints all files
    print(dir_list)

    codec_addtitional_params = {}
    config_path = 'src/config.json'
    is_jpegai = False
    if 'jpegai' in args.codec:
        is_jpegai = True
        config_path = 'config.json'
    
    with open(config_path) as json_file:
        config = json.load(json_file)
        module = config['module']
        for k in config.keys():
            if k != 'module':
                codec_addtitional_params[k] = config[k]
    
    module_path = f'src.{module}'
    if is_jpegai:
        module_path = f'{module}'
    module = importlib.import_module(module_path)
    model = module.CodecModel(args.device, **codec_addtitional_params)
    main_codec = None
    if is_jpegai:
        main_codec = module.CodecModel(args.device, is_main=True, **codec_addtitional_params)
    
    if "setup.sh" in os.listdir('defence'):
        subprocess.run('bash defence/setup.sh', shell=True, check=True)
    dfnce = importlib.import_module(f'defence.defence')

    if os.path.exists('defence/defence_presets.json'):
        defence_args = load_defence_params_json(args.defence_preset)
        print(f'Defence args:{defence_args}')
        defence = dfnce.Defense(**defence_args)
        defended_model = dfnd.CodecModel(model, dfnce.Defense(**defence_args), args.device)
        defended_main_codec = None
        if is_jpegai:
            defended_main_codec = dfnd.CodecModel(main_codec, dfnce.Defense(**defence_args), args.device)
    else:
        print(f'No defence args')
        defence = dfnce.Defense()
        defended_model = dfnd.CodecModel(model, dfnce.Defense(), args.device)
        defended_main_codec = None
        if is_jpegai:
            defended_main_codec = dfnd.CodecModel(main_codec, dfnce.Defense(), args.device)
    defence_name = defended_model.defence.defence_name

    defended_model.eval()
    batch_size = 1

    os.environ['LOSS_NAME'] = str(loss_func_name)
    with open('loss_f.txt', 'w') as t:
        txt = f'LOSS_NAME="{loss_func_name}"'
        t.write(txt)
    if not os.path.exists(args.dump_path):
        os.makedirs(args.dump_path)

    list_of_presets = [args.preset]
    if args.only_default_preset:
        print("[Warning] Ignoring attack preset and run-all-presets arguments, running attack only with default params.")
        list_of_presets = [-1]
    elif args.run_all_presets == 1:
        print("[Warning] Ignoring attack preset argument, running all presets from 0 to 3.")
        list_of_presets = [x for x in range(0,3)]
    
    raw_results_df = pd.DataFrame(columns=RAW_RESULTS_COLS)
    main_codec_full_raw_df = pd.DataFrame(columns=RAW_RESULTS_COLS)
    scores_df = pd.DataFrame(columns=['attack','attack_preset', 'defence_preset', 'score', 'value'])
    scores_main_codec_df = pd.DataFrame(columns=['attack','attack_preset', 'defence_preset', 'score', 'value'])
    total_calls = 0
    total_time = 0

    total_attack_time = 0
    total_attack_calls = 0

    main_codec_total_time = 0
    main_codec_total_calls = 0
    for cur_preset in list_of_presets:
        print(f'======== Current Preset: {cur_preset} ========')
        # load attack configs
        if cur_preset != -1:
            variable_args = load_attack_params_json(cur_preset, args.attack)
        else:
            variable_args = {}
        print(f'Loaded preset {cur_preset}: {variable_args}')

        for test_dataset, dataset_path in zip(args.test_dataset, args.dataset_path):
            csv_name = f'{args.codec}_test.csv'
            csv_save_path = os.path.join(args.save_path, csv_name)

            if args.attacked_dataset_path == '':
                attacked_dset_path = None
            else:
                attacked_dset_path = str(Path(args.attacked_dataset_path) / loss_func_name / defence_name / str(cur_preset) / args.attack / args.codec / test_dataset)
                Path(attacked_dset_path).mkdir(parents=True, exist_ok=True)
            
            if args.reconstructed_dataset_path == '' or args.reconstructed_dataset_path is None:
                reconstructed_dataset_path = None
            else:
                reconstructed_dataset_path = str(Path(args.reconstructed_dataset_path) / loss_func_name / defence_name / str(cur_preset) / args.attack / args.codec / test_dataset)
                Path(reconstructed_dataset_path).mkdir(parents=True, exist_ok=True)
            
            cur_raw_results, mean_time, attack_time, main_codec_raw_df, main_codec_mean_time = run(
                            model=defended_model,
                            undefended_model=model,
                            defence=defence,
                            dataset_path=dataset_path,
                            test_dataset=test_dataset,
                            attack_callback=attack_callback,
                            save_path=csv_save_path,
                            batch_size=batch_size,
                            device=args.device,
                            dump_path=args.dump_path,
                            dump_freq=args.dump_freq,
                            variable_params=variable_args,
                            preset=cur_preset,
                            defence_preset=args.defence_preset,
                            defence_name=defence_name,
                            codec_name=args.codec,
                            save_freq=args.save_freq,
                            attacked_save_path=attacked_dset_path,
                            main_codec=defended_main_codec,
                            undef_main_codec=main_codec,
                            is_jpegai=is_jpegai,
                            loss_func=loss_func,
                            loss_func_name=loss_func_name,
                            reconstructed_save_path=reconstructed_dataset_path
                        )
            total_time += mean_time
            total_calls += 2 * len(cur_raw_results)

            total_attack_time += attack_time
            total_attack_calls += len(cur_raw_results)

            cur_raw_results['attack'] = args.attack
            cur_raw_results['attack_preset'] = cur_preset
            cur_raw_results['attack_preset'] = cur_raw_results['attack_preset'].astype(int)
            cur_raw_results['defence_preset'] = args.defence_preset
            cur_raw_results['defence_preset'] = cur_raw_results['defence_preset'].astype(int)
            if main_codec_raw_df is not None:
                main_codec_total_time += main_codec_mean_time
                main_codec_total_calls += 2 * len(main_codec_raw_df)
                main_codec_raw_df['attack'] = args.attack
                main_codec_raw_df['attack_preset'] = cur_preset
                main_codec_raw_df['attack_preset'] = main_codec_raw_df['attack_preset'].astype(int)
                main_codec_raw_df['defence_preset'] = args.defence_preset
                main_codec_raw_df['defence_preset'] = main_codec_raw_df['defence_preset'].astype(int)
                main_codec_full_raw_df = pd.concat([main_codec_full_raw_df, main_codec_raw_df]).reset_index(drop=True)

            # Merge raw results
            raw_results_df = pd.concat([raw_results_df, cur_raw_results]).reset_index(drop=True)

            # Calculate scores
            cur_scores = calc_scores_codec(cur_raw_results)
            cur_scores.loc[len(cur_scores)] = {'score':'mean_time', 'value':mean_time / (2 * len(cur_raw_results))}
            cur_scores.loc[len(cur_scores)] = {'score':'mean_attack_time', 'value':attack_time / len(cur_raw_results)}
            cur_scores['test_dataset'] = test_dataset
            cur_scores['attack'] = args.attack
            cur_scores['attack_preset'] = cur_preset
            cur_scores['attack_preset'] = cur_scores['attack_preset'].astype(int)
            cur_scores['defence_preset'] = args.defence_preset
            cur_scores['defence_preset'] = cur_scores['defence_preset'].astype(int)
            scores_df = pd.concat([scores_df, cur_scores]).reset_index(drop=True)

            if main_codec_raw_df is not None:
                # Calculate scores
                cur_scores_main = calc_scores_codec(main_codec_raw_df)
                cur_scores_main.loc[len(cur_scores_main)] = {'score':'mean_time', 'value':main_codec_mean_time / (2 * len(main_codec_raw_df))}
                cur_scores_main.loc[len(cur_scores_main)] = {'score':'mean_attack_time', 'value':attack_time / len(main_codec_raw_df)}
                cur_scores_main['test_dataset'] = test_dataset
                cur_scores_main['attack'] = args.attack
                cur_scores_main['attack_preset'] = cur_preset
                cur_scores_main['attack_preset'] = cur_scores_main['attack_preset'].astype(int)
                cur_scores_main['defence_preset'] = args.defence_preset
                cur_scores_main['defence_preset'] = cur_scores_main['defence_preset'].astype(int)
                scores_main_codec_df = pd.concat([scores_main_codec_df, cur_scores_main]).reset_index(drop=True)

            # SAVE INDEPENDENT CSV FILES FOR EACH DATASET
            # Save CSVs
            cur_dset_log_name = f'{test_dataset}_log.csv'
            cur_dset_rawdata_name = f'{args.codec}_{test_dataset}_test.csv'
            cur_scores.reset_index(drop=True).to_csv(os.path.join(args.log_file, cur_dset_log_name))
            cur_raw_results.reset_index(drop=True).to_csv(os.path.join(args.save_path, cur_dset_rawdata_name))

            cur_dset_mainc_log_name = f'mainc_{test_dataset}_log.csv'
            cur_dset_mainc_rawdata_name = f'mainc_{test_dataset}_test.csv'
            if main_codec is not None:
                cur_scores_main.reset_index(drop=True).to_csv(os.path.join(args.mainc_log_file, cur_dset_mainc_log_name))
                main_codec_raw_df.reset_index(drop=True).to_csv(os.path.join(args.mainc_save_path, cur_dset_mainc_rawdata_name))
    

    # SAVE FULL CSVS
    #write_log(args.log_file, test_dataset, mean_time, cur_preset)
    total_scores = calc_scores_codec(raw_results_df)
    total_scores.loc[len(total_scores)] = {'score':'mean_time', 'value':total_time / total_calls}
    total_scores.loc[len(total_scores)] = {'score':'mean_time', 'value':total_attack_time / total_attack_calls}
    total_scores['test_dataset'] = 'total'
    total_scores['attack'] = 'total'
    total_scores['attack_preset'] = 'total'
    #total_scores['attack_preset'] = total_scores['attack_preset'].astype(int)
    total_scores['defence_preset'] = args.defence_preset
    total_scores['defence_preset'] = total_scores['defence_preset'].astype(int)
    scores_df = pd.concat([total_scores, scores_df]).reset_index(drop=True)
    if main_codec is not None:
        total_scores_main = calc_scores_codec(main_codec_full_raw_df)
        total_scores_main.loc[len(total_scores_main)] = {'score':'mean_time', 'value':main_codec_total_time / main_codec_total_calls}
        total_scores_main.loc[len(total_scores_main)] = {'score':'mean_time', 'value':total_attack_time / total_attack_calls}
        total_scores_main['test_dataset'] = 'total'
        total_scores_main['attack'] = 'total'
        total_scores_main['attack_preset'] = 'total'
        #total_scores['attack_preset'] = total_scores['attack_preset'].astype(int)
        total_scores_main['defence_preset'] = args.defence_preset
        total_scores_main['defence_preset'] = total_scores_main['defence_preset'].astype(int)
        scores_main_codec_df = pd.concat([total_scores_main, scores_main_codec_df]).reset_index(drop=True)
    # Save CSVs
    log_name = 'log.csv'
    rawdata_name = f'{args.codec}_test.csv'
    scores_df.reset_index(drop=True).to_csv(os.path.join(args.log_file, log_name))
    raw_results_df.reset_index(drop=True).to_csv(os.path.join(args.save_path, rawdata_name))

    mainc_log_name = f'mainc_log.csv'
    mainc_rawdata_name = f'mainc_test.csv'
    if main_codec is not None:
        scores_main_codec_df.reset_index(drop=True).to_csv(os.path.join(args.mainc_log_file, mainc_log_name))
        main_codec_full_raw_df.reset_index(drop=True).to_csv(os.path.join(args.mainc_save_path, mainc_rawdata_name))

    
