import torch
import os
import numpy as np
from pathlib import Path
import glymur


def jpeg2k_compress(src, dump_path, target_quality, device):
    src = src.clone()
    if src.shape[-1] != 3:
        src = src.permute(0, 2, 3, 1)

    src = (src * 255).clamp(0, 255).byte().detach().cpu().numpy().astype(np.uint8)
    # target_quality = fr_clear_rec_def_clear["psnr"]
    lst = []
    jpeg_bpp = []

    for i in range(len(src)):

        if dump_path is not None:
            path_jpeg = os.path.join(dump_path, f'tmp.jp2')
        else:
            path_jpeg = 'tmp.jp2'
        
        jp2 = glymur.Jp2k(path_jpeg, data=src[i], psnr=[target_quality[i]])
        lst.append(jp2[:])

        info = os.stat(path_jpeg)
        bpp = info.st_size * 8 / (src.shape[1] * src.shape[2])
        os.remove(path_jpeg)
        jpeg_bpp.append(bpp)
    
    images_tensor = torch.from_numpy(np.stack(lst)).float().to(device) / 255.0  # [0, 1]
    images_tensor = images_tensor.permute(0, 3, 1, 2) 

    return images_tensor, jpeg_bpp

def jpeg2k_compress_fix_bpp(src, dump_path, bpp, device):
    src = src.clone()
    if src.shape[-1] != 3:
        src = src.permute(0, 2, 3, 1)

    src = (src * 255).clamp(0, 255).byte().detach().cpu().numpy().astype(np.uint8)
    # target_quality = fr_clear_rec_def_clear["psnr"]
    lst = []
    jpeg_bpp = []

    for i in range(len(src)):

        if dump_path is not None:
            path_jpeg = os.path.join(dump_path, f'tmp.jp2')
        else:
            path_jpeg = 'tmp.jp2'
        
        jp2 = glymur.Jp2k(path_jpeg, data=src[i], cratios=[24 / bpp[i]])
        lst.append(jp2[:])

        info = os.stat(path_jpeg)
        bpp = info.st_size * 8 / (src.shape[1] * src.shape[2])
        os.remove(path_jpeg)
        jpeg_bpp.append(bpp)
    
    images_tensor = torch.from_numpy(np.stack(lst)).float().to(device) / 255.0  # [0, 1]
    images_tensor = images_tensor.permute(0, 3, 1, 2) 

    return images_tensor, jpeg_bpp
