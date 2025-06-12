import torch
import os
from basicsr.archs.rrdbnet_arch import RRDBNet
from basicsr.utils.download_util import load_file_from_url
from defence.realesrgan import RealESRGANer
from defence.realesrgan.archs.srvgg_arch import SRVGGNetCompact
import numpy as np


class RealESRGANDefense:
    def load_model(self, model_name, model_path):
        # determine models according to model names
        if model_name == 'RealESRGAN_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth']
        elif model_name == 'RealESRNet_x4plus':  # x4 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth']
        elif model_name == 'RealESRGAN_x4plus_anime_6B':  # x4 RRDBNet model with 6 blocks
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4)
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth']
        elif model_name == 'RealESRGAN_x2plus':  # x2 RRDBNet model
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
            netscale = 2
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth']
        elif model_name == 'realesr-animevideov3':  # x4 VGG-style model (XS size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=16, upscale=4, act_type='prelu')
            netscale = 4
            file_url = ['https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth']
        elif model_name == 'realesr-general-x4v3':  # x4 VGG-style model (S size)
            model = SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64, num_conv=32, upscale=4, act_type='prelu')
            netscale = 4
            file_url = [
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth',
                'https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth'
            ]

        # determine model paths
        if model_path is not None:
            model_path = model_path
        else:
            model_path = os.path.join('weights', model_name + '.pth')
            if not os.path.isfile(model_path):
                ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
                for url in file_url:
                    # model_path will be updated
                    model_path = load_file_from_url(
                        url=url, model_dir=os.path.join(ROOT_DIR, 'weights'), progress=True, file_name=None)
        return netscale, model, model_path




    def __init__(self, model_name = 'realesr-general-x4v3', 
                        denoise_strength = 0.2, 
                        outscale = 1, 
                        model_path = None, 
                        tile = 0, 
                        tile_pad = 10, 
                        pre_pad = 0,  
                        fp32 = False):
        self.model_name = model_name
        self.denoise_strength = denoise_strength
        self.outscale = outscale
        self.model_path = model_path
        self.tile = tile
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.fp32 = fp32

        self.netscale, self.model, self.model_path = self.load_model(self.model_name, self.model_path)

        self.dni_weight = None
        if self.model_name == 'realesr-general-x4v3' and self.denoise_strength != 1:
            wdn_model_path = self.model_path.replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
            self.model_path = [self.model_path, wdn_model_path]
            self.dni_weight = [self.denoise_strength, 1 - self.denoise_strength]

        print("self.model_path", self.model_path)
        print("self.dni_weight", self.dni_weight)

        self.upsampler = RealESRGANer(
            scale=self.netscale,
            model_path=self.model_path,
            dni_weight=self.dni_weight,
            model=self.model,
            tile=self.tile,
            tile_pad=self.tile_pad,
            pre_pad=self.pre_pad,
            half=not self.fp32)

    def __call__(self, image):
        # print(image)
        # cv2.imwrite('/home/24a_guh@lab.graphicon.ru/original.png', image * 255.)
        
        output, _ = self.upsampler.enhance(image, outscale=self.outscale)

        # print('1111################################################################')
        # print(image)
        # cv2.imwrite('/home/24a_guh@lab.graphicon.ru/realesrgan.png', np.array(output, dtype=np.float32))
        image = output
        # print('2222################################################################')
        # print(image)
        # print('3333################################################################')
        # print(image)
        return image  


class Defense:
    def __init__(self):
        self.defence_name = 'realesrgan'
        self.defence = RealESRGANDefense(model_name='realesr-general-x4v3', fp32=True)


    def __call__(self, image):
        return self.defence(image)
   
