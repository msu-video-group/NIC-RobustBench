import argparse
import subprocess

codec_2_command = {
    'cdc-xparam-b00032':'image-l2-use_weight5-vimeo-d64-t8193-b0.0032-x-cosine-01-float32-aux0.0_2.pt',
    'cdc-xparam-b01024':'image-l2-use_weight5-vimeo-d64-t8193-b0.1024-x-cosine-01-float32-aux0.9lpips_2.pt',
    'cdc-xparam-b02048':'image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt',
    'elic-0004':'ELIC_0004_ft_3980_Plateau.pth.tar',
    'elic-0016':'ELIC_0016_ft_3980_Plateau.pth.tar',
    'elic-0450':'ELIC_0450_ft_3980_Plateau.pth.tar',
    'evc-1':'EVC_LL.pth.tar',
    'evc-2':'EVC_LL.pth.tar',
    'evc-4':'EVC_LL.pth.tar',
    'evc-6':'EVC_LL.pth.tar',
    'fixed-point':'fixed_point_models.zip',
    'hific-014':'hific_low.pt',
    'hific-030':'hific_med.pt',
    'hific-045':'hific_hi.pt',
    'lic-tcm-00025':'LIC-TCM-64-0.0025.pth.tar',
    'lic-tcm-0013':'LIC-TCM-64-0.013.pth.tar',
    'lic-tcm-005':'LIC-TCM-64-0.05.pth.tar',
}
if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--codec_name", type=str, required=True)
   parser.add_argument("--attack_name", type=str, required=True)
   parser.add_argument("--model_path", type=str, required=False, default="/test/models/models")

   args = parser.parse_args()
   codec_name = args.codec_name
   model_path = args.model_path
   if codec_name in codec_2_command.keys():
      codec_weights_path = f'{model_path}/{codec_2_command[codec_name]}'
      subprocess.run(f'cp {codec_weights_path} ./', shell=True, text=True, capture_output=True)
   elif 'qres-vae' in codec_name:
      subprocess.run(f'python -m pip install -e ./src', shell=True, text=True, capture_output=True)
   elif 'fixed-point' in codec_name:
      subprocess.run(f'unzip fixed_point_models.zip', shell=True, text=True, capture_output=True)
   else:
      print('Codec not found or codec does not require weights.')
   
   if args.attack_name == 'cadv':
      attack_weights_path = f'{model_path}/cadv-colorization-model.pth'
      subprocess.run(f'cp {attack_weights_path} ./', shell=True, text=True, capture_output=True)
   else:
      print('Attack not found or it does not require weights.')