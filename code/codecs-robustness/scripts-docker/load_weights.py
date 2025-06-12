import argparse
import subprocess
import os

if __name__ == "__main__":
   parser = argparse.ArgumentParser()
   parser.add_argument("--codec_name", type=str, required=True)

   args = parser.parse_args()
   codec_name = args.codec_name

   if not os.path.isfile("/test/models.zip"):
      subprocess.run("wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/o3kmmUJdU/models.zip \
         https://titan.gml-team.ru:5003/fsdownload/o3kmmUJdU/models.zip && rm models.zip.1", shell=True, text=True, capture_output=True)
   else:
      subprocess.run("cp /test/models.zip models.zip", shell=True, text=True, capture_output=True)

   subprocess.run("unzip models.zip", shell=True, text=True, capture_output=True)
   subprocess.run("cp models/models/* ./", shell=True, text=True, capture_output=True)
   
   if 'qres-vae' in codec_name:
      subprocess.run(f'python -m pip install -e ./src', shell=True, text=True, capture_output=True)
   elif 'fixed-point' in codec_name:
      subprocess.run(f'unzip fixed_point_models.zip', shell=True, text=True, capture_output=True)
      
