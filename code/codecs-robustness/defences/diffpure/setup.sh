#!/bin/bash

echo "Uploading......"

wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/JtdTGHGqw/256x256_diffusion_uncond.pt  https://titan.gml-team.ru:5003/fsdownload/JtdTGHGqw/256x256_diffusion_uncond.pt \
 && rm 256x256_diffusion_uncond.pt.1
