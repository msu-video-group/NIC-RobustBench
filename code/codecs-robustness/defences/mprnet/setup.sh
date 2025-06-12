#!/bin/bash

wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/Qo5uhUQh4/mprnet_denoise.pth  https://titan.gml-team.ru:5003/fsdownload/Qo5uhUQh4/mprnet_denoise.pth \
 && rm mprnet_denoise.pth.1
