#!/bin/bash

echo "Uploading......"

wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/zBJ7GXlz0/disco_pgd.pth  https://titan.gml-team.ru:5003/fsdownload/zBJ7GXlz0/disco_pgd.pth \
 && rm disco_pgd.pth.1
