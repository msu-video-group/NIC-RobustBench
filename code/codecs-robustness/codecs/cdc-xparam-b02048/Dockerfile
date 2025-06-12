FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/5bqjrS1NI/image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt  https://titan.gml-team.ru:5003/fsdownload/5bqjrS1NI/image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt \
 && rm image-l2-use_weight5-vimeo-d64-t8193-b0.2048-x-cosine-01-float32-aux0.9lpips_2.pt.1

RUN pip install ema-pytorch

COPY src /src
COPY config.json /src