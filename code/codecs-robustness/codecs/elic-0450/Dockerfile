FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN pip install compressai
RUN pip install thop
RUN pip install ptflops
RUN pip install timm

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/uxpavzo7Q/ELIC_0450_ft_3980_Plateau.pth.tar  https://titan.gml-team.ru:5003/fsdownload/uxpavzo7Q/ELIC_0450_ft_3980_Plateau.pth.tar \
 && rm ELIC_0450_ft_3980_Plateau.pth.tar.1

COPY src /src
COPY config.json /src