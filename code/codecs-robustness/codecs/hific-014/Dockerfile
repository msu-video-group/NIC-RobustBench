FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/6rjeEmt8a/hific_low.pt  https://titan.gml-team.ru:5003/fsdownload/6rjeEmt8a/hific_low.pt \
 && rm hific_low.pt.1

 RUN pip install autograd

COPY src /src
COPY config.json /src