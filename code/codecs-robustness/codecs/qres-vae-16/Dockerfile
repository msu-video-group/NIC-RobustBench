FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget git dos2unix libgl1 libglib2.0-0

RUN pip install compressai

COPY src /src
COPY config.json /src

WORKDIR /src
RUN python -m pip install -e .
WORKDIR /