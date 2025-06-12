FROM cnstark/pytorch:1.13.1-py3.9.12-cuda11.7.1-devel-ubuntu20.04

WORKDIR /

RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget unzip git dos2unix libgl1 libglib2.0-0

RUN pip install addict==2.4.0 \
    attrs==21.4.0 \
    decorator==4.4.2 \
    dvc[ssh]==2.10.2 \
    einops==0.8.0 \
    fsspec==2024.6.1 \
    GPUtil==1.4.0 \
    hyperopt==0.2.7 \
    ipython==7.34.0 \ 
    lightgbm==3.3.2 \ 
    numpy==1.26.4 \
    opencv-python==4.10.0.84 \
    openpyxl \
    packaging==24.1 \
    pandas==2.2.2 \
    pre-commit==2.20.0 \
    prettytable==0.7.2 \
    protobuf==4.25.3 \
    psutil==6.0.0 \
    ptflops==0.6.5 \
    pybind11 \
    pynvml==8.0.4 \
    pyrtools==1.0.0 \
    pytorch-msssim==0.2.1 \
    scikit-image==0.24.0 \
    scikit-learn==1.5.1 \
    scipy==1.13.1 \
    setuptools==58.1.0 \
    tensorboard==2.17.0 \
    torch==1.13.1+cu117 \
    torchvision==0.14.1+cu117 \
    wheel==0.34.2 \
    scikit-learn-extra==0.3.0 \
    commentjson==0.9.0

RUN pip install kornia 
ARG CACHEBUST=1
RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/eJDGKllrJ/JPEG_AIv6_main.zip  https://titan.gml-team.ru:5003/fsdownload/eJDGKllrJ/JPEG_AIv6_main.zip \
 && rm JPEG_AIv6_main.zip.1
RUN unzip JPEG_AIv6_main.zip -d ./
RUN echo $(ls ./JPEG_AIv6_main/src/codec)
#COPY JPEG_AI ./
#RUN find ./JPEG_AI | grep -E "(/__pycache__$|\.pyc$|\.pyo$)" | xargs rm -rf
RUN python -Bc "for p in __import__('pathlib').Path('.').rglob('*.py[co]'): p.unlink()"
RUN python -Bc "for p in __import__('pathlib').Path('.').rglob('__pycache__'): p.rmdir()"
RUN echo $(ls)
RUN echo $(ls ./JPEG_AIv6_main)
RUN echo $(ls ./JPEG_AIv6_main/src/codec)
COPY model.py ./
COPY config.json ./
WORKDIR "./JPEG_AIv6_main/src/codec/entropy_coding/cpp_exts/mans/"
RUN make
WORKDIR /
WORKDIR "./JPEG_AIv6_main/src/codec/entropy_coding/cpp_exts/direct/"
RUN make
WORKDIR /