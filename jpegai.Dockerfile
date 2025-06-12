FROM nvcr.io/nvidia/pytorch:24.04-py3

WORKDIR /

RUN apt-get update -q 
RUN apt-get install build-essential -qqy
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget unzip zip git dos2unix libgl1 libglib2.0-0 libopenjp2-7-dev

RUN pip3 install --upgrade pip
RUN pip3 install addict==2.4.0 \
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
    wheel==0.34.2 \
    scikit-learn-extra==0.3.0 \
    commentjson==0.9.0
RUN pip3 install opencv-python
RUN pip3 install numpy
RUN pip3 install tqdm
RUN pip3 install av
RUN pip3 install scipy
RUN pip3 install scikit-image
RUN pip3 install PyWavelets
RUN pip3 install IQA_pytorch
RUN pip3 install frozendict
RUN pip3 install lpips
RUN pip3 install torchmetrics
RUN pip3 install pytorch-wavelets
RUN pip3 install pandas
RUN pip3 install scikit-learn
RUN pip3 install pyiqa
RUN pip3 install pytorch-msssim
RUN pip3 install kornia
RUN pip3 install compressai
RUN pip3 install thop
RUN pip3 install ptflops
RUN pip3 install timm
RUN pip3 install ema-pytorch
RUN pip3 install autograd
RUN pip3 install glymur==0.9.3


RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/UF73KSB8P/JPEG_AIv5_main.zip  https://titan.gml-team.ru:5003/fsdownload/UF73KSB8P/JPEG_AIv5_main.zip \
 && rm JPEG_AIv5_main.zip.1
RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/eJDGKllrJ/JPEG_AIv6_main.zip  https://titan.gml-team.ru:5003/fsdownload/eJDGKllrJ/JPEG_AIv6_main.zip \
 && rm JPEG_AIv6_main.zip.1
RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/8vkiurxbs/JPEG_AIv4_main.zip  https://titan.gml-team.ru:5003/fsdownload/8vkiurxbs/JPEG_AIv4_main.zip \
 && rm JPEG_AIv4_main.zip.1
RUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/DPpBlRNdN/JPEG_AIv7_main.zip  https://titan.gml-team.ru:5003/fsdownload/DPpBlRNdN/JPEG_AIv7_main.zip \
 && rm JPEG_AIv7_main.zip.1

RUN unzip JPEG_AIv5_main.zip -d /workdir
RUN unzip JPEG_AIv6_main.zip -d /workdir
RUN unzip JPEG_AIv4_main.zip -d /workdir
RUN unzip JPEG_AIv7_main.zip -d /workdir
RUN echo $(ls workdir/JPEG_AIv5_main/src/codec)

RUN python -Bc "for p in __import__('pathlib').Path('.').rglob('*.py[co]'): p.unlink()"
RUN python -Bc "for p in __import__('pathlib').Path('.').rglob('__pycache__'): p.rmdir()"
RUN echo $(ls)
RUN echo $(ls workdir/JPEG_AIv5_main)
RUN echo $(ls workdir/JPEG_AIv6_main)
RUN echo $(ls workdir/JPEG_AIv7_main)
WORKDIR "/workdir/JPEG_AIv5_main/src/codec/entropy_coding/cpp_exts/mans/"
RUN make
WORKDIR "/workdir/JPEG_AIv6_main/src/codec/entropy_coding/cpp_exts/mans/"
RUN make
WORKDIR "/workdir/JPEG_AIv6_main/src/codec/entropy_coding/cpp_exts/direct/"
RUN make
WORKDIR "/workdir/JPEG_AIv4_main/src/codec/entropy_coding/cpp_exts/mans/"
RUN make
WORKDIR "/workdir/JPEG_AIv7_main/src/codec/entropy_coding/cpp_exts/mans/"
RUN make
WORKDIR "/workdir/JPEG_AIv7_main/src/codec/entropy_coding/cpp_exts/direct/"
RUN make
WORKDIR /workdir

RUN apt install ocl-icd-libopencl1

RUN pip uninstall -y $(pip list --format=freeze | grep opencv)
RUN rm -rf /usr/local/lib/python3.10/dist-packages/cv2/
RUN pip install opencv-python-headless

RUN apt update && apt install -y build-essential pkg-config meson ninja-build nasm doxygen xxd
RUN git clone https://github.com/Netflix/vmaf.git && cd vmaf/libvmaf \
    && meson setup build \
      --buildtype release \
      -Dcpp_std=c++17 \
      -Denable_tests=false
RUN cd vmaf/libvmaf && ninja -vC build
RUN cd vmaf/libvmaf && ninja -vC build install
RUN export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
RUN pkg-config --modversion libvmaf

RUN git clone https://github.com/FFmpeg/FFmpeg && cd FFmpeg \
 && ./configure --enable-libvmaf \
 && make -j$(nproc) \
 && make install