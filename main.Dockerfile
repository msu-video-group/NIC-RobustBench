FROM nvcr.io/nvidia/pytorch:24.04-py3

WORKDIR /

RUN apt-get update -q 
RUN apt-get install build-essential -qqy
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget unzip zip git dos2unix libgl1 libglib2.0-0 libopenjp2-7-dev

RUN pip3 install --upgrade pip
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
