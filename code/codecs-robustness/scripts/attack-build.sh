#!/bin/bash

. "$CI_PROJECT_DIR"/scripts/attack-init.sh

set -euxo pipefail

cd "attacks/$METHOD_NAME"
load_method_multimetric "$METHOD_NAME"

load_cadv "$METHOD_NAME"

load_method_trainable "$METHOD_NAME"

DEFENCE='no_defence'

load_diffpure "$DEFENCE"

load_disco "$DEFENCE"

# to work on LOMONOSOV-270. user must be maindev 
DOCKER_PARAMS=( )
if [[ $GML_SHARED == *"maindev"* ]]; then
  DOCKER_PARAMS=( --add-host=titan.gml-team.ru:10.32.0.32 )
fi

cp -a "$CI_PROJECT_DIR"/attacks/utils/. ./
cp -a "$CI_PROJECT_DIR"/defences/"$DEFENCE"/ ./defence

ls -l 
printf "\nCOPY /defence ./defence\n" >> Dockerfile

if (( METHOD_DISCO != 0 )); then
    printf "\nRUN pip3 install geotorch\n">> Dockerfile
    printf "\nRUN pip3 install torchdiffeq\n">> Dockerfile
    printf "\nRUN pip3 install tensorboardX\n">> Dockerfile
fi

if (( METHOD_DIFFPURE != 0 )); then 

    printf "\nRUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        autoconf \
        automake \
        libtool \
        pkg-config \
        ca-certificates \
        wget \
        git \
        curl \
        ca-certificates \
        libjpeg-dev \
        libpng-dev \
        python \
        python3-dev \
        python3-pip \
        python3-setuptools \
        zlib1g-dev \
        swig \
        cmake \
        vim \
        locales \
        locales-all \
        screen \
        zip \
        unzip\n">> Dockerfile
    printf "\nRUN apt-get clean\n">> Dockerfile

    printf "\nRUN pip3 install numpy==1.19.4 \
                pyyaml==5.3.1 \
                wheel==0.34.2 \
                pillow==7.2.0 \
                matplotlib==3.3.0 \
                tqdm==4.56.1 \
                tensorboardX==2.0 \
                seaborn==0.10.1 \
                pandas \
		        opencv-python \
                requests==2.25.0 \
                xvfbwrapper==0.2.9 \
                torchdiffeq==0.2.1 \                
                scikit-image==0.19 \
                timm \
                lmdb \
                Ninja \
                foolbox \
                torchsde \
                git+https://github.com/RobustBench/robustbench.git@v1.0\n">> Dockerfile
        
         printf "\nRUN pip3 install scipy==1.10\n">> Dockerfile

fi

printf "\nRUN apt-get clean\n">> Dockerfile
printf "\nRUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libopenjp2-7-dev\n">> Dockerfile
printf "\nRUN apt-get clean\n">> Dockerfile
# printf "\nRUN rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/* /var/cache/apt/archives/* && apt-get clean\n">> Dockerfile
# printf "\nRUN apt-get -o Acquire::AllowInsecureRepositories=true update && \
#     apt-get install -y --no-install-recommends \
#     build-essential \
#     libopenjp2-7-dev && \
#     rm -rf /var/lib/apt/lists/*\n">> Dockerfile


printf "\nRUN pip3 install glymur==0.9.3\n">> Dockerfile

# add ffmpeg with vmaf installation to dockerfile
cat >> Dockerfile <<'EOF'
RUN apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget unzip zip git dos2unix libgl1 libglib2.0-0

RUN apt update && apt install -y build-essential pkg-config ninja-build nasm doxygen xxd \
    && pip3 install --no-cache-dir 'meson>=0.56.1'
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
EOF

if (( METHOD_TRAINABLE != 0 )); then
    printf "\nCOPY train.py /train.py\n" >> Dockerfile
fi

if (( METHOD_CADV != 0)); then
    printf "\nRUN wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/ydzYpLFwY/cadv-colorization-model.pth  https://titan.gml-team.ru:5003/fsdownload/ydzYpLFwY/cadv-colorization-model.pth \
 && rm cadv-colorization-model.pth.1\n" >> Dockerfile
fi

if (( METHOD_MULTIMETRIC != 0 )); then 
    for i in "${METRICS[@]}"
    do
    	printf "\nCOPY --from=${NEW_CI_REGISTRY}/metric/${i}:${LAUNCH_ID} /src /${i}/\n" >> Dockerfile
    	weights=($(jq -r '.weight' "${CI_PROJECT_DIR}/subjects/${i}/config.json"  | tr -d '[]," '))
    	for fn in "${weights[@]}"
        do
            printf "\nCOPY --from=${NEW_CI_REGISTRY}/metric/${i}:${LAUNCH_ID} /${fn} /${i}/${fn}\n" >> Dockerfile
        done
    done
    printf "\nCOPY run.py /run.py\n" >> Dockerfile
    docker build -t "$IMAGE" "${DOCKER_PARAMS[@]}" .
    
else
    docker build -t "$IMAGE" "${DOCKER_PARAMS[@]}" --build-arg METRIC_IMAGE="$NEW_CI_REGISTRY/metric/$METRIC_NAME:$LAUNCH_ID" .
fi
docker push "$IMAGE"
