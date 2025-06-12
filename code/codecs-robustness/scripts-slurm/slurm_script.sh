#!/bin/bash
set -e
codec_name=$1
attack_name=$2
attack_preset=$3
loss_name=$4
#artifacts_path=$5
env_vars_path=$5

source "${env_vars_path}"

# for vmaf
apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -qqy gcc wget unzip zip git dos2unix libgl1 libglib2.0-0
apt update && apt install -y build-essential pkg-config ninja-build nasm doxygen xxd 
pip3 install --no-cache-dir 'meson>=0.56.1'
git clone https://github.com/Netflix/vmaf.git && cd vmaf/libvmaf
meson setup build \
      --buildtype release \
      -Dcpp_std=c++17 \
      -Denable_tests=false

ninja -vC build
ninja -vC build install
export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH
pkg-config --modversion libvmaf
git clone https://github.com/FFmpeg/FFmpeg && cd FFmpeg
./configure --enable-libvmaf
make -j$(nproc)
make install
cd /

#ATTACKED_DATASET_PATH="/test/attacked-dataset-updated/"
#SAVE_RECONSTRUCTED_ARG=""
SAVE_RECONSTRUCTED_ARG="--reconstructed-dataset-path ${RECONSTRUCTED_DATASET_PATH}"

echo "Codec name ${codec_name}"
echo "Attack name ${attack_name}"
echo "Attack preset ${attack_preset}"
echo "Loss name ${loss_name}"
echo "Artifacts path: ${artifacts_path}"
python -c "import os; print('number of cpus: ', len(os.sched_getaffinity(0)))"

cd /
dpkg -i /vqmt-14.1.12839.pro-Linux.deb && vqmt -activate < /vqmt_config.json

cd /workdir/
cp -a "${REPO_CONTAINER_PATH}/codecs/${codec_name}/." "./"
rm Dockerfile
mv config.json ./src

cp -a "${REPO_CONTAINER_PATH}/attacks/utils/." ./
cp "${REPO_CONTAINER_PATH}/attacks/${attack_name}/run.py" "./run.py"
cp -a "${REPO_CONTAINER_PATH}/defences/no_defence/." ./defence

cp "${CONTAINER_SCRIPTS_PATH}/load_weights.py" ./
python ./load_weights.py --codec_name "${codec_name}" --attack_name "${attack_name}" --model_path "${MODEL_WEIGHTS_PATH}"
mkdir -p dumps
mkdir -p artifacts

apt-get clean
apt-get update && apt-get install -y --no-install-recommends build-essential libopenjp2-7-dev
apt-get clean
pip3 install glymur==0.9.3

python ./run.py --test-dataset "${TEST_DATASET_NAMES[@]}" --codec "${codec_name}" --dataset-path "${TEST_DATASET_PATHS[@]}" --reconstructed-dataset-path $RECONSTRUCTED_DATASET_PATH \
      --save-path "./artifacts" --device cuda:0 --dump-path ./dumps --dump-freq 20 --log-file ./artifacts --preset "${attack_preset}" --defence-preset -1 --run-all-presets 0 \
      --attacked-dataset-path "${ATTACKED_DATASET_PATH}"  --save-freq 25 --only-default-preset 0 --attack "${attack_name}" --loss_name "${loss_name}"
source loss_f.txt
zip -r dumps.zip ./dumps
echo "Loss name ${LOSS_NAME}"
mkdir -p "${artifacts_path}/csvs/${LOSS_NAME}/no_defence/${attack_preset}/${attack_name}/${codec_name}/"
mkdir -p "${artifacts_path}/dumps/${LOSS_NAME}/no_defence/${attack_preset}/${attack_name}/${codec_name}/"
mv ./dumps.zip "${artifacts_path}/dumps/${LOSS_NAME}/no_defence/${attack_preset}/${attack_name}/${codec_name}/"
mv ./artifacts/*.csv "${artifacts_path}/csvs/${LOSS_NAME}/no_defence/${attack_preset}/${attack_name}/${codec_name}/"