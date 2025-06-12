#!/bin/bash
set -e
codec_name=$1
attack_name=$2
attack_preset=$3
loss_name=$4
env_vars_path=$5

source "/test/code/codecs-robustness/scripts-docker/env_vars.sh"

SAVE_RECONSTRUCTED_ARG="--reconstructed-dataset-path ${RECONSTRUCTED_DATASET_PATH}"

echo "Codec name ${codec_name}"
echo "Attack name ${attack_name}"
echo "Attack preset ${attack_preset}"
echo "Loss name ${loss_name}"
echo "Artifacts path: ${artifacts_path}"
python -c "import os; print('number of cpus: ', len(os.sched_getaffinity(0)))"

mkdir /workdir && cd /workdir/
cp -a "${REPO_CONTAINER_PATH}/codecs/${codec_name}/." "./"
rm Dockerfile

# apt-get install -qq -y libopenjp2-7-dev
# pip3 install -qq glymur==0.9.3

cp config.json ./src

cp -a "${REPO_CONTAINER_PATH}/attacks/utils/." ./
cp "${REPO_CONTAINER_PATH}/attacks/${attack_name}/run.py" "./run.py"
cp -a "${REPO_CONTAINER_PATH}/defences/no_defence/." ./defence

cp "${CONTAINER_SCRIPTS_PATH}/load_weights.py" ./
python ./load_weights.py --codec_name "${codec_name}"
mkdir -p dumps
mkdir -p artifacts

export TORCH_CUDNN_V8_API_DISABLED=1
if [[ "$codec_name" == *jpegai* ]]; then
  PARAMS=( --mainc-save-path "./artifacts"
           --mainc-log-file "./artifacts" )
else
  PARAMS=()
fi


python ./run.py --test-dataset "${TEST_DATASET_NAMES[@]}" \
      --codec "${codec_name}" \
      --dataset-path "${TEST_DATASET_PATHS[@]}" \
      --reconstructed-dataset-path $RECONSTRUCTED_DATASET_PATH \
      --save-path "./artifacts" \
      --device cuda:0 \
      --dump-path ./dumps \
      --dump-freq 20 \
      --log-file ./artifacts \
      --preset "${attack_preset}" \
      --defence-preset -1 \
      --run-all-presets 0 \
      --attacked-dataset-path "${ATTACKED_DATASET_PATH}"  \
      --save-freq 25 \
      --only-default-preset 0 \
      --attack "${attack_name}" \
      --loss_name "${loss_name}" \
      "${PARAMS[@]}"
source loss_f.txt
zip -r dumps.zip ./dumps
echo "Loss name ${LOSS_NAME}"
mkdir -p "${artifacts_path}/csvs/${LOSS_NAME}/no_defence/${attack_preset}/${attack_name}/${codec_name}/"
mkdir -p "${artifacts_path}/dumps/${LOSS_NAME}/no_defence/${attack_preset}/${attack_name}/${codec_name}/"
mv ./dumps.zip "${artifacts_path}/dumps/${LOSS_NAME}/no_defence/${attack_preset}/${attack_name}/${codec_name}/"
mv ./artifacts/*.csv "${artifacts_path}/csvs/${LOSS_NAME}/no_defence/${attack_preset}/${attack_name}/${codec_name}/"