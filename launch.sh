#!/bin/bash

set -e
attack_preset=$1
loss_name=$2
attack=$3
codec=$4
GPU_CURRENT=$5

env_vars_path="./code/codecs-robustness/scripts-docker/env_vars.sh"
source $env_vars_path
DOCKER_IMAGE="codecs_main"
DOCKER_IMAGE_JPEGAI="codecs_jpegai"
working_dir="$PWD"

if [ -f "models.zip" ]; then
    echo "Weights downloaded"
else
    echo "Downloading weights..."
    wget --backups=1 -nv https://titan.gml-team.ru:5003/fsdownload/o3kmmUJdU/models.zip \
         https://titan.gml-team.ru:5003/fsdownload/o3kmmUJdU/models.zip && rm models.zip.1
fi

# mapfile -t codecs < $codec_list_path
# attacks=("${attack_name}")

# GPU_CURRENT=0
# echo "Loss ${loss_name}"
# for codec in "${codecs[@]}"; do
#     for attack in "${attacks[@]}"; do
if [[ $codec == *"jpegai"* ]]; then
    echo "${codec} - ${attack} Starting JPEGAI job"
    docker run --gpus "device=${GPU_CURRENT}" --ipc host --rm \
        -v "$working_dir:/test" \
        -e CUDA_VISIBLE_DEVICES=${GPU_CURRENT} \
        --name CODECS_ROBUSTNESS${codec}${attack} \
        "${DOCKER_IMAGE_JPEGAI}" \
        bash -c "cp ${CONTAINER_SCRIPTS_PATH}/docker_script.sh ./ && chmod 777 ./docker_script.sh && ./docker_script.sh ${codec} ${attack} ${attack_preset} ${loss_name} ${CONTAINER_ENV_VARS_PATH}" &
else
    echo "${codec} - ${attack} Starting ORDINARY job"
    docker run --gpus "device=${GPU_CURRENT}" --ipc host --rm \
        -v "$working_dir:/test" \
        -e CUDA_VISIBLE_DEVICES=${GPU_CURRENT} \
        --name CODECS_ROBUSTNESS${codec}${attack} \
        "${DOCKER_IMAGE}" \
        bash -c "cp ${CONTAINER_SCRIPTS_PATH}/docker_script.sh ./ && chmod 777 ./docker_script.sh && ./docker_script.sh ${codec} ${attack} ${attack_preset} ${loss_name} ${CONTAINER_ENV_VARS_PATH}" &
fi
wait
#         GPU_CURRENT=$(( GPU_CURRENT + 1 ))
#     done
# done
# wait