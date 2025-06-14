#!/bin/bash

set -e
attack_preset=$1
loss_name=$2
attack=$3
codec=$4
defence_name=$5
GPU_CURRENT=$6

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

echo -e "\e[1mMake sure that each dataset subfolder contains at least one image!\e[0m"
echo "TEST_DATASET_NAMES: ${TEST_DATASET_NAMES[@]}"
echo "TEST_DATASET_PATHS: ${TEST_DATASET_PATHS[@]}"

if [[ $codec == *"jpegai"* ]]; then
    echo "${codec} - ${attack} - ${defence_name} Starting JPEGAI job"
    docker run --gpus "device=${GPU_CURRENT}" --ipc host --rm \
        -v "$working_dir:/test" \
        -e CUDA_VISIBLE_DEVICES=${GPU_CURRENT} \
        --name CODECS_ROBUSTNESS${codec}${attack} \
        "${DOCKER_IMAGE_JPEGAI}" \
        bash -c "cp ${CONTAINER_SCRIPTS_PATH}/docker_script.sh ./ && chmod 777 ./docker_script.sh && ./docker_script.sh ${codec} ${attack} ${attack_preset} ${loss_name} ${defence_name} ${CONTAINER_ENV_VARS_PATH}" &
else
    echo "${codec} - ${attack} - ${defence_name} Starting ORDINARY job"
    docker run --gpus "device=${GPU_CURRENT}" --ipc host --rm \
        -v "$working_dir:/test" \
        -e CUDA_VISIBLE_DEVICES=${GPU_CURRENT} \
        --name CODECS_ROBUSTNESS${codec}${attack} \
        "${DOCKER_IMAGE}" \
        bash -c "cp ${CONTAINER_SCRIPTS_PATH}/docker_script.sh ./ && chmod 777 ./docker_script.sh && ./docker_script.sh ${codec} ${attack} ${attack_preset} ${loss_name} ${defence_name} ${CONTAINER_ENV_VARS_PATH}" &
fi
wait