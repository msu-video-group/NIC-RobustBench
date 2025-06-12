#!/bin/bash

#SBATCH --time=1-8:59:59
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-task=1
#SBATCH --gpus=8
#SBATCH --ntasks=8
set -e
attack_preset=$1
loss_name=$2
attack_name=$3
env_vars_path=$4

source $env_vars_path

echo "CONTAINER_SCRIPTS_PATH ${CONTAINER_SCRIPTS_PATH}"
# artifacts_path="/test/artifacts-updated"
# outer_artifacts_path="~/users/26k_abu/framework/artifacts-updated"
# # file to probe if it exists
# probe_file="log123.csv"
mapfile -t codecs < $codec_list_path
#mapfile -t attacks < ~/users/26k_abu/framework/launches_new/attacks.txt
attacks=("${attack_name}")

echo "Loss ${loss_name}"
for codec in "${codecs[@]}"; do
    for attack in "${attacks[@]}"; do
        echo $codec
        echo $attack
        if [ ! -f $outer_artifacts_path/csvs/$loss_name/no_defence/$attack_preset/$attack/$codec/$probe_file ]; then
            if [[ $codec == *"jpegai"* ]]; then
                echo "${codec} - ${attack} Starting JPEGAI job"
                srun --cpus-per-task 16 --ntasks 1 -G 1 --container-image $container_image_path_jpegai --container-mounts $working_dir:/test bash -c "cp ${CONTAINER_SCRIPTS_PATH}/slurm_script_jpegai.sh ./ && chmod 777 ./slurm_script_jpegai.sh && ./slurm_script_jpegai.sh ${codec} ${attack} ${attack_preset} ${loss_name} ${CONTAINER_ENV_VARS_PATH}" &
                #srun --cpus-per-task 16 --ntasks 1 -G 1 --container-image ~/users/26k_abu/framework/images/26k_abu+jpegai_framework_image_2+latest.sqsh --container-mounts /scratch/imolodetskikh/users/26k_abu/framework/:/test bash -c "echo 'run ${codec} ${attack} ${attack_preset} ${loss_name} ${artifacts_path}'" &
            else
                echo "${codec} - ${attack} Starting ORDINARY job"
                srun --cpus-per-task 16 --ntasks 1 -G 1 --container-image $container_image_path --container-mounts $working_dir:/test bash -c "cp ${CONTAINER_SCRIPTS_PATH}/slurm_script.sh ./ && chmod 777 ./slurm_script.sh && ./slurm_script.sh ${codec} ${attack} ${attack_preset} ${loss_name} ${CONTAINER_ENV_VARS_PATH}" &
                #srun --cpus-per-task 16 --ntasks 1 -G 1 --container-image ~/users/26k_abu/framework/images/26k_abu+jpegai_framework_image_2+latest.sqsh --container-mounts /scratch/imolodetskikh/users/26k_abu/framework/:/test bash -c "echo 'run ${codec} ${attack} ${attack_preset} ${loss_name} ${artifacts_path}'" &
            fi
        else
            echo "${codec} - ${attack} skipped"
        fi
    done
done
wait