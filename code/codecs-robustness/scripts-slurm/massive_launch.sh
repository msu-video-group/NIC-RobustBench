#!/bin/bash

# This script launches attacks from the list below with the specified attack-preset (number) 
# and with specified loss function on codecs specified in codecs.txt

# Specify the absolute path to scripts-slurm folder of the directory here
if [[ -z $REPO_PATH ]]; then
    REPO_PATH="/scratch/imolodetskikh/users/26k_abu/framework/code/codecs-robustness"
    echo "Repository path was not specified in REPO_PATH variable, defaulting to ${REPO_PATH}"
fi
env_vars_path="${REPO_PATH}/scripts-slurm/env_vars.sh"

set -e
attack_preset=$1
loss_name=$2
attacks=(
    #'madc-linf'
    #'madc-norm'
    #'ftda-linf'
    #'mad-mix'

    # 'ftda'
    # 'ftda-randn-init'
    # 'madc'
    # 'madc-randn-init'
    # 'ifgsm'
    # 'pgd-ifgsm'
     'ssah'
     'ssah-randn-init'
    #'random-noise'
)
echo "Attack preset ${attack_preset}"
echo "Loss ${loss_name}"
script_path="${REPO_PATH}/scripts-slurm/codecs_run_universal_1_attack.sh"
for attack in "${attacks[@]}"; do
    echo "Launching sbatch for ${attack} attack"
    sbatch --nodes 1 $script_path $attack_preset $loss_name $attack $env_vars_path
done