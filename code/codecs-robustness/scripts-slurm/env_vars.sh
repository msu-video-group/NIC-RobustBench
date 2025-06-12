# This file defines main varibles used in slurm scripts

# path to the folder where artifacts will be saved WITHIN the job container (i.e. mounted path)
export artifacts_path="/test/new-results-10-06/artifacts"
# absolute path to the folder where artifacts will be saved.
export outer_artifacts_path="/scratch/imolodetskikh/users/26k_abu/framework/new-results-10-06/"

# log file to probe if it exists. If exists, skips calculations for this codec-attack-preset set.
# Put non-existing file name here to always restart calculations from scratch.
export probe_file="log123.csv"
export outer_repo_path="/scratch/imolodetskikh/users/26k_abu/framework/code/codecs-robustness"
export outer_scripts_path="${outer_repo_path}/scripts-slurm"
# absolute path to the txt file containing the list of codecs to run. Each line in it - separate codec name
export codec_list_path="${outer_scripts_path}/codecs_debug.txt"
export container_image_path_jpegai="/scratch/imolodetskikh/users/26k_abu/framework/images_new/framework_jpegai+codecs.sqsh"
export container_image_path="/scratch/imolodetskikh/users/26k_abu/framework/images/26k_abu+universal_framework_image+codecs.sqsh"
export working_dir="/scratch/imolodetskikh/users/26k_abu/framework/"

# used inside a job container
export CONTAINER_ENV_VARS_PATH="/test/code/codecs-robustness/scripts-slurm/env_vars.sh"
export ATTACKED_DATASET_PATH="/test/new-results-10-06/attacked-dataset/"
export RECONSTRUCTED_DATASET_PATH="/test/new-results-10-06/attacked-reconstructed-dataset/"
export REPO_CONTAINER_PATH="/test/code/codecs-robustness"
export CONTAINER_SCRIPTS_PATH="${REPO_CONTAINER_PATH}/scripts-slurm"
export MODEL_WEIGHTS_PATH="/test/models/models"
export TEST_DATASET_NAMES=( KODAK 
                            CITYSCAPES 
                            NIPS-100)
export TEST_DATASET_PATHS=(
    /test/datasets/codec-datasets/kodak/ 
    /test/datasets/codec-datasets/city/ 
    /test/datasets/codec-datasets/nips100/
)