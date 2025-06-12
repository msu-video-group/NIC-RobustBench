# This file defines main varibles used in slurm scripts

# path to the folder where artifacts will be saved WITHIN the job container (i.e. mounted path)
export artifacts_path="/test/results/artifacts"

# path to the txt file containing the list of codecs to run. Each line in it - separate codec name
SCRIPT_PATH="$(readlink -f "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
export codec_list_path="$SCRIPT_DIR/codecs.txt"

# used inside a job container
export CONTAINER_ENV_VARS_PATH="/test/code/codecs-robustness/scripts-docker/env_vars.sh"
export ATTACKED_DATASET_PATH="/test/results/attacked-dataset/"
export RECONSTRUCTED_DATASET_PATH="/test/results/attacked-reconstructed-dataset/"
export REPO_CONTAINER_PATH="/test/code/codecs-robustness"
export CONTAINER_SCRIPTS_PATH="${REPO_CONTAINER_PATH}/scripts-docker"
export MODEL_WEIGHTS_PATH="/test/models/models"
export TEST_DATASET_NAMES=( KODAK 
                            CITYSCAPES 
                            NIPS-100)
export TEST_DATASET_PATHS=(
    /test/datasets/codec-datasets/kodak/ 
    /test/datasets/codec-datasets/city/ 
    /test/datasets/codec-datasets/nips100/
)