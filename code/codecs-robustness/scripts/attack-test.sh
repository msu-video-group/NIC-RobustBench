#!/bin/bash

. "$CI_PROJECT_DIR"/scripts/attack-init.sh
. "$CI_PROJECT_DIR"/scripts/attack-generate-pipeline.sh

set -euxo pipefail
shopt -s extglob

trap
trap 'echo TRAPPED! "$@"' err
# Path to the storage where attacked dataset will be saved
# TODO: replace with ifs for different runners
# if set to "" images wont save
# ATTACKED_DATASET_PATH="/space/space-lenovo-3/codec-attacks/"
if [[ $GML_SHARED == *"maindev"* ]]; then
  ATTACKED_DATASET_PATH="${GML_SHARED}/DIONE/work/Framework_Datasets/dataset/codecs-attacked-dataset"
elif [[ $GML_SHARED == *"aantsiferova"* ]]; then
  ATTACKED_DATASET_PATH="${GML_SHARED}/DIONE/work/Framework_Datasets/dataset/codecs-attacked-dataset"
elif [[ $GML_SHARED == *"vg_runner"* ]]; then
  ATTACKED_DATASET_PATH="/space/space-lenovo-3/codec-attacks/"
fi

# save each n-th attacked image in 
# ATTACKED_DATASET_PATH / defence name / attack name / attack preset name (-1 == default) / codec / dataset name / image name
ATTACKED_SAVE_FREQ=10

# Path to source images for attacked dataset (not used)
SOURCE_DATASET_PATH="${GML_SHARED}/DIONE/work/Framework_Datasets/dataset/koniq10k"
# CSV for MOSes for SOURCE_DATASET_PATH
MOSES_CSV_NAME="koniq10k_scores.csv"

DATASETS_STORAGE="${GML_SHARED}/DIONE/work/Framework_Datasets/dataset/"
# If set to 1, all presets run single job, ignoring PRESET variable
RUN_ALL_PRESETS=0
# If set to 1, ignore RUN_ALL_PRESETS and PRESET and only run attacks with default parameter values
USE_ONLY_DEFAULT_ATTACK_PRESET=0

PRESET_2_AMPLITUDE=( 0.1 0.18889 0.27778 0.36667 0.45556 0.54444 0.63333 0.72222 0.81111 0.9 )
PRESET=0
LOSS_NAME="reconstr_loss"
PRESET_AMPLITUDE=${PRESET_2_AMPLITUDE[$PRESET]}

# Embedded defence type is specified in attack-build.sh
# -1 for default preset, preset number otherwise 
DEFENCE_PRESET=-1

load_method_trainable_blackbox "$METHOD_NAME"
load_method_non_uap_blackbox "$METHOD_NAME"

if [[ "$METHOD_TRAINABLE_BLACKBOX" == 1 ]]; then
 
	TRAIN_DATASETS=( BLACK-BOX )
	TRAIN_DATASET_PATHS=( 
		"/train/black-box-dataset"
	)
else
	TRAIN_DATASETS=( VOC2012 )
	TRAIN_DATASET_PATHS=( 
		#"/train/COCO_25e_shu/train"
		"/train/VOC2012/JPEGImages"
	)
fi

if [[ "$NON_UAP_BLACKBOX_METHODS" == 1 ]]; then
 
	TEST_DATASETS=( SMALL-KONIQ-50 )
	TEST_DATASET_PATHS=(
		#"/test/black-box-dataset"
    "/test/quality-sampled-datasets/koniq_sampled_MOS/50_10_clusters"
	)
else
	#TEST_DATASETS=( CITYSCAPES )
  TEST_DATASETS=( KODAK )
  TEST_DATASET_PATHS=( 
  #  "/test/DERF"
  #  "/test/small-sampled-datasets/nips_sampled/10"
   # "/test/quality-sampled-datasets/koniq_sampled_MOS/50_10_clusters"
    #"/test/NIPS 2017/images"
    #"/test/quality-sampled-datasets/koniq_sampled_MOS/1000_10_clusters"
    "/test/codec-datasets/kodak"
    #"/test/codec-datasets/city"
    #"/test/codec-datasets/nips100"
  )
fi
 

load_metric_launch_params "$METRIC_NAME" "$METHOD_NAME"
load_method_trainable "$METHOD_NAME"
load_method_multimetric "$METHOD_NAME"
load_video_metric "$METRIC_NAME"




if [[ "$METHOD_NAME" == "noattack" ]]; then

    TEST_DATASETS=( DIV2K_valid_HR )
    TEST_DATASET_PATHS=( 
        "/test/DIV2K_valid_HR"
      )
    
fi



codecs_param="--codec libx264 libx265"



cd "attacks/$METHOD_NAME"
cp -a run.py "$CACHE/"


DUMPS_STORAGE="${CACHE}/dumps"
mkdir -p DUMPS_STORAGE


DOCKER_PARAMS=( --init --gpus device="${CUDA_VISIBLE_DEVICES-0}" -t --rm --name "gitlab-$CI_PROJECT_PATH_SLUG-$CI_JOB_ID" )
if [[ $GML_SHARED == *"maindev"* ]]; then
  DOCKER_PARAMS+=("--add-host=titan.gml-team.ru:10.32.0.32")
fi

if (( PARAM_TRAIN != 0 )); then

    cp -a train.py "$CACHE/"


    docker run "${DOCKER_PARAMS[@]}" \
      -v "$DATASETS_STORAGE":"/train":ro \
      -v "$CACHE:/artifacts" \
      -v "$CACHE/train.py:/train.py" \
      "$IMAGE" \
      python ./train.py \
        "${METRIC_LAUNCH_PARAMS[@]}" \
        --path-train "${TRAIN_DATASET_PATHS[@]}" \
        --codec  "${METRIC_NAME}" \
        --train-dataset "${TRAIN_DATASETS[@]}" \
        --defence-preset "${DEFENCE_PRESET}" \
        --save-dir /artifacts \
        --device "cuda:0" \
      | tee "$CI_PROJECT_DIR/${CI_JOB_NAME//:/_}.$LAUNCH_ID.log"
    
    for train_dataset in "${TRAIN_DATASETS[@]}"; do
            mv "$CACHE/${train_dataset}.npy" "$CI_PROJECT_DIR/${METRIC_NAME}_${train_dataset}.npy"
            mv "$CACHE/${train_dataset}.png" "$CI_PROJECT_DIR/${METRIC_NAME}_${train_dataset}.png"
    done


elif (( METHOD_TRAINABLE != 0 )); then
    
    mkdir -p "$CACHE/uap"
    UAP_PATHS=()
    for train_dataset in "${TRAIN_DATASETS[@]}"; do
        uap_fn="${METRIC_NAME}_${train_dataset}.npy"
        UAP_PATHS+=("/uap/${uap_fn}")
        cp -a "$CI_PROJECT_DIR/${uap_fn}" "$CACHE/uap"
    done
    
    docker run "${DOCKER_PARAMS[@]}" \
      -v "$DATASETS_STORAGE":"/test":ro \
      -v "$CACHE:/artifacts" \
      -v "$CACHE/run.py:/run.py" \
      -v "$CACHE/uap:/uap" \
      -v "$DUMPS_STORAGE":"/dumps" \
      -v "$ATTACKED_DATASET_PATH":"/attacked-dataset" \
      -v "$SOURCE_DATASET_PATH":"/source-dataset":ro \
      "$IMAGE" \
      python ./run.py \
        "${METRIC_LAUNCH_PARAMS[@]}" \
        --amplitude $PRESET_AMPLITUDE \
        --codec  "${METRIC_NAME}" \
        --uap-path "${UAP_PATHS[@]}" \
        --train-dataset "${TRAIN_DATASETS[@]}" \
        --test-dataset "${TEST_DATASETS[@]}" \
        --dataset-path "${TEST_DATASET_PATHS[@]}" \
        --save-path "/artifacts/${METRIC_NAME}_test.csv" \
        --device "cuda:0" \
        --dump-path "/dumps" \
        --dump-freq 50 \
        --log-file "/artifacts/log.csv" \
        --run-all-presets "${RUN_ALL_PRESETS}" \
        --only-default-preset "${USE_ONLY_DEFAULT_ATTACK_PRESET}" \
        --attacked-dataset-path "/attacked-dataset" \
        --save-freq "${ATTACKED_SAVE_FREQ}" \
        --defence-preset "${DEFENCE_PRESET}" \
        --preset $PRESET \
        --attack "${METHOD_NAME}" \
      | tee "$CI_PROJECT_DIR/${CI_JOB_NAME//:/_}.$LAUNCH_ID.txt"
    
    #mv "$CACHE/${METRIC_NAME}_test.csv" "$CI_PROJECT_DIR/"
    
    zip -r -q "$CACHE/dumps.zip" "${DUMPS_STORAGE}"
    mv "$CACHE/dumps.zip" "$CI_PROJECT_DIR/"
    #mv "$CACHE/log.csv" "$CI_PROJECT_DIR/"
    cd "$CACHE" && mv *.csv "$CI_PROJECT_DIR/"

    
  
elif (( METHOD_MULTIMETRIC != 0 )); then 
    docker run "${DOCKER_PARAMS[@]}" \
      -v "$DATASETS_STORAGE":"/test":ro \
      -v "$CACHE:/artifacts" \
      -v "$CACHE/run.py:/run.py" \
      -v "$DUMPS_STORAGE":"/dumps" \
      -v "$ATTACKED_DATASET_PATH":"/attacked-dataset" \
      -v "$SOURCE_DATASET_PATH":"/source-dataset":ro \
      "$IMAGE" \
      python ./run.py \
        --test-dataset "${TEST_DATASETS[@]}" \
        --metric  "${METRIC_NAME}" \
        --dataset-path "${TEST_DATASET_PATHS[@]}" \
        --metric-list "${METRICS[@]}" \
        --target-metric ${METRIC_NAME} \
        $quality_param \
        $video_param \
        $codecs_param \
        --save-path "/artifacts/${METRIC_NAME}_test.csv" \
        --device "cuda:0" \
        --dump-path "/dumps" \
        --dump-freq 50 \
        --log-file "/artifacts/log.csv" \
        --preset $PRESET \
        --run-all-presets "${RUN_ALL_PRESETS}" \
        --defence-preset "${DEFENCE_PRESET}" \
        --attack "${METHOD_NAME}" \
        --mos-path "/source-dataset/${MOSES_CSV_NAME}" \
        --attacked-dataset-path "/attacked-dataset" \
        --save-freq "${ATTACKED_SAVE_FREQ}" \
      | tee "$CI_PROJECT_DIR/${CI_JOB_NAME//:/_}.$LAUNCH_ID.txt"
      
    #mv "$CACHE/${METRIC_NAME}_test.csv" "$CI_PROJECT_DIR/"
    
    zip -r -q "$CACHE/dumps.zip" "${DUMPS_STORAGE}"
    mv "$CACHE/dumps.zip" "$CI_PROJECT_DIR/"
    #mv "$CACHE/log.csv" "$CI_PROJECT_DIR/"
    cd "$CACHE" && mv *.csv "$CI_PROJECT_DIR/"
else

    
    
    docker run "${DOCKER_PARAMS[@]}" \
      -v "$DATASETS_STORAGE":"/test":ro \
      -v "$CACHE:/artifacts" \
      -v "$CACHE/run.py:/run.py" \
      -v "$DUMPS_STORAGE":"/dumps" \
      -v "$ATTACKED_DATASET_PATH":"/attacked-dataset" \
      -v "$SOURCE_DATASET_PATH":"/source-dataset":ro \
      "$IMAGE" \
      python ./run.py \
        "${METRIC_LAUNCH_PARAMS[@]}" \
        --test-dataset "${TEST_DATASETS[@]}" \
        --codec  "${METRIC_NAME}" \
        --dataset-path "${TEST_DATASET_PATHS[@]}" \
        --save-path "/artifacts" \
        --device "cuda:0" \
        --dump-path "/dumps" \
        --dump-freq 5 \
        --log-file "/artifacts" \
        --preset $PRESET \
        --defence-preset "${DEFENCE_PRESET}" \
        --run-all-presets "${RUN_ALL_PRESETS}" \
        --attacked-dataset-path "/attacked-dataset" \
        --save-freq "${ATTACKED_SAVE_FREQ}" \
        --only-default-preset "${USE_ONLY_DEFAULT_ATTACK_PRESET}" \
        --attack "${METHOD_NAME}" \
        --loss_name "${LOSS_NAME}" \
      | tee "$CI_PROJECT_DIR/${CI_JOB_NAME//:/_}.$LAUNCH_ID.txt"
      
    #mv "$CACHE/${METRIC_NAME}_test.csv" "$CI_PROJECT_DIR/"
    
    zip -r -q "$CACHE/dumps.zip" "${DUMPS_STORAGE}"
    mv "$CACHE/dumps.zip" "$CI_PROJECT_DIR/"
    #mv "$CACHE/log.csv" "$CI_PROJECT_DIR/"
    cd "$CACHE" && mv *.csv "$CI_PROJECT_DIR/"


fi
