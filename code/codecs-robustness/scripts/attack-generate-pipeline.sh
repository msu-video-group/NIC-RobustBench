#!/bin/bash


. "$CI_PROJECT_DIR"/scripts/attack-init.sh


set -euo pipefail



load_metric_launch_params() {
  METRIC_LAUNCH_PARAMS=()

    if (( PARAM_TRAIN != 0 ));  then
      if [[ "$1" = paq2piq || "$1" = spaq || "$1" = cvrkd-iqa || "$1" = eonss || "$1" = fid || "$1" = iqt || "$1" = swin-iqa  || "$1" = liqe || "$1" = ahiq ]]; then
        METRIC_LAUNCH_PARAMS+=( --batch-size 1 )
      elif  [[ "$1" = maniqa && "$2" = generative-uap ]]; then
        METRIC_LAUNCH_PARAMS+=( --batch-size 2 )
      elif  [[ "$1" = maniqa || "$2" = generative-uap ]]; then
        METRIC_LAUNCH_PARAMS+=( --batch-size 4 )
      fi
    fi

}



if [[ "${1:-default}" == "generate" ]]; then
  set -e
  GPU_COUNTER=0
  for method in "${METHODS[@]}"; do

    {

      {
cat <<EOF
include:
  - local: scripts/runner.yml

stages:
  - build
  - train
  - test

EOF
      }

      load_method_trainable "$method"
      
      load_method_multimetric "$method"
      
      
      
      if (( METHOD_MULTIMETRIC != 0 )); then 
      
cat <<EOF_OUTER
${method}_common:build:
  allow_failure: false
  tags:
    - storage
  extends: .common
  stage: build
  when: manual
  script:
    - apk add --no-cache bash jq zip
    - ./scripts/attack-build.sh
  retry: 2
EOF_OUTER
      fi

      for metric in "${METRICS[@]}"; do
        {
      if (( METHOD_MULTIMETRIC == 0 )); then 
      
cat <<EOF_OUTER
${method}_${metric}:build:
  allow_failure: false
  tags:
    - storage
  extends: .common
  stage: build
  when: manual
  script:
    - apk add --no-cache bash jq zip
    - ./scripts/attack-build.sh
  retry: 2
EOF_OUTER

          image_metric="$metric"
      else
          image_metric="common"
      fi
      


if (( METHOD_TRAINABLE != 0 )); then cat <<EOF_OUTER
${method}_${metric}:train:
  allow_failure: false
  tags:
    - jpeg-ai
  extends: .common
  stage: train
  needs:
    - job: ${method}_${image_metric}:build
      artifacts: false
  variables:
    PARAM_TRAIN: 1
  script:
    - apk add --no-cache bash jq zip
    - ./scripts/attack-test.sh
  when: on_success
  artifacts:
    name: ${method}_${metric}_train
    paths:
      - "*.log"
      - "*.npy"
      - "*.png"
    when: always
EOF_OUTER
fi

cat <<EOF_OUTER 
${method}_${metric}:test:
  allow_failure: false
  when: on_success
  tags:
    - jpeg-ai
  extends: .common
  stage: test
  needs:
    - job: ${method}_${image_metric}:build
      artifacts: false
EOF_OUTER
if (( METHOD_TRAINABLE != 0 )); then cat <<EOF_OUTER
    - job: ${method}_${metric}:train
EOF_OUTER
fi
cat <<EOF_OUTER 
  variables:
    PARAM_TRAIN: 0
  script:
    - apk add --no-cache bash jq zip
    - ./scripts/attack-test.sh
  artifacts:
    name: ${method}_${metric}_test
    paths:
      - "*.log"
      - "*.csv"
      - "*.zip"
    when: always



EOF_OUTER
        }
        (( GPU_COUNTER += 1 ))
      done


    } > "pipeline-${method}.yml"

  done
fi
