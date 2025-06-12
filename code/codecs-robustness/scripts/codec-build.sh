#!/bin/bash

. "$CI_PROJECT_DIR"/scripts/codec-init.sh

set -euxo pipefail

apk add git --no-cache

if (( PARAM_IQA_PYTORCH != 0 )); then
    git submodule update --init --recursive --depth=1 subjects/IQA-PyTorch
fi
git submodule sync
git submodule update --init --recursive --depth=1 "codecs/$METRIC_NAME"
cd "codecs/$METRIC_NAME"

# to work on LOMONOSOV-270. user must be maindev 
DOCKER_PARAMS=( )
if [[ $GML_SHARED == *"maindev"* ]]; then
  DOCKER_PARAMS=( --add-host=titan.gml-team.ru:10.32.0.32 )
fi

if (( PARAM_IQA_PYTORCH != 0 )); then
  docker build -t "$IMAGE" "${DOCKER_PARAMS[@]}" -f Dockerfile ..
else
  docker build -t "$IMAGE" "${DOCKER_PARAMS[@]}" "$PARAM_DOCKER_BUILD_PATH"
fi
docker push "$IMAGE"
