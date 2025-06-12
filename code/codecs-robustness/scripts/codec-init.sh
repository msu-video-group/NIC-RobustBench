#!/bin/bash

METRIC_NAME="${CI_JOB_NAME%:*}"

docker login -u gitlab-ci-token -p "$CI_JOB_TOKEN" "$NEW_CI_REGISTRY"
IMAGE="$NEW_CI_REGISTRY/metric/$METRIC_NAME:$LAUNCH_ID"
