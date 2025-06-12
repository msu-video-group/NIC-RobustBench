#!/bin/bash

JOB_NAME="${CI_JOB_NAME%:*}"
METRIC_NAME="${JOB_NAME#*_}"
METHOD_NAME="${JOB_NAME%_*}"

docker login -u gitlab-ci-token -p "$CI_JOB_TOKEN" "$NEW_CI_REGISTRY"


load_method_multimetric() {
  case "$1" in
    26k-abu)
      METHOD_MULTIMETRIC=1
      ;;
    *)
      METHOD_MULTIMETRIC=0
      ;;
  esac
}

load_cadv() {
  case "$1" in
    cadv)
      METHOD_CADV=1
      ;;
    *)
      METHOD_CADV=0
      ;;
  esac
}

load_diffpure() {
  case "$1" in
    diffpure)
      METHOD_DIFFPURE=1
      ;;
    *)
      METHOD_DIFFPURE=0
      ;;
  esac
}

load_disco() {
  case "$1" in
    disco)
      METHOD_DISCO=1
      ;;
    *)
      METHOD_DISCO=0
      ;;
  esac
}

load_method_trainable() {
  value_found=0
  for element in "${TRAINABLE_METHODS[@]}"
  do
    if [ "$element" == "$1" ]; then
        value_found=1
        break
    fi
  done
  if [ "$value_found" -eq 1 ]; then
    METHOD_TRAINABLE=1
  else
    METHOD_TRAINABLE=0
  fi
}

load_method_trainable_blackbox() {
  value_found=0
  for element in "${TRAINABLE_BLACKBOX_METHODS[@]}"
  do
    if [ "$element" == "$1" ]; then
        value_found=1
        break
    fi
  done
  if [ "$value_found" -eq 1 ]; then
    METHOD_TRAINABLE_BLACKBOX=1
  else
    METHOD_TRAINABLE_BLACKBOX=0
  fi
}

load_method_non_uap_blackbox() {
  value_found=0
  for element in "${NON_UAP_BLACKBOX_METHODS[@]}"
  do
    if [ "$element" == "$1" ]; then
        value_found=1
        break
    fi
  done
  if [ "$value_found" -eq 1 ]; then
    NON_UAP_BLACKBOX_METHODS=1
  else
    NON_UAP_BLACKBOX_METHODS=0
  fi
}

load_method_multimetric "$METHOD_NAME"
if (( METHOD_MULTIMETRIC != 0 )); then
    IMAGE="$NEW_CI_REGISTRY/attack/$METHOD_NAME:$LAUNCH_ID-common"
else
    IMAGE="$NEW_CI_REGISTRY/attack/$METHOD_NAME:$LAUNCH_ID-$METRIC_NAME"
fi


readarray -t METRICS < "$CI_PROJECT_DIR"/scripts/codecs.txt
readarray -t VIDEO_METRICS < "$CI_PROJECT_DIR"/scripts/video_metrics.txt
readarray -t METHODS < "$CI_PROJECT_DIR"/scripts/methods.txt
readarray -t TRAINABLE_METHODS < "$CI_PROJECT_DIR"/scripts/trainable_methods.txt
readarray -t TRAINABLE_BLACKBOX_METHODS < "$CI_PROJECT_DIR"/scripts/trainable_blackbox_methods.txt
readarray -t NON_UAP_BLACKBOX_METHODS < "$CI_PROJECT_DIR"/scripts/non_uap_blackbox.txt

load_video_metric() {
  value_found=0
  for element in "${VIDEO_METRICS[@]}"
  do
    if [ "$element" == "$1" ]; then
        value_found=1
        break
    fi
  done
  if [ "$value_found" -eq 1 ]; then
    VIDEO_METRIC=1
  else
    VIDEO_METRIC=0
  fi
}
