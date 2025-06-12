#!/bin/bash

. "$CI_PROJECT_DIR"/scripts/codec-init.sh

set -euxo pipefail
#cd "subjects/$METRIC_NAME"

# create video pipes

TEST_PATH="$GML_SHARED/video-tests-data/dataset/distorted_and_reference_videos/crowd_run"
TEST_REF="$TEST_PATH/reference/crowd_run_1920x1080_50.mp4"
TEST_DIST="$TEST_PATH/enc_res_x264_mv_fast_2k_crowd_run_33.mp4"
IFS="," read -r RW RH RF <<< $( ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=width,height,nb_read_packets -of csv=p=0 "$TEST_REF" )
IFS="," read -r DW DH DF <<< $( ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=width,height,nb_read_packets -of csv=p=0 "$TEST_DIST" )
if (( RW != DW || RH != DH || RF != DF )); then
  echo Dimensions mismatch reference is ${RW}x${RH}x${RF} distorted is ${DW}x${DH}x${DF} >&2
  exit 1
fi
WIDTH="$RW" HEIGHT="$RH" FRAMES="$RF"

if (( PARAM_NOREF == 0 )); then
  mkfifo "$CACHE/ref"
  ffmpeg -hide_banner -loglevel warning -i "$TEST_REF" -t 1 -f image2pipe -pix_fmt "$PARAM_COLORSPACE" -vcodec rawvideo "$CACHE/ref" -y &
fi
mkfifo "$CACHE/dist"
ffmpeg -hide_banner -loglevel warning -i "$TEST_DIST" -t 1 -f image2pipe -pix_fmt "$PARAM_COLORSPACE" -vcodec rawvideo "$CACHE/dist" -y &


# run metric image

DOCKER_PARAMS=( --init --gpus device="${CUDA_VISIBLE_DEVICES-0}" -t --rm --name "gitlab-$CI_PROJECT_PATH_SLUG-$CI_JOB_ID" )
LAUNCH_PARAMS=()

case "$METRIC_NAME" in

  vmaf)
    DOCKER_PARAMS+=(
      -v "$CACHE/ref":/ref.yuv
      -v "$CACHE/dist":/dist.yuv
    )
    LAUNCH_PARAMS+=( "$PARAM_COLORSPACE" "$WIDTH" "$HEIGHT" /ref.yuv /dist.yuv )
    ;;

  wadiqam|image-similarity-measures)
    cp -a test.py "$CACHE/test.py"
    DOCKER_PARAMS+=(
      -v "$CACHE/test.py":/src/test.py
      -v "$CACHE/dist":/test.dist
      -v "$CACHE/ref":/test.ref
    )
    LAUNCH_PARAMS+=( /src/test.py --dist_path /test.dist --width "$WIDTH" --height "$HEIGHT" --ref_path /test.ref )
    ;;

  *)
    cp -a test.py "$CACHE/test.py"
    DOCKER_PARAMS+=(
      -v "$CACHE/test.py":/test.py
      -v "$CACHE/dist":/test.dist
    )
    LAUNCH_PARAMS+=( /test.py --dist_path /test.dist --width "$WIDTH" --height "$HEIGHT" )
    if (( PARAM_NOREF == 0 )); then
      DOCKER_PARAMS+=( -v "$CACHE/ref":/test.ref )
      LAUNCH_PARAMS+=( --ref_path /test.ref )
    fi
    ;;

esac

docker run "${DOCKER_PARAMS[@]}" "$IMAGE" "${LAUNCH_PARAMS[@]}" | tee "$CI_PROJECT_DIR/$METRIC_NAME.${LAUNCH_ID}.log"
