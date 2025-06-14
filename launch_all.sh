#!/usr/bin/env bash
set -euo pipefail               

codecs=("jpegai-v71-bop-b0012" "cheng2020-attn-4")
attacks=("ftda" "madc")
losses=("bpp_increase_loss")
defences=("reversible_flip")
attack_preset=0
gpu_id=0

for codec in "${codecs[@]}"; do
  for attack in "${attacks[@]}"; do
    for loss in "${losses[@]}"; do
      for defence in "${defences[@]}"; do
      ./launch.sh "$attack_preset" "$loss" "$attack" "$codec" "$defence" "$gpu_id"
      done
    done
  done
done