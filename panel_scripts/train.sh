#!/usr/bin/env bash
set -euo pipefail

# $1 pos folder
# $2 neg folder
# $3 out folder
# extra args will be passed to train.py

pos_root="$1"
neg_root="$2"
out_root="$3"
shift 3 || true

train_args=("-c" "0" "-g" "0" "$@")

for fold in 0 1 2 3; do
    python train.py \
        -p "${pos_root}/fold${fold}" \
        -n "${neg_root}/fold${fold}" \
        -o "${out_root}/fold${fold}" \
        "${train_args[@]}"
done
