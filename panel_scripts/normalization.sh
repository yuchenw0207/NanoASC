#!/usr/bin/env bash
set -euo pipefail

# $1 pos folder
# $2 neg folder
# extra args will be passed to normalization.py

pos_root="$1"
neg_root="$2"
shift 2 || true

norm_args=("-c" "0" "$@")

run_fold() {
    root="$1"
    fold="$2"
    fold_dir="${root}/fold${fold}"

    rm -rf "${fold_dir}/train_sw_batches" "${fold_dir}/valid_sw_batches"

    python normalization.py \
        -d "$fold_dir" \
        "${norm_args[@]}"

    python extra_data/merge_data.py \
        --input_dir "${fold_dir}/train_sw_batches" \
        --output_path "${fold_dir}/train_preprocessed.npy"

    python extra_data/merge_data.py \
        --input_dir "${fold_dir}/valid_sw_batches" \
        --output_path "${fold_dir}/valid_preprocessed.npy"
}

for fold in 0 1 2 3; do
    run_fold "$pos_root" "$fold"
done

for fold in 0 1 2 3; do
    run_fold "$neg_root" "$fold"
done
