#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   bash extra_data/extra_data.sh <raw_path_txt> <split_dir> <out_dir> <panel_bam_dir> <pod5|fast5>
#
# Inputs:
#   $1  txt file containing pod5/fast5 paths, one path per line
#   $2  split directory containing fold0/fold1/fold2/fold3
#       fold_0/fold_1/fold_2/fold_3 are also accepted as a fallback
#   $3  output directory
#   $4  directory containing panel-mapped BAM files
#   $5  raw signal type: pod5 or fast5

if [[ $# -ne 5 ]]; then
    echo "Usage: bash extra_data/extra_data.sh <raw_path_txt> <split_dir> <out_dir> <panel_bam_dir> <pod5|fast5>" >&2
    exit 1
fi

raw_path_txt="$1"
split_root="$2"
out_dir="$3"
bam_dir="$4"
raw_type="$5"

if [[ ! -f "$raw_path_txt" ]]; then
    echo "[ERROR] raw path txt not found: $raw_path_txt" >&2
    exit 1
fi

if [[ ! -d "$split_root" ]]; then
    echo "[ERROR] split directory not found: $split_root" >&2
    exit 1
fi

if [[ ! -d "$bam_dir" ]]; then
    echo "[ERROR] panel BAM directory not found: $bam_dir" >&2
    exit 1
fi

case "$raw_type" in
    pod5)
        pos_signal_script="extra_data/extra_signal_pod.py"
        pos_signal_arg="--pod5_path_txt"
        neg_signal_script="extra_data/extra_neg_pod.py"
        ;;
    fast5)
        pos_signal_script="extra_data/extra_signal_fast5.py"
        pos_signal_arg="--fast5_path_txt"
        neg_signal_script="extra_data/extra_neg_npy.py"
        ;;
    *)
        echo "[ERROR] raw signal type must be 'pod5' or 'fast5', got: $raw_type" >&2
        exit 1
        ;;
esac

if [[ ! -f "$pos_signal_script" ]]; then
    echo "[ERROR] positive signal script not found: $pos_signal_script" >&2
    exit 1
fi

if [[ ! -f "$neg_signal_script" ]]; then
    echo "[ERROR] negative signal script not found: $neg_signal_script" >&2
    exit 1
fi

if ! command -v samtools >/dev/null 2>&1; then
    echo "[ERROR] samtools was not found in PATH" >&2
    exit 1
fi

resolve_fold_dir() {
    local fold_id="$1"
    local compact="${split_root}/fold${fold_id}"
    local underscored="${split_root}/fold_${fold_id}"

    if [[ -d "$compact" ]]; then
        printf "%s\n" "$compact"
    elif [[ -d "$underscored" ]]; then
        printf "%s\n" "$underscored"
    else
        echo "[ERROR] fold directory not found: $compact or $underscored" >&2
        exit 1
    fi
}

require_split_files() {
    local fold_split_dir="$1"
    for split_name in train valid test; do
        if [[ ! -f "${fold_split_dir}/${split_name}.tsv" ]]; then
            echo "[ERROR] missing split file: ${fold_split_dir}/${split_name}.tsv" >&2
            exit 1
        fi
    done
}

count_data_rows() {
    local fold_split_dir="$1"
    local train_lines
    local valid_lines
    local test_lines

    train_lines=$(wc -l < "${fold_split_dir}/train.tsv")
    valid_lines=$(wc -l < "${fold_split_dir}/valid.tsv")
    test_lines=$(wc -l < "${fold_split_dir}/test.tsv")

    printf "%s\n" $((train_lines + valid_lines + test_lines - 3))
}

mkdir -p "${out_dir}/pos"

for fold_id in 0 1 2 3; do
    fold_split_dir="$(resolve_fold_dir "$fold_id")"
    require_split_files "$fold_split_dir"

    pos_fold_dir="${out_dir}/pos/fold${fold_id}"
    h5_dir="${pos_fold_dir}/h5"
    mkdir -p "$h5_dir"

    echo "[INFO] Extracting positive ${raw_type} signals for fold${fold_id}" >&2
    python "$pos_signal_script" \
        "$pos_signal_arg" "$raw_path_txt" \
        --regions_tsv \
        "${fold_split_dir}/train.tsv" \
        "${fold_split_dir}/valid.tsv" \
        "${fold_split_dir}/test.tsv" \
        --output "$h5_dir"

    echo "[INFO] Merging positive h5 files for fold${fold_id}" >&2
    python extra_data/merge_pos_h52npy.py \
        -i "$h5_dir" \
        -o "$pos_fold_dir"
done

neg_dir="${out_dir}/neg"
unmap_by_bam_dir="${neg_dir}/unmap_by_bam"
all_neg_dir="${neg_dir}/all_neg"

rm -rf "$unmap_by_bam_dir" "$all_neg_dir"
mkdir -p "$unmap_by_bam_dir" "$all_neg_dir"

shopt -s nullglob
bam_files=("${bam_dir}"/*.bam "${bam_dir}"/*.BAM)

if [[ ${#bam_files[@]} -eq 0 ]]; then
    echo "[ERROR] no BAM files found in: $bam_dir" >&2
    exit 1
fi

echo "[INFO] Extracting unmapped read IDs from ${#bam_files[@]} BAM files" >&2
for bam in "${bam_files[@]}"; do
    bam_name="$(basename "$bam")"
    sample="${bam_name%.bam}"
    sample="${sample%.BAM}"
    unmap_out="${unmap_by_bam_dir}/${sample}.unmap.txt"

    samtools view -f 4 "$bam" | awk '{print $1}' | sort -u > "$unmap_out"
done

cat "${unmap_by_bam_dir}"/*.unmap.txt | sort -u > "${neg_dir}/unmap.txt"

fold0_split_dir="$(resolve_fold_dir 0)"
require_split_files "$fold0_split_dir"
non_target_num="$(count_data_rows "$fold0_split_dir")"

if [[ "$non_target_num" -le 0 ]]; then
    echo "[ERROR] non_target_num must be > 0, got: $non_target_num" >&2
    exit 1
fi

echo "[INFO] non_target_num=${non_target_num}" >&2
echo "[INFO] Extracting negative ${raw_type} signals" >&2

case "$raw_type" in
    pod5)
        python "$neg_signal_script" \
            "${neg_dir}/unmap.txt" \
            "$raw_path_txt" \
            -o "$all_neg_dir"
        ;;
    fast5)
        python "$neg_signal_script" \
            "${neg_dir}/unmap.txt" \
            "$raw_path_txt" \
            -o "$all_neg_dir" \
            -r
        ;;
esac

echo "[INFO] Sampling and splitting negative signals into 4 folds" >&2
python extra_data/sample_npy_fold4.py \
    -i "$all_neg_dir" \
    -o "$neg_dir" \
    -n "$non_target_num" \
    --overwrite

echo "[DONE] output directory: $out_dir" >&2
