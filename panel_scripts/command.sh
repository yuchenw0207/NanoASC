#!/usr/bin/env bash
set -euo pipefail

# input raw signal type pod5/fast5
#
# Usage:
#   bash extra_data/command.sh <panel|bed_path> <bam_dir> <out_dir>
#
# <panel|bed_path>
#   148: panel_bed/148_genes.bed
#   odd: panel_bed/odd_cosmic_genes.bed
#   all: panel_bed/all_cosmic_genes.bed
#   or provide a BED file path directly
#
# <bam_dir>
#   Directory containing one or more BAM files.

if [[ $# -ne 3 ]]; then
    echo "Usage: bash extra_data/command.sh <panel|bed_path> <bam_dir> <out_dir>" >&2
    exit 1
fi

panel="$1"
bam_dir="$2"
out_dir="$3"

case "$panel" in
    148)
        bed="panel_bed/148_genes.bed"
        ;;
    odd)
        bed="panel_bed/odd_cosmic_genes.bed"
        ;;
    all)
        bed="panel_bed/all_cosmic_genes.bed"
        ;;
    *)
        bed="$panel"
        ;;
esac

if [[ ! -f "$bed" ]]; then
    echo "[ERROR] BED file not found: $bed" >&2
    exit 1
fi

if [[ ! -d "$bam_dir" ]]; then
    echo "[ERROR] BAM directory not found: $bam_dir" >&2
    exit 1
fi

mkdir -p "$out_dir/seq_position" "$out_dir/signal_position"

shopt -s nullglob
bam_files=("$bam_dir"/*.bam "$bam_dir"/*.BAM)
signal_files=()

if [[ ${#bam_files[@]} -eq 0 ]]; then
    echo "[ERROR] No BAM files found in: $bam_dir" >&2
    exit 1
fi

for bam in "${bam_files[@]}"; do
    bam_name="$(basename "$bam")"
    sample="${bam_name%.bam}"
    sample="${sample%.BAM}"

    seq_out="$out_dir/seq_position/${sample}.seq_pos.tsv"
    signal_out="$out_dir/signal_position/${sample}.signal_pos.tsv"

    echo "[INFO] Processing BAM: $bam" >&2
    python extra_data/extra_seq_pos.py \
        -b "$bed" \
        -a "$bam" \
        -o "$seq_out"

    python extra_data/extra_signal_pos.py \
        "$seq_out" \
        "$bam" \
        -o "$signal_out"

    signal_files+=("$signal_out")
done

awk 'FNR==1 && NR!=1 {next} {print}' "${signal_files[@]}" > "$out_dir/signal_position.tsv"

awk -F'\t' 'BEGIN{OFS="\t"}
NR==1 {
    print;
    next
}
{
    start = ($11 > 1500 ? $11 : 1500);
    len = $12 - start;

    if ($3 != 1 && len >= 3000) {
        print
    }
}' "$out_dir/signal_position.tsv" > "$out_dir/signal_qualified.position.tsv"

rm -f "$out_dir/signal_position.tsv"

mkdir -p "$out_dir/split"

python extra_data/split_fold4.py \
    "$out_dir/signal_qualified.position.tsv" \
    -o "$out_dir/split"
