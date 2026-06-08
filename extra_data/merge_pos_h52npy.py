#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import gc
import argparse
import numpy as np
import h5py


SPLITS_DEFAULT = ["train", "valid", "test"]


def collect_h5_files(input_dirs, split_name):
    h5_paths = []
    for d in input_dirs:
        h5_path = os.path.join(d, f"{split_name}.h5")
        if os.path.isfile(h5_path):
            h5_paths.append(h5_path)
        else:
            print(f"[WARN] file not found, skip: {h5_path}")
    return h5_paths


def count_total_segments(h5_paths):
    total = 0
    for h5_path in h5_paths:
        with h5py.File(h5_path, "r") as h5f:
            n = h5f["meta"]["offset_start"].shape[0]
            total += n
            print(f"[INFO] {h5_path} -> {n} segments")
    return total


def read_segments_from_one_h5(h5_path, batch_size=4000):
    with h5py.File(h5_path, "r") as h5f:
        signals = h5f["signals"]
        offset_start = h5f["meta"]["offset_start"]
        offset_end = h5f["meta"]["offset_end"]

        n_segments = offset_start.shape[0]
        print(f"[INFO] reading {h5_path}, n_segments={n_segments}")

        for i in range(0, n_segments, batch_size):
            j = min(i + batch_size, n_segments)

            starts = offset_start[i:j]
            ends = offset_end[i:j]

            if len(starts) == 0:
                continue

            global_start = int(starts[0])
            global_end = int(ends[-1])

            big_chunk = signals[global_start:global_end]

            batch_segments = []
            for s, e in zip(starts, ends):
                local_s = int(s) - global_start
                local_e = int(e) - global_start
                seg = np.array(big_chunk[local_s:local_e], dtype=np.int16)
                batch_segments.append(seg)

            yield batch_segments


def merge_one_split(split_name, input_dirs, output_dir, batch_size=4000):
    print("=" * 80)
    print(f"[INFO] start merging split: {split_name}")

    h5_paths = collect_h5_files(input_dirs, split_name)
    if not h5_paths:
        print(f"[WARN] no h5 files found for split={split_name}, skip")
        return

    total_segments = count_total_segments(h5_paths)
    print(f"[INFO] total segments for {split_name}: {total_segments}")

    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{split_name}.npy")

    merged = np.empty(total_segments, dtype=object)

    write_idx = 0
    for h5_path in h5_paths:
        for batch_segments in read_segments_from_one_h5(h5_path, batch_size=batch_size):
            n = len(batch_segments)
            merged[write_idx:write_idx + n] = batch_segments
            write_idx += n

            print(
                f"\r[INFO] {split_name}: written {write_idx}/{total_segments}",
                end="",
                flush=True,
            )

            del batch_segments
            gc.collect()

    print()

    if write_idx != total_segments:
        print(
            f"[WARN] split={split_name}, write_idx={write_idx} != total_segments={total_segments}"
        )

    np.save(out_path, merged, allow_pickle=True)
    print(f"[DONE] saved: {out_path}, shape={merged.shape}, dtype={merged.dtype}")

    del merged
    gc.collect()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge train/valid/test h5 signal segments into npy files."
    )

    parser.add_argument(
        "--input_dirs",
        "-i",
        nargs="+",
        required=True,
        help="Input h5 directories. Each directory should contain train.h5, valid.h5, and/or test.h5.",
    )

    parser.add_argument(
        "--output_dir",
        "-o",
        required=True,
        help="Output directory for merged train.npy, valid.npy, and test.npy.",
    )

    parser.add_argument(
        "--batch_size",
        "-b",
        type=int,
        default=4000,
        help="Number of segments read from each h5 file per batch. Default: 4000.",
    )

    parser.add_argument(
        "--splits",
        "-s",
        nargs="+",
        default=SPLITS_DEFAULT,
        help="Splits to merge. Default: train valid test.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    print("[INFO] input dirs:")
    for d in args.input_dirs:
        print(f"  - {d}")

    print(f"[INFO] output dir: {args.output_dir}")
    print(f"[INFO] batch size: {args.batch_size}")
    print(f"[INFO] merging order: {' -> '.join(args.splits)}")

    for split_name in args.splits:
        merge_one_split(
            split_name=split_name,
            input_dirs=args.input_dirs,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

    print("[DONE] all splits finished")


if __name__ == "__main__":
    main()