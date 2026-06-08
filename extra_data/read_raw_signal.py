#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os

import numpy as np

try:
    from ont_fast5_api.fast5_interface import get_fast5_file
except ImportError:
    get_fast5_file = None

try:
    import pod5
except ImportError:
    pod5 = None


def load_read_ids(read_ids_path):
    ids = set()
    with open(read_ids_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(line.split()[0])
    return ids


def collect_files(file_dir, suffix):
    paths = []
    for root, _, files in os.walk(file_dir):
        for file_name in files:
            if file_name.lower().endswith(suffix):
                paths.append(os.path.join(root, file_name))
    return sorted(paths)


def read_fast5(file_dir, min_length, size, ids=None):
    if get_fast5_file is None:
        raise ImportError("ont-fast5-api is not installed.")

    file_count, reads_count, total_len = 0, 0, 0
    accepted_reads, rejected_reads, short_reads = 0, 0, 0
    reads_array = []

    fast5_files = collect_files(file_dir, ".fast5")
    for file_path in fast5_files:
        file_count += 1
        print("current file:", file_path)

        with get_fast5_file(file_path, mode="r") as f5:
            for read_id in f5.get_read_ids():
                read_id = str(read_id)
                reads_count += 1

                if ids is not None and read_id not in ids:
                    rejected_reads += 1
                    continue

                read = f5.get_read(read_id)
                signal = np.asarray(read.get_raw_data())

                if len(signal) < min_length:
                    short_reads += 1
                    continue

                accepted_reads += 1
                total_len += len(signal)
                reads_array.append(signal)

                if accepted_reads % 1000 == 0:
                    print(
                        f"fast5 files count {file_count}, reads count {reads_count}, "
                        f"accepted reads {accepted_reads}, rejected reads {rejected_reads}, "
                        f"short reads {short_reads}"
                    )

                if accepted_reads >= size:
                    return np.array(reads_array, dtype=object)

    print(
        f"finished fast5 reading: files={file_count}, reads={reads_count}, "
        f"accepted={accepted_reads}, rejected={rejected_reads}, short={short_reads}, "
        f"total_signal_len={total_len}"
    )
    return np.array(reads_array, dtype=object)


def read_pod5(file_dir, min_length, size, ids=None):
    if pod5 is None:
        raise ImportError("pod5 is not installed.")

    file_count, reads_count, total_len = 0, 0, 0
    accepted_reads, rejected_reads, short_reads = 0, 0, 0
    reads_array = []

    pod5_files = collect_files(file_dir, ".pod5")
    for file_path in pod5_files:
        file_count += 1
        print("current file:", file_path)

        with pod5.Reader(file_path) as reader:
            for read in reader.reads():
                read_id = str(read.read_id)
                reads_count += 1

                if ids is not None and read_id not in ids:
                    rejected_reads += 1
                    continue

                signal = np.asarray(read.signal)

                if len(signal) < min_length:
                    short_reads += 1
                    continue

                accepted_reads += 1
                total_len += len(signal)
                reads_array.append(signal)

                if accepted_reads % 1000 == 0:
                    print(
                        f"pod5 files count {file_count}, reads count {reads_count}, "
                        f"accepted reads {accepted_reads}, rejected reads {rejected_reads}, "
                        f"short reads {short_reads}"
                    )

                if accepted_reads >= size:
                    return np.array(reads_array, dtype=object)

    print(
        f"finished pod5 reading: files={file_count}, reads={reads_count}, "
        f"accepted={accepted_reads}, rejected={rejected_reads}, short={short_reads}, "
        f"total_signal_len={total_len}"
    )
    return np.array(reads_array, dtype=object)


def ratio_211_counts(n):
    raw = [n * 2 / 4, n / 4, n / 4]
    counts = [int(x) for x in raw]
    remain = n - sum(counts)

    order = sorted(range(3), key=lambda i: (raw[i] - counts[i], -i), reverse=True)
    for i in order[:remain]:
        counts[i] += 1

    return counts


def get_split_counts(n, train_size, valid_size, test_size):
    requested_total = train_size + valid_size + test_size
    if n >= requested_total:
        return train_size, valid_size, test_size

    train_count, valid_count, test_count = ratio_211_counts(n)
    print(
        f"accepted reads ({n}) less than requested size ({requested_total}), "
        f"use 2:1:1 split: train={train_count}, valid={valid_count}, test={test_count}"
    )
    return train_count, valid_count, test_count


def save_reads(reads, output, train_size, valid_size, test_size, seed):
    rng = np.random.default_rng(seed)
    rng.shuffle(reads)

    train_count, valid_count, test_count = get_split_counts(
        len(reads), train_size, valid_size, test_size
    )

    train_data = reads[:train_count]
    valid_data = reads[train_count:train_count + valid_count]
    test_data = reads[train_count + valid_count:train_count + valid_count + test_count]

    os.makedirs(output, exist_ok=True)

    np.save(os.path.join(output, "train.npy"), train_data, allow_pickle=True)
    print(f"train data saved to {os.path.join(output, 'train.npy')}, shape: {train_data.shape}")

    np.save(os.path.join(output, "valid.npy"), valid_data, allow_pickle=True)
    print(f"valid data saved to {os.path.join(output, 'valid.npy')}, shape: {valid_data.shape}")

    np.save(os.path.join(output, "test.npy"), test_data, allow_pickle=True)
    print(f"test data saved to {os.path.join(output, 'test.npy')}, shape: {test_data.shape}")


def parse_args():
    parser = argparse.ArgumentParser(description="Read raw fast5/pod5 signals and split into train/valid/test npy files.")
    parser.add_argument("--file_dir", "-dir", type=str, required=True, help="Directory containing fast5 or pod5 files")
    parser.add_argument("--input_type", "-type", choices=("fast5", "pod5"), required=True, help="Input file type")
    parser.add_argument("--output", "-o", type=str, required=True, help="Storage path for output files")
    parser.add_argument("--read_ids", "-ids", type=str, default=None, help="Path for read ids file")
    parser.add_argument("--min_length", "-len", type=int, default=4500, help="Minimum signal length, default 4500")
    parser.add_argument("--train_size", "-train", type=int, default=20000, help="Training signal count, default 20000")
    parser.add_argument("--valid_size", "-valid", type=int, default=10000, help="Validation signal count, default 10000")
    parser.add_argument("--test_size", "-test", type=int, default=10000, help="Testing signal count, default 10000")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling, default 42")
    return parser.parse_args()


def main():
    args = parse_args()

    for arg in vars(args):
        print(f"{arg}: {getattr(args, arg)}")

    ids = load_read_ids(args.read_ids) if args.read_ids is not None else None
    total_size = args.train_size + args.valid_size + args.test_size

    if args.input_type == "fast5":
        signals = read_fast5(args.file_dir, args.min_length, total_size, ids)
    else:
        signals = read_pod5(args.file_dir, args.min_length, total_size, ids)

    save_reads(
        reads=signals,
        output=args.output,
        train_size=args.train_size,
        valid_size=args.valid_size,
        test_size=args.test_size,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
