import argparse
import gc
import random
from pathlib import Path

import numpy as np


def load_npy_one_file(path):
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError:
        return np.load(path, allow_pickle=True)


def save_object_array(path, samples):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.empty(len(samples), dtype=object)
    arr[:] = samples
    np.save(path, arr, allow_pickle=True)


def save_split(output_path, samples, split, overwrite):
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists. Use --overwrite to replace it.")

    save_object_array(output_path, samples)
    print(f"Saved {len(samples)} {split} samples -> {output_path}")
    samples.clear()
    gc.collect()


def iter_valid_signals(input_dir, length, rng, np_rng):
    npy_files = sorted(Path(input_dir).glob("*.npy"))
    if not npy_files:
        raise FileNotFoundError(f"No .npy files found in {input_dir}")

    rng.shuffle(npy_files)
    for file_idx, npy_path in enumerate(npy_files, start=1):
        print(f"Loading {file_idx}/{len(npy_files)}: {npy_path}")
        data = load_npy_one_file(npy_path)
        indices = np.arange(len(data))
        np_rng.shuffle(indices)

        kept = 0
        skipped = 0
        for idx in indices:
            sig = data[idx]
            if len(sig) < length + 1500:
                skipped += 1
                continue

            kept += 1
            yield np.asarray(sig[1500:], dtype=np.float32)

        print(f"Finished {npy_path}: kept={kept}, skipped_short={skipped}")
        del data
        gc.collect()


def process_negative_samples(
    input_dir,
    output_train,
    output_valid,
    output_test,
    train_count,
    valid_count,
    test_count,
    length,
    seed,
    overwrite,
):
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    if train_count is None or valid_count is None or test_count is None or length is None:
        raise ValueError("train_count, valid_count, test_count, and length are required")

    output_paths = {
        "train": output_train,
        "valid": output_valid,
        "test": output_test,
    }
    if not overwrite:
        for split, output_path in output_paths.items():
            if Path(output_path).exists():
                raise FileExistsError(f"{split} output {output_path} already exists. Use --overwrite to replace it.")

    target_counts = {
        "train": int(train_count),
        "valid": int(valid_count),
        "test": int(test_count),
    }
    written_counts = {"test": 0, "train": 0, "valid": 0}
    split_order = ("train", "valid", "test")
    split_idx = 0
    current_samples = []

    for signal in iter_valid_signals(input_dir, int(length), rng, np_rng):
        while split_idx < len(split_order) and written_counts[split_order[split_idx]] >= target_counts[split_order[split_idx]]:
            split_idx += 1
        if split_idx >= len(split_order):
            break

        split = split_order[split_idx]
        current_samples.append(signal)
        written_counts[split] += 1
        if written_counts[split] >= target_counts[split]:
            save_split(output_paths[split], current_samples, split, overwrite)
            split_idx += 1

    if current_samples and split_idx < len(split_order):
        split = split_order[split_idx]
        save_split(output_paths[split], current_samples, split, overwrite)

    print("Done.")
    for split in split_order:
        target = target_counts[split]
        actual = written_counts[split]
        status = "OK" if actual == target else "NOT_ENOUGH"
        print(f"{split}: {actual}/{target} samples, status={status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream negative npy samples into train/valid/test outputs.")
    parser.add_argument("--input_dir", "-i", required=True, help="Directory containing input .npy files")
    parser.add_argument("--train_out", "-tr", default="train.npy", help="Train output .npy path")
    parser.add_argument("--valid_out", "-v", default="valid.npy", help="Valid output .npy path")
    parser.add_argument("--test_out", "-te", default="test.npy", help="Test output .npy path")
    parser.add_argument("--train_count", "-train_count", type=int, required=True, help="Number of train samples")
    parser.add_argument("--valid_count", "-valid_count", type=int, required=True, help="Number of valid samples")
    parser.add_argument("--test_count", "-test_count", type=int, required=True, help="Number of test samples")
    parser.add_argument("--length", "-l", type=int, required=True, help="Signal segment length")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files")
    args = parser.parse_args()

    process_negative_samples(
        args.input_dir,
        args.train_out,
        args.valid_out,
        args.test_out,
        args.train_count,
        args.valid_count,
        args.test_count,
        args.length,
        args.seed,
        args.overwrite,
    )