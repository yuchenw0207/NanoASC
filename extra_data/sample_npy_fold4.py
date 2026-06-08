#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import gc
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np


N_FOLDS = 4


def load_npy(path: Path):
    try:
        return np.load(path, mmap_mode="r", allow_pickle=False)
    except ValueError:
        return np.load(path, allow_pickle=True)


def collect_npy_files(input_dir: Path) -> List[Path]:
    files = sorted(input_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in: {input_dir}")
    return files


def count_samples(npy_files: Sequence[Path]) -> List[int]:
    counts = []
    for npy_path in npy_files:
        data = load_npy(npy_path)
        if data.ndim < 1:
            raise ValueError(f"{npy_path} has invalid shape: {data.shape}")
        counts.append(int(data.shape[0]))
        print(f"[INFO] {npy_path}: {data.shape[0]} samples, shape={data.shape}, dtype={data.dtype}")
        del data
        gc.collect()
    return counts


def select_global_indices(total: int, num: int, seed: int) -> np.ndarray:
    if num <= 0:
        raise ValueError("--num must be greater than 0")
    if num > total:
        raise ValueError(f"--num ({num}) is larger than total samples ({total})")

    rng = np.random.default_rng(seed)
    indices = rng.choice(total, size=num, replace=False)
    rng.shuffle(indices)
    return indices


def group_indices_by_file(
    global_indices: np.ndarray,
    counts: Sequence[int],
) -> List[List[Tuple[int, int]]]:
    cumulative = np.cumsum(np.asarray(counts, dtype=np.int64))
    groups: List[List[Tuple[int, int]]] = [[] for _ in counts]

    for output_idx, global_idx in enumerate(global_indices):
        file_idx = int(np.searchsorted(cumulative, global_idx, side="right"))
        prev_end = 0 if file_idx == 0 else int(cumulative[file_idx - 1])
        local_idx = int(global_idx) - prev_end
        groups[file_idx].append((output_idx, local_idx))

    return groups


def convert_signal(signal, dtype: str):
    arr = np.asarray(signal)
    if dtype == "keep":
        return np.asarray(arr)
    return np.asarray(arr, dtype=np.dtype(dtype))


def load_selected_trimmed_samples(
    npy_files: Sequence[Path],
    groups: Sequence[Sequence[Tuple[int, int]]],
    num: int,
    trim_start: int,
    dtype: str,
) -> List[np.ndarray]:
    if trim_start < 0:
        raise ValueError("--trim-start must be >= 0")

    samples: List[np.ndarray] = [None] * num  # type: ignore[list-item]

    for file_idx, (npy_path, selected) in enumerate(zip(npy_files, groups), start=1):
        if not selected:
            continue

        print(f"[INFO] Loading selected samples from {file_idx}/{len(npy_files)}: {npy_path}")
        data = load_npy(npy_path)

        for output_idx, local_idx in selected:
            signal = convert_signal(data[local_idx], dtype=dtype)
            samples[output_idx] = signal[trim_start:]

        del data
        gc.collect()

    missing = sum(sample is None for sample in samples)
    if missing:
        raise RuntimeError(f"{missing} selected samples were not loaded")

    return samples


def save_object_array(path: Path, samples: Sequence[np.ndarray], overwrite: bool) -> None:
    if path.exists() and not overwrite:
        raise FileExistsError(f"{path} already exists. Use --overwrite to replace it.")

    path.parent.mkdir(parents=True, exist_ok=True)
    out = np.empty(len(samples), dtype=object)
    out[:] = list(samples)
    np.save(path, out, allow_pickle=True)
    print(f"[DONE] saved {len(samples)} samples -> {path}")


def write_four_folds(output_dir: Path, samples: Sequence[np.ndarray], overwrite: bool) -> None:
    indices = np.arange(len(samples))
    buckets = [indices[i::N_FOLDS] for i in range(N_FOLDS)]

    for fold_id in range(N_FOLDS):
        test_idx = buckets[fold_id]
        valid_idx = buckets[(fold_id + 1) % N_FOLDS]
        train_idx = np.concatenate([
            buckets[(fold_id + 2) % N_FOLDS],
            buckets[(fold_id + 3) % N_FOLDS],
        ])

        fold_dir = output_dir / f"fold{fold_id}"
        save_object_array(fold_dir / "train.npy", [samples[i] for i in train_idx], overwrite)
        save_object_array(fold_dir / "valid.npy", [samples[i] for i in valid_idx], overwrite)
        save_object_array(fold_dir / "test.npy", [samples[i] for i in test_idx], overwrite)

        print(
            f"[INFO] fold{fold_id}: "
            f"train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}"
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Sample num signals from a directory of .npy files, trim each signal from "
            "position 1500 onward, and write 4 folds with a 2:1:1 train/valid/test split."
        )
    )
    parser.add_argument("--input_dir", "-i", required=True, help="Directory containing input .npy files")
    parser.add_argument("--num", "-n", type=int, required=True, help="Number of samples to extract")
    parser.add_argument("--output_dir", "-o", required=True, help="Output directory")
    parser.add_argument("--trim-start", type=int, default=1500, help="Trim each signal as signal[trim_start:]. Default: 1500")
    parser.add_argument("--seed", type=int, default=42, help="Random seed. Default: 42")
    parser.add_argument(
        "--dtype",
        default="float32",
        help="Output signal dtype, or 'keep' to keep input dtype. Default: float32",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing fold output files")
    return parser.parse_args()


def main():
    args = parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory not found: {input_dir}")

    npy_files = collect_npy_files(input_dir)
    counts = count_samples(npy_files)
    total = int(sum(counts))
    print(f"[INFO] total merged samples: {total}")

    global_indices = select_global_indices(total=total, num=args.num, seed=args.seed)
    groups = group_indices_by_file(global_indices=global_indices, counts=counts)
    samples = load_selected_trimmed_samples(
        npy_files=npy_files,
        groups=groups,
        num=args.num,
        trim_start=args.trim_start,
        dtype=args.dtype,
    )

    write_four_folds(output_dir=output_dir, samples=samples, overwrite=args.overwrite)
    print("[DONE] all folds finished")


if __name__ == "__main__":
    main()
