#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Optional
try:
    import numpy as np
except ImportError:
    np = None

try:
    import pod5
except ImportError:
    pod5 = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None


DEFAULT_CHUNK_SIZE = 10240


def parse_target_txt(txt_path: str) -> set:
    """
    Parse read IDs from the first column of a text file.
    """
    target_reads = set()
    print(f"===== Parsing read ID list: {txt_path} =====")

    with open(txt_path, "r", encoding="utf-8") as f:
        for _, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = re.split(r"\s+", line)
            read_id = parts[0].strip()
            if not read_id:
                continue

            target_reads.add(read_id)
            if len(target_reads) % 1000 == 0:
                print(f"Parsed {len(target_reads)} read IDs")

    print(f"\nFinished parsing read IDs: {len(target_reads)} total\n")
    return target_reads


def collect_pod5_files(pod5_path: str):
    """
    Collect pod5 files from a single pod5 file or a directory.
    Directory input is searched recursively.
    """
    pod5_files = []

    if os.path.isfile(pod5_path) and pod5_path.endswith(".pod5"):
        pod5_files = [Path(pod5_path).resolve()]
        print(f"Found single pod5 file: {pod5_path}")
        return pod5_files

    if os.path.isdir(pod5_path):
        for root, _, files in os.walk(pod5_path):
            for fn in files:
                if fn.endswith(".pod5"):
                    pod5_files.append(Path(root, fn).resolve())

        print(f"Found {len(pod5_files)} pod5 files under directory: {pod5_path}")
        return sorted(pod5_files)

    raise FileNotFoundError(f"Invalid pod5 path: {pod5_path}")


def normalize_read_id_for_match(read_id: str) -> str:
    """
    Hook for read ID normalization if txt IDs and pod5 IDs differ in format.
    """
    return read_id


def save_signal_chunk(signals, output_dir: str, chunk_idx: int):
    """
    Save one buffered chunk to output_dir/signal{chunk_idx}.npy
    """
    if not signals:
        return None

    arr = np.array(signals, dtype=object)
    output_path = os.path.join(output_dir, f"signal{chunk_idx}.npy")
    np.save(output_path, arr)
    return output_path


def extract_filtered_signals(
    pod5_path: str,
    target_reads: set,
    output_dir: str,
    min_length: int = 5000,
    num: Optional[int] = None,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
):
    """
    Extract signals from pod5 for read IDs listed in txt.

    Signals shorter than min_length are skipped.
    If num is provided, stop immediately after saving num qualified signals.
    """
    if num is not None and num <= 0:
        raise ValueError("--num must be > 0")
    if chunk_size <= 0:
        raise ValueError("--chunk-size must be > 0")

    pod5_files = collect_pod5_files(pod5_path)
    if not pod5_files:
        raise RuntimeError("No pod5 files were found.")

    os.makedirs(output_dir, exist_ok=True)
    existing_npy = [name for name in os.listdir(output_dir) if name.endswith(".npy")]
    if existing_npy:
        print(
            f"[WARN] Output directory already contains {len(existing_npy)} .npy files: {output_dir}",
            file=sys.stderr,
        )

    total_saved = 0
    chunk_idx = 0
    buffer = []
    seen_read_ids = set()

    for p5_path in tqdm(pod5_files, desc="Processing pod5 files"):
        try:
            with pod5.Reader(str(p5_path)) as reader:
                for read in reader.reads():
                    read_id = normalize_read_id_for_match(str(read.read_id))

                    if read_id not in target_reads:
                        continue
                    if read_id in seen_read_ids:
                        continue

                    raw = np.asarray(read.signal, dtype=np.float32)
                    if len(raw) < min_length:
                        continue

                    buffer.append(raw)
                    seen_read_ids.add(read_id)
                    total_saved += 1

                    print(f"\rMatched and buffered signals: {total_saved}", end="")

                    if len(buffer) >= chunk_size:
                        save_signal_chunk(buffer, output_dir, chunk_idx)
                        chunk_idx += 1
                        buffer = []

                    # if num is not None and total_saved >= num:
                    #     if buffer:
                    #         save_signal_chunk(buffer, output_dir, chunk_idx)
                    #         chunk_idx += 1
                    #     print()
                    #     return {
                    #         "saved_count": total_saved,
                    #         "chunk_count": chunk_idx,
                    #         "stopped_early": True,
                    #     }

        except Exception as e:
            print(f"\n[WARN] Failed to read {p5_path}: {e}", file=sys.stderr)
            continue

    if buffer:
        save_signal_chunk(buffer, output_dir, chunk_idx)
        chunk_idx += 1

    print()
    return {
        "saved_count": total_saved,
        "chunk_count": chunk_idx,
        "stopped_early": False,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Extract target read signals from pod5 and save them as chunked npy files."
    )
    parser.add_argument("txt_path", help="Text file whose first column contains target read IDs.")
    parser.add_argument("pod5_path", help="A pod5 file or a directory containing pod5 files.")
    parser.add_argument(
        "-o",
        "--output_dir",
        default="filtered_signals",
        help="Output directory for signal*.npy files. Default: filtered_signals",
    )
    parser.add_argument(
        "-l",
        "--min_length",
        type=int,
        default=5000,
        help="Minimum signal length to keep. Default: 5000",
    )
    parser.add_argument(
        "-n",
        "--num",
        "--count",
        dest="num",
        type=int,
        default=None,
        help="Stop after saving this many qualified signals.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help=f"How many signals to store per npy chunk. Default: {DEFAULT_CHUNK_SIZE}",
    )
    args = parser.parse_args()

    if np is None:
        raise ImportError("numpy is not installed in the current environment.")
    if pod5 is None:
        raise ImportError("pod5 is not installed in the current environment.")
    if tqdm is None:
        raise ImportError("tqdm is not installed in the current environment.")

    target_reads = parse_target_txt(args.txt_path)
    if not target_reads:
        print("[WARN] No read IDs were found in the input txt.", file=sys.stderr)
        return

    stats = extract_filtered_signals(
        pod5_path=args.pod5_path,
        target_reads=target_reads,
        output_dir=args.output_dir,
        min_length=args.min_length,
        num=args.num,
        chunk_size=args.chunk_size,
    )

    print("\n===== Done =====")
    print(f"Target read IDs: {len(target_reads)}")
    print(f"Saved signals: {stats['saved_count']}")
    print(f"Output directory: {args.output_dir}")
    print(f"Output chunks: {stats['chunk_count']}")
    if stats["stopped_early"]:
        print(f"Stopped early after reaching num={args.num}")


if __name__ == "__main__":
    main()
