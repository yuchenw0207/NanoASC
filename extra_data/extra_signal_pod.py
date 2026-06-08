#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import sys
import traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    import h5py
except ImportError:
    h5py = None

try:
    import numpy as np
except ImportError:
    np = None

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

try:
    import pod5
except ImportError:
    pod5 = None


VALID_SPLITS = ("train", "valid", "test")
DEFAULT_FLUSH_SEGMENTS = 2048
DEFAULT_FLUSH_SIGNAL_POINTS = 5_000_000


def collect_pod5_paths(pod5_path_txt):
    """
    Read all pod5 file paths from a text file.

    Each non-empty line can be:
      1) a directory
      2) a single .pod5 file
    """
    pod5_files = []
    seen = set()

    with open(pod5_path_txt, "r", encoding="utf-8") as f:
        entries = [line.strip() for line in f if line.strip()]

    for entry in entries:
        if os.path.isfile(entry) and entry.endswith(".pod5"):
            ap = os.path.abspath(entry)
            if ap not in seen:
                pod5_files.append(ap)
                seen.add(ap)
            continue

        if os.path.isdir(entry):
            for root, _, files in os.walk(entry):
                for fn in files:
                    if fn.endswith(".pod5"):
                        ap = os.path.abspath(os.path.join(root, fn))
                        if ap not in seen:
                            pod5_files.append(ap)
                            seen.add(ap)
            continue

        print(f"[WARN] Skip invalid pod5 path entry: {entry}", file=sys.stderr)

    return sorted(pod5_files)


def resolve_regions_tsvs(tsv_paths):
    """
    Support either:
      - 1 TSV: infer split from file stem if possible, otherwise default to train
      - 3 TSVs: interpreted in order train valid test
    """
    if len(tsv_paths) == 1:
        path = os.path.abspath(tsv_paths[0])
        split = Path(path).stem.lower()
        if split not in VALID_SPLITS:
            split = "train"
        return {split: path}

    if len(tsv_paths) == 3:
        return {
            split: os.path.abspath(path)
            for split, path in zip(VALID_SPLITS, tsv_paths)
        }

    raise ValueError(
        "--regions_tsv must receive either 1 TSV or exactly 3 TSVs in order: "
        "train.tsv valid.tsv test.tsv"
    )


def load_signal_regions_tsvs(tsv_by_split):
    """
    Load region TSV(s).

    Required columns:
      parent_read_id
      parent_signal_start
      parent_signal_end
    """
    read_to_records = defaultdict(list)
    all_rows_meta = {}
    split_row_counts = {split: 0 for split in tsv_by_split}

    for split, tsv_path in tsv_by_split.items():
        with open(tsv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            if reader.fieldnames is None:
                raise ValueError(f"TSV has no header: {tsv_path}")

            required = {"parent_read_id", "parent_signal_start", "parent_signal_end"}
            missing = required - set(reader.fieldnames)
            if missing:
                raise ValueError(
                    f"TSV missing required columns: {', '.join(sorted(missing))}\n"
                    f"TSV: {tsv_path}\n"
                    f"Existing columns: {', '.join(reader.fieldnames)}"
                )

            for i, row in enumerate(reader, start=0):
                parent_read_id = (row.get("parent_read_id") or "").strip()
                if not parent_read_id:
                    continue

                try:
                    signal_start = int(row["parent_signal_start"])
                    signal_end = int(row["parent_signal_end"])
                except Exception:
                    print(
                        f"[WARN] Skip non-integer signal interval at {tsv_path}:{i + 2}",
                        file=sys.stderr,
                    )
                    continue

                if signal_end < signal_start:
                    signal_start, signal_end = signal_end, signal_start

                record_key = (split, i)
                rec = {
                    "record_key": record_key,
                    "tsv_row_idx": i,
                    "parent_read_id": parent_read_id,
                    "bed_name": (row.get("bed_name") or "").strip(),
                    "dataset_split": split,
                    "parent_signal_start": signal_start,
                    "parent_signal_end": signal_end,
                    "source_regions_tsv": tsv_path,
                }
                read_to_records[parent_read_id].append(rec)
                all_rows_meta[record_key] = rec
                split_row_counts[split] += 1

    return read_to_records, all_rows_meta, split_row_counts


def process_one_pod5(p5_path, read_to_records, min_start=1500, truncate_len=None):
    """
    Process one pod5 file once and return:
      hits: extracted signal segments
      skipped: skipped records / file-level failures
    """
    hits = []
    skipped = []

    try:
        with pod5.Reader(p5_path) as reader:
            for read in reader.reads():
                parent_read_id = str(read.read_id)
                if parent_read_id not in read_to_records:
                    continue

                try:
                    raw = np.asarray(read.signal, dtype=np.int16)
                except Exception as e:
                    for rec in read_to_records[parent_read_id]:
                        skipped.append({
                            "record_key": rec["record_key"],
                            "tsv_row_idx": rec["tsv_row_idx"],
                            "parent_read_id": parent_read_id,
                            "bed_name": rec["bed_name"],
                            "dataset_split": rec["dataset_split"],
                            "reason": f"read_signal_failed:{e}",
                            "source_pod5": p5_path,
                        })
                    continue

                raw_len = len(raw)

                for rec in read_to_records[parent_read_id]:
                    orig_start = rec["parent_signal_start"]
                    orig_end = rec["parent_signal_end"]

                    start = max(orig_start, min_start)
                    end = min(orig_end, raw_len)

                    if end <= start:
                        skipped.append({
                            "record_key": rec["record_key"],
                            "tsv_row_idx": rec["tsv_row_idx"],
                            "parent_read_id": parent_read_id,
                            "bed_name": rec["bed_name"],
                            "dataset_split": rec["dataset_split"],
                            "reason": (
                                "invalid_or_empty_interval_after_clip:"
                                f"start={start},end={end},raw_len={raw_len}"
                            ),
                            "source_pod5": p5_path,
                        })
                        continue

                    signal = raw[start:end]

                    if truncate_len is not None and len(signal) > truncate_len:
                        signal = signal[:truncate_len]
                        used_end = start + truncate_len
                    else:
                        used_end = end

                    signal = np.asarray(signal, dtype=np.int16)

                    hits.append({
                        "record_key": rec["record_key"],
                        "tsv_row_idx": rec["tsv_row_idx"],
                        "parent_read_id": parent_read_id,
                        "bed_name": rec["bed_name"],
                        "dataset_split": rec["dataset_split"],
                        "parent_signal_start": orig_start,
                        "parent_signal_end": orig_end,
                        "used_signal_start": start,
                        "used_signal_end": used_end,
                        "raw_length": raw_len,
                        "signal_length": int(len(signal)),
                        "source_pod5": p5_path,
                        "signal": signal,
                    })

    except Exception as e:
        skipped.append({
            "record_key": None,
            "tsv_row_idx": -1,
            "parent_read_id": "",
            "bed_name": "",
            "dataset_split": "",
            "reason": f"open_pod5_failed:{e}",
            "source_pod5": p5_path,
        })

    return hits, skipped


def init_h5(output_path, compression="gzip"):
    """
    Create HDF5 with concatenated signal storage.
    """
    h5f = h5py.File(output_path, "w")

    str_dt = h5py.string_dtype(encoding="utf-8")
    compression_kwargs = {"chunks": True}
    if compression != "none":
        compression_kwargs["compression"] = compression

    h5f.create_dataset(
        "signals",
        shape=(0,),
        maxshape=(None,),
        dtype=np.int16,
        **compression_kwargs,
    )

    meta = h5f.create_group("meta")
    meta.create_dataset("offset_start", shape=(0,), maxshape=(None,), dtype=np.int64, **compression_kwargs)
    meta.create_dataset("offset_end", shape=(0,), maxshape=(None,), dtype=np.int64, **compression_kwargs)
    meta.create_dataset("tsv_row_idx", shape=(0,), maxshape=(None,), dtype=np.int64, **compression_kwargs)

    meta.create_dataset("parent_signal_start", shape=(0,), maxshape=(None,), dtype=np.int64, **compression_kwargs)
    meta.create_dataset("parent_signal_end", shape=(0,), maxshape=(None,), dtype=np.int64, **compression_kwargs)
    meta.create_dataset("used_signal_start", shape=(0,), maxshape=(None,), dtype=np.int64, **compression_kwargs)
    meta.create_dataset("used_signal_end", shape=(0,), maxshape=(None,), dtype=np.int64, **compression_kwargs)
    meta.create_dataset("raw_length", shape=(0,), maxshape=(None,), dtype=np.int64, **compression_kwargs)
    meta.create_dataset("signal_length", shape=(0,), maxshape=(None,), dtype=np.int64, **compression_kwargs)

    meta.create_dataset("parent_read_id", shape=(0,), maxshape=(None,), dtype=str_dt, chunks=True)
    meta.create_dataset("bed_name", shape=(0,), maxshape=(None,), dtype=str_dt, chunks=True)
    meta.create_dataset("dataset_split", shape=(0,), maxshape=(None,), dtype=str_dt, chunks=True)
    meta.create_dataset("source_pod5", shape=(0,), maxshape=(None,), dtype=str_dt, chunks=True)

    return h5f


def append_batch_to_h5(h5f, batch):
    """
    Append one batch of extracted signals into HDF5.
    """
    if not batch:
        return 0

    sig_ds = h5f["signals"]
    meta = h5f["meta"]

    n_seg_old = meta["offset_start"].shape[0]
    n_seg_add = len(batch)

    total_signal_add = sum(item["signal_length"] for item in batch)
    sig_old = sig_ds.shape[0]

    sig_ds.resize((sig_old + total_signal_add,))

    concat_signal = np.concatenate([item["signal"] for item in batch], axis=0)
    sig_ds[sig_old:sig_old + total_signal_add] = concat_signal

    offset_start = np.empty(n_seg_add, dtype=np.int64)
    offset_end = np.empty(n_seg_add, dtype=np.int64)

    cur = sig_old
    for i, item in enumerate(batch):
        offset_start[i] = cur
        cur += item["signal_length"]
        offset_end[i] = cur

    for key in meta.keys():
        ds = meta[key]
        ds.resize((n_seg_old + n_seg_add,))

    meta["offset_start"][n_seg_old:n_seg_old + n_seg_add] = offset_start
    meta["offset_end"][n_seg_old:n_seg_old + n_seg_add] = offset_end
    meta["tsv_row_idx"][n_seg_old:n_seg_old + n_seg_add] = [x["tsv_row_idx"] for x in batch]

    meta["parent_signal_start"][n_seg_old:n_seg_old + n_seg_add] = [x["parent_signal_start"] for x in batch]
    meta["parent_signal_end"][n_seg_old:n_seg_old + n_seg_add] = [x["parent_signal_end"] for x in batch]
    meta["used_signal_start"][n_seg_old:n_seg_old + n_seg_add] = [x["used_signal_start"] for x in batch]
    meta["used_signal_end"][n_seg_old:n_seg_old + n_seg_add] = [x["used_signal_end"] for x in batch]
    meta["raw_length"][n_seg_old:n_seg_old + n_seg_add] = [x["raw_length"] for x in batch]
    meta["signal_length"][n_seg_old:n_seg_old + n_seg_add] = [x["signal_length"] for x in batch]

    meta["parent_read_id"][n_seg_old:n_seg_old + n_seg_add] = [x["parent_read_id"] for x in batch]
    meta["bed_name"][n_seg_old:n_seg_old + n_seg_add] = [x["bed_name"] for x in batch]
    meta["dataset_split"][n_seg_old:n_seg_old + n_seg_add] = [x["dataset_split"] for x in batch]
    meta["source_pod5"][n_seg_old:n_seg_old + n_seg_add] = [x["source_pod5"] for x in batch]

    return n_seg_add


def flush_split_buffer(split, h5_handles, pending_hits_by_split, pending_signal_points_by_split):
    """
    Flush one split's in-memory buffer into its HDF5 file.
    """
    batch = pending_hits_by_split[split]
    if not batch:
        return 0

    n_add = append_batch_to_h5(h5_handles[split], batch)
    pending_hits_by_split[split] = []
    pending_signal_points_by_split[split] = 0
    return n_add


def flush_all_split_buffers(h5_handles, pending_hits_by_split, pending_signal_points_by_split, total_written_by_split):
    """
    Flush all pending split buffers.
    """
    for split in list(pending_hits_by_split.keys()):
        n_add = flush_split_buffer(
            split,
            h5_handles,
            pending_hits_by_split,
            pending_signal_points_by_split,
        )
        total_written_by_split[split] += n_add


def write_report_tsv(path, rows, fieldnames):
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Extract signal segments from pod5 using one or three TSV region files. "
            "When three TSVs are provided, they are interpreted in order: "
            "train.tsv valid.tsv test.tsv. The script traverses pod5 only once "
            "and writes split-specific HDF5 files into the output directory."
        )
    )
    parser.add_argument(
        "--pod5_path_txt",
        dest="pod5_path_txt",
        required=True,
        help="Text file containing pod5 directories or pod5 file paths, one per line.",
    )
    parser.add_argument(
        "--regions_tsv",
        nargs="+",
        required=True,
        help=(
            "One TSV, or exactly three TSVs in order: train.tsv valid.tsv test.tsv. "
            "Each TSV must contain parent_read_id, parent_signal_start, parent_signal_end."
        ),
    )
    parser.add_argument(
        "--output",
        "--output-dir",
        dest="output",
        required=True,
        help="Output directory. Will create train.h5 / valid.h5 / test.h5 for provided splits.",
    )
    parser.add_argument("--threads", type=int, default=4, help="Number of worker threads. Default: 4")
    parser.add_argument("--min-start", type=int, default=1500, help="Lower bound for signal start. Default: 1500")
    parser.add_argument("--truncate-len", type=int, default=None, help="Optional max signal length after clipping.")
    parser.add_argument(
        "--flush-segments",
        type=int,
        default=DEFAULT_FLUSH_SEGMENTS,
        help=(
            "Buffered write threshold by segment count per split. "
            f"Default: {DEFAULT_FLUSH_SEGMENTS}"
        ),
    )
    parser.add_argument(
        "--flush-signal-points",
        type=int,
        default=DEFAULT_FLUSH_SIGNAL_POINTS,
        help=(
            "Buffered write threshold by total signal points per split. "
            f"Default: {DEFAULT_FLUSH_SIGNAL_POINTS}"
        ),
    )
    parser.add_argument(
        "--compression",
        choices=("gzip", "lzf", "none"),
        default="gzip",
        help="HDF5 compression mode for numeric datasets. Default: gzip",
    )
    parser.add_argument(
        "--scale",
        action="store_true",
        help="Kept for CLI compatibility. Ignored for pod5 input; raw read.signal is used.",
    )
    args = parser.parse_args()

    if h5py is None:
        raise ImportError(
            "h5py is not installed in the current environment. "
            "Please install it before running this script."
        )

    if np is None:
        raise ImportError(
            "numpy is not installed in the current environment. "
            "Please install it before running this script."
        )

    if tqdm is None:
        raise ImportError(
            "tqdm is not installed in the current environment. "
            "Please install it before running this script."
        )

    if pod5 is None:
        raise ImportError(
            "pod5 is not installed in the current environment. "
            "Please install it before running this script."
        )

    if args.scale:
        print(
            "[WARN] --scale is ignored for pod5 input. The script uses raw read.signal values.",
            file=sys.stderr,
        )

    if args.flush_segments <= 0:
        raise ValueError("--flush-segments must be > 0")

    if args.flush_signal_points <= 0:
        raise ValueError("--flush-signal-points must be > 0")

    tsv_by_split = resolve_regions_tsvs(args.regions_tsv)
    read_to_records, all_rows_meta, split_row_counts = load_signal_regions_tsvs(tsv_by_split)
    pod5_files = collect_pod5_paths(args.pod5_path_txt)

    output_dir = Path(args.output).resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise RuntimeError(f"Output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found pod5 files: {len(pod5_files)}")
    print(f"Target parent read ids: {len(read_to_records)}")
    print(f"Target records: {len(all_rows_meta)}")
    print(
        "Write buffering: "
        f"flush_segments={args.flush_segments}, "
        f"flush_signal_points={args.flush_signal_points}, "
        f"compression={args.compression}"
    )
    for split in tsv_by_split:
        print(f"  {split}: {split_row_counts.get(split, 0)} rows from {tsv_by_split[split]}")

    if len(pod5_files) == 0:
        raise RuntimeError("No pod5 files were found.")

    if len(all_rows_meta) == 0:
        raise RuntimeError("No valid records were loaded from --regions_tsv.")

    h5_handles = {}
    for split in tsv_by_split:
        h5_handles[split] = init_h5(
            str(output_dir / f"{split}.h5"),
            compression=args.compression,
        )

    skipped_rows_by_split = {split: [] for split in tsv_by_split}
    duplicate_rows_by_split = {split: [] for split in tsv_by_split}
    total_written_by_split = {split: 0 for split in tsv_by_split}
    pending_hits_by_split = {split: [] for split in tsv_by_split}
    pending_signal_points_by_split = {split: 0 for split in tsv_by_split}
    global_skipped_rows = []

    written_keys = set()
    seen_any_keys = set()

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.threads)) as ex:
            futures = {
                ex.submit(
                    process_one_pod5,
                    p5_path,
                    read_to_records,
                    args.min_start,
                    args.truncate_len,
                ): p5_path
                for p5_path in pod5_files
            }

            for fut in tqdm(as_completed(futures), total=len(futures), desc="Processing pod5"):
                p5_path = futures[fut]
                try:
                    hits, skipped = fut.result()
                except Exception:
                    print(f"[ERROR] Failed to process: {p5_path}", file=sys.stderr)
                    traceback.print_exc()
                    continue

                for row in skipped:
                    record_key = row.get("record_key")
                    split = row.get("dataset_split", "")
                    clean_row = {
                        "tsv_row_idx": row["tsv_row_idx"],
                        "parent_read_id": row["parent_read_id"],
                        "bed_name": row["bed_name"],
                        "reason": row["reason"],
                        "source_pod5": row["source_pod5"],
                    }
                    if record_key is None or split not in skipped_rows_by_split:
                        global_skipped_rows.append(clean_row)
                        continue

                    seen_any_keys.add(record_key)
                    skipped_rows_by_split[split].append(clean_row)

                hits_by_split = defaultdict(list)
                for item in hits:
                    record_key = item["record_key"]
                    split = item["dataset_split"]
                    seen_any_keys.add(record_key)

                    if record_key in written_keys:
                        duplicate_rows_by_split[split].append({
                            "tsv_row_idx": item["tsv_row_idx"],
                            "parent_read_id": item["parent_read_id"],
                            "bed_name": item["bed_name"],
                            "source_pod5": item["source_pod5"],
                            "reason": "duplicate_hit_skip",
                        })
                        continue

                    written_keys.add(record_key)
                    hits_by_split[split].append(item)

                for split, batch in hits_by_split.items():
                    pending_hits_by_split[split].extend(batch)
                    pending_signal_points_by_split[split] += sum(
                        item["signal_length"] for item in batch
                    )

                    if (
                        len(pending_hits_by_split[split]) >= args.flush_segments
                        or pending_signal_points_by_split[split] >= args.flush_signal_points
                    ):
                        total_written_by_split[split] += flush_split_buffer(
                            split,
                            h5_handles,
                            pending_hits_by_split,
                            pending_signal_points_by_split,
                        )

        flush_all_split_buffers(
            h5_handles,
            pending_hits_by_split,
            pending_signal_points_by_split,
            total_written_by_split,
        )

        for h5f in h5_handles.values():
            h5f.flush()

    finally:
        for h5f in h5_handles.values():
            try:
                h5f.close()
            except Exception:
                pass

    missing_rows_by_split = {split: [] for split in tsv_by_split}
    for record_key, rec in all_rows_meta.items():
        split = rec["dataset_split"]
        if record_key not in seen_any_keys and record_key not in written_keys:
            missing_rows_by_split[split].append({
                "tsv_row_idx": rec["tsv_row_idx"],
                "parent_read_id": rec["parent_read_id"],
                "bed_name": rec["bed_name"],
                "reason": "read_not_found_in_any_pod5",
            })

    for split in tsv_by_split:
        write_report_tsv(
            str(output_dir / f"{split}.skipped.tsv"),
            skipped_rows_by_split[split],
            ["tsv_row_idx", "parent_read_id", "bed_name", "reason", "source_pod5"],
        )
        write_report_tsv(
            str(output_dir / f"{split}.missing.tsv"),
            missing_rows_by_split[split],
            ["tsv_row_idx", "parent_read_id", "bed_name", "reason"],
        )
        write_report_tsv(
            str(output_dir / f"{split}.duplicate.tsv"),
            duplicate_rows_by_split[split],
            ["tsv_row_idx", "parent_read_id", "bed_name", "source_pod5", "reason"],
        )

    if global_skipped_rows:
        write_report_tsv(
            str(output_dir / "global.skipped.tsv"),
            global_skipped_rows,
            ["tsv_row_idx", "parent_read_id", "bed_name", "reason", "source_pod5"],
        )

    print("\nExtraction finished")
    for split in tsv_by_split:
        print(
            f"{split}: written={total_written_by_split[split]}, "
            f"skipped={len(skipped_rows_by_split[split])}, "
            f"missing={len(missing_rows_by_split[split])}, "
            f"duplicate={len(duplicate_rows_by_split[split])}, "
            f"output={output_dir / f'{split}.h5'}"
        )
    if global_skipped_rows:
        print(f"global skipped report: {output_dir / 'global.skipped.tsv'}")


if __name__ == "__main__":
    main()