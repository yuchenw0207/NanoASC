#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import hashlib
import os
import random
import re
import sys
from collections import defaultdict
from typing import Dict, List, Set, Tuple


N_SPLITS = 4
N_BUCKETS = 4  # 4 个 bucket 中 2 个 train、1 个 valid、1 个 test，即 2:1:1


def sanitize_name(name: str) -> str:
    """把 bed_name 转成适合文件名的形式。"""
    name = name.strip()
    name = re.sub(r"[\\/:*?\"<>|]", "_", name)
    name = re.sub(r"\s+", "_", name)
    return name if name else "EMPTY_BED_NAME"


def stable_gene_seed(base_seed: int, gene: str) -> int:
    """
    基于全局 seed 和 gene 名称生成稳定的 gene-level seed。

    这样做的好处是：
    1. 所有基因仍由同一个 base_seed 控制；
    2. 每个基因内部的 shuffle 互不影响；
    3. 新增或删除某个基因时，不会改变其他基因的划分结果。
    """
    text = f"{base_seed}::{gene}".encode("utf-8")
    digest = hashlib.md5(text).hexdigest()
    return int(digest[:8], 16)


def read_tsv(path: str) -> Tuple[List[Dict[str, str]], List[str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("输入 TSV 缺少表头")
        rows = list(reader)
        return rows, reader.fieldnames


def write_tsv(path: str, fieldnames: List[str], rows: List[Dict[str, str]]) -> None:
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_is_split(value: str) -> bool:
    """
    兼容 is_split_read / is_split 列中的常见写法：
      1, true, True, yes, Y
    """
    if value is None:
        return False
    v = str(value).strip().lower()
    return v in {"1", "true", "yes", "y", "t"}


def get_logical_read_id(row: Dict[str, str], has_parent_read_id: bool,
                        has_old_read_id: bool, has_subread_id: bool) -> str:
    """parent_read_id 优先，其次 read_id，最后 subread_id。"""
    if has_parent_read_id:
        return (row.get("parent_read_id") or "").strip()
    if has_old_read_id:
        return (row.get("read_id") or "").strip()
    if has_subread_id:
        return (row.get("subread_id") or "").strip()
    return ""


def assign_reads_to_buckets(read_ids: List[str], seed: int, n_buckets: int = N_BUCKETS) -> Dict[str, int]:
    """
    将某个基因内部的 free reads 分配到 0..n_buckets-1。

    默认 n_buckets=4。每个 split 中：
      test 使用 1 个 bucket；
      valid 使用 1 个 bucket；
      train 使用剩余 2 个 bucket。
    因此每个 split 内部近似为 2:1:1。

    注意：如果某个基因 free reads 数量很少，部分 bucket 可能为空。
    """
    shuffled = sorted(read_ids)
    rng = random.Random(seed)
    rng.shuffle(shuffled)

    bucket_map: Dict[str, int] = {}
    for i, rid in enumerate(shuffled):
        bucket_map[rid] = i % n_buckets
    return bucket_map


def add_rows_for_reads(
    output_rows: List[Dict[str, str]],
    read_ids: List[str],
    read2rows: Dict[str, List[Dict[str, str]]],
    out_fieldnames: List[str],
    split_name: str,
    split_id: int,
) -> int:
    """按 read_id 将原始行写入对应 split，并补充 dataset_split/cv_fold。返回写入行数。"""
    n_written = 0
    for rid in read_ids:
        for row in read2rows[rid]:
            new_row = dict(row)
            new_row["dataset_split"] = split_name
            new_row["cv_fold"] = str(split_id)
            for field in out_fieldnames:
                if field not in new_row:
                    new_row[field] = ""
            output_rows.append(new_row)
            n_written += 1
    return n_written


def main():
    ap = argparse.ArgumentParser(
        description=(
            "按 bed_name 分组，并按 parent_read_id/read_id/subread_id 在每个基因内部做 4 组 2:1:1 划分；"
            "跨多个基因的原始 read 强制进入每一组 train；"
            "is_split_read=1 的原始 read 也强制进入每一组 train。"
        )
    )
    ap.add_argument("input_tsv", help="输入 TSV，建议使用 signal_position.tsv")
    ap.add_argument("-o", "--out-prefix", default="split_by_gene_4fold_211", help="输出目录，默认 split_by_gene_4fold_211")
    ap.add_argument("--seed", type=int, default=42, help="划分随机种子，默认 42")
    ap.add_argument("--n-splits", type=int, default=N_SPLITS, help="输出 split 数，默认 4")
    ap.add_argument("--n-buckets", type=int, default=N_BUCKETS, help="内部 bucket 数；默认 4，对应 2:1:1")
    ap.add_argument(
        "--write-per-gene",
        action="store_true",
        help="额外输出每个基因在每个 split 下的 train/valid/test 文件"
    )
    args = ap.parse_args()

    if args.n_buckets < 3:
        raise ValueError("--n-buckets 必须 >= 3。2:1:1 推荐使用 --n-buckets 4")
    if args.n_splits < 1:
        raise ValueError("--n-splits 必须 >= 1")
    if args.n_splits > args.n_buckets:
        raise ValueError("--n-splits 不应大于 --n-buckets；2:1:1 默认 n_splits=4, n_buckets=4")

    train_bucket_count = args.n_buckets - 2
    ratio_note = f"ratio_{train_bucket_count}_1_1_by_{args.n_buckets}_buckets"

    rows, fieldnames = read_tsv(args.input_tsv)

    # 兼容新旧格式
    has_parent_read_id = "parent_read_id" in fieldnames
    has_subread_id = "subread_id" in fieldnames
    has_old_read_id = "read_id" in fieldnames
    has_is_split_read = "is_split_read" in fieldnames
    has_is_split = "is_split" in fieldnames

    required = {"bed_name"}
    missing = required - set(fieldnames)
    if missing:
        raise ValueError(f"输入 TSV 缺少必要列: {', '.join(sorted(missing))}")

    if not (has_parent_read_id or has_old_read_id or has_subread_id):
        raise ValueError("输入 TSV 至少需要包含以下列之一：parent_read_id / read_id / subread_id")

    # gene -> logical_read_id -> [rows...]
    gene_read_rows: Dict[str, Dict[str, List[Dict[str, str]]]] = defaultdict(lambda: defaultdict(list))
    # logical_read_id -> set(genes)
    read_to_genes: Dict[str, Set[str]] = defaultdict(set)
    # logical_read_id -> whether split
    read_is_split: Dict[str, bool] = defaultdict(bool)

    n_empty_read_id = 0

    for row in rows:
        gene = (row.get("bed_name") or "").strip()
        if not gene:
            gene = "EMPTY_BED_NAME"

        logical_read_id = get_logical_read_id(row, has_parent_read_id, has_old_read_id, has_subread_id)
        if not logical_read_id:
            n_empty_read_id += 1
            print("[WARN] 发现逻辑 read_id 为空的行，已跳过", file=sys.stderr)
            continue

        is_split_flag = False
        if has_is_split_read:
            is_split_flag = parse_is_split(row.get("is_split_read", "0"))
        elif has_is_split:
            is_split_flag = parse_is_split(row.get("is_split", "0"))

        gene_read_rows[gene][logical_read_id].append(row)
        read_to_genes[logical_read_id].add(gene)
        if is_split_flag:
            read_is_split[logical_read_id] = True

    # 跨多个基因的原始 read，以及任一行带 split 标记的原始 read，都强制进入 train
    multi_gene_reads = {rid for rid, genes in read_to_genes.items() if len(genes) > 1}
    split_reads = {rid for rid, flag in read_is_split.items() if flag}
    all_forced_train_reads = sorted(multi_gene_reads | split_reads)

    out_fieldnames = fieldnames[:]
    if "dataset_split" not in out_fieldnames:
        out_fieldnames.append("dataset_split")
    if "cv_fold" not in out_fieldnames:
        out_fieldnames.append("cv_fold")

    os.makedirs(args.out_prefix, exist_ok=True)

    forced_train_report_rows: List[Dict[str, str]] = []
    for rid in all_forced_train_reads:
        reasons = []
        if rid in multi_gene_reads:
            reasons.append("multi_gene")
        if rid in split_reads:
            reasons.append("split_read")
        forced_train_report_rows.append({
            "parent_read_id": rid,
            "n_genes": str(len(read_to_genes[rid])),
            "genes": ",".join(sorted(read_to_genes[rid])),
            "is_split_read": "1" if rid in split_reads else "0",
            "force_reason": ",".join(reasons),
        })

    # 全局报告：每个 gene 内部每条 free read 被分到哪个 bucket
    read_bucket_assignment_rows: List[Dict[str, str]] = []

    # 每个 split 分别积累 train/valid/test
    split_outputs = []
    for split_id in range(args.n_splits):
        split_outputs.append({
            "train": [],
            "valid": [],
            "test": [],
            "summary": [],
            "skipped": [],
        })

    total_genes = 0
    genes_with_no_free_reads = 0

    for gene in sorted(gene_read_rows.keys()):
        total_genes += 1
        read2rows = gene_read_rows[gene]
        unique_reads = sorted(read2rows.keys())
        n_reads = len(unique_reads)
        n_rows = sum(len(v) for v in read2rows.values())

        forced_train_reads = [
            rid for rid in unique_reads
            if (rid in multi_gene_reads) or (rid in split_reads)
        ]
        free_reads = [rid for rid in unique_reads if rid not in forced_train_reads]

        forced_n = len(forced_train_reads)
        free_n = len(free_reads)

        if free_n == 0:
            genes_with_no_free_reads += 1

        gene_seed = stable_gene_seed(args.seed, gene)
        bucket_map = assign_reads_to_buckets(free_reads, seed=gene_seed, n_buckets=args.n_buckets)

        for rid in free_reads:
            read_bucket_assignment_rows.append({
                "bed_name": gene,
                "parent_read_id": rid,
                "assigned_bucket": str(bucket_map[rid]),
                "is_forced_train": "0",
                "force_reason": "",
            })

        for rid in forced_train_reads:
            reasons = []
            if rid in multi_gene_reads:
                reasons.append("multi_gene")
            if rid in split_reads:
                reasons.append("split_read")
            read_bucket_assignment_rows.append({
                "bed_name": gene,
                "parent_read_id": rid,
                "assigned_bucket": "forced_train",
                "is_forced_train": "1",
                "force_reason": ",".join(reasons),
            })

        for split_id in range(args.n_splits):
            # 默认 n_splits=4, n_buckets=4：
            # split_0: test_bucket=0, valid_bucket=1, train=其余2个
            # split_1: test_bucket=1, valid_bucket=2, train=其余2个
            # split_2: test_bucket=2, valid_bucket=3, train=其余2个
            # split_3: test_bucket=3, valid_bucket=0, train=其余2个
            test_bucket = split_id % args.n_buckets
            valid_bucket = (split_id + 1) % args.n_buckets

            if valid_bucket == test_bucket:
                raise RuntimeError("valid_bucket 与 test_bucket 重叠，请检查 --n-splits 和 --n-buckets 设置")

            test_reads = [rid for rid in free_reads if bucket_map.get(rid) == test_bucket]
            valid_reads = [rid for rid in free_reads if bucket_map.get(rid) == valid_bucket]
            train_reads = [
                rid for rid in free_reads
                if bucket_map.get(rid) not in {test_bucket, valid_bucket}
            ]
            train_reads = forced_train_reads + train_reads

            n_train_rows = add_rows_for_reads(
                split_outputs[split_id]["train"],
                train_reads,
                read2rows,
                out_fieldnames,
                "train",
                split_id,
            )
            n_valid_rows = add_rows_for_reads(
                split_outputs[split_id]["valid"],
                valid_reads,
                read2rows,
                out_fieldnames,
                "valid",
                split_id,
            )
            n_test_rows = add_rows_for_reads(
                split_outputs[split_id]["test"],
                test_reads,
                read2rows,
                out_fieldnames,
                "test",
                split_id,
            )

            status_parts = ["ok"]
            if forced_n > 0:
                status_parts.append("with_forced_train_reads")
            if free_n < args.n_buckets:
                status_parts.append("free_reads_less_than_n_buckets")
            if len(test_reads) == 0:
                status_parts.append("empty_test_for_this_gene")
            if len(valid_reads) == 0:
                status_parts.append("empty_valid_for_this_gene")

            split_outputs[split_id]["summary"].append({
                "cv_fold": str(split_id),
                "bed_name": gene,
                "n_reads": str(n_reads),
                "n_rows": str(n_rows),
                "forced_train_reads": str(forced_n),
                "free_reads": str(free_n),
                "train_reads": str(len(train_reads)),
                "valid_reads": str(len(valid_reads)),
                "test_reads": str(len(test_reads)),
                "train_rows": str(n_train_rows),
                "valid_rows": str(n_valid_rows),
                "test_rows": str(n_test_rows),
                "test_bucket": str(test_bucket),
                "valid_bucket": str(valid_bucket),
                "status": ";".join(status_parts),
                "note": ratio_note,
            })

            if free_n == 0:
                split_outputs[split_id]["skipped"].append({
                    "cv_fold": str(split_id),
                    "bed_name": gene,
                    "n_reads": str(n_reads),
                    "n_rows": str(n_rows),
                    "reason": "no_free_reads_all_forced_train_or_empty",
                })

            if args.write_per_gene:
                safe_gene = sanitize_name(gene)
                gene_dir = os.path.join(args.out_prefix, f"fold_{split_id}", "per_gene", safe_gene)
                os.makedirs(gene_dir, exist_ok=True)

                gene_train_rows: List[Dict[str, str]] = []
                gene_valid_rows: List[Dict[str, str]] = []
                gene_test_rows: List[Dict[str, str]] = []

                add_rows_for_reads(gene_train_rows, train_reads, read2rows, out_fieldnames, "train", split_id)
                add_rows_for_reads(gene_valid_rows, valid_reads, read2rows, out_fieldnames, "valid", split_id)
                add_rows_for_reads(gene_test_rows, test_reads, read2rows, out_fieldnames, "test", split_id)

                write_tsv(os.path.join(gene_dir, "train.tsv"), out_fieldnames, gene_train_rows)
                write_tsv(os.path.join(gene_dir, "valid.tsv"), out_fieldnames, gene_valid_rows)
                write_tsv(os.path.join(gene_dir, "test.tsv"), out_fieldnames, gene_test_rows)

    summary_fieldnames = [
        "cv_fold", "bed_name", "n_reads", "n_rows",
        "forced_train_reads", "free_reads",
        "train_reads", "valid_reads", "test_reads",
        "train_rows", "valid_rows", "test_rows",
        "test_bucket", "valid_bucket", "status", "note",
    ]

    skipped_fieldnames = ["cv_fold", "bed_name", "n_reads", "n_rows", "reason"]

    total_rows_report: List[Dict[str, str]] = []
    for split_id in range(args.n_splits):
        split_dir = os.path.join(args.out_prefix, f"fold_{split_id}")
        os.makedirs(split_dir, exist_ok=True)

        train_rows = split_outputs[split_id]["train"]
        valid_rows = split_outputs[split_id]["valid"]
        test_rows = split_outputs[split_id]["test"]

        write_tsv(os.path.join(split_dir, "train.tsv"), out_fieldnames, train_rows)
        write_tsv(os.path.join(split_dir, "valid.tsv"), out_fieldnames, valid_rows)
        write_tsv(os.path.join(split_dir, "test.tsv"), out_fieldnames, test_rows)
        write_tsv(os.path.join(split_dir, "summary.tsv"), summary_fieldnames, split_outputs[split_id]["summary"])
        write_tsv(os.path.join(split_dir, "skipped.tsv"), skipped_fieldnames, split_outputs[split_id]["skipped"])

        total_rows_report.append({
            "cv_fold": str(split_id),
            "train_rows": str(len(train_rows)),
            "valid_rows": str(len(valid_rows)),
            "test_rows": str(len(test_rows)),
        })

        print(
            f"[DONE] fold_{split_id}: train_rows={len(train_rows)}, "
            f"valid_rows={len(valid_rows)}, test_rows={len(test_rows)}",
            file=sys.stderr,
        )

    write_tsv(
        os.path.join(args.out_prefix, "forced_train_reads.tsv"),
        ["parent_read_id", "n_genes", "genes", "is_split_read", "force_reason"],
        forced_train_report_rows,
    )

    write_tsv(
        os.path.join(args.out_prefix, "read_bucket_assignment.tsv"),
        ["bed_name", "parent_read_id", "assigned_bucket", "is_forced_train", "force_reason"],
        read_bucket_assignment_rows,
    )

    write_tsv(
        os.path.join(args.out_prefix, "fold_row_counts.tsv"),
        ["cv_fold", "train_rows", "valid_rows", "test_rows"],
        total_rows_report,
    )

    print(f"[DONE] total_genes={total_genes}", file=sys.stderr)
    print(f"[DONE] empty_logical_read_id_rows_skipped={n_empty_read_id}", file=sys.stderr)
    print(f"[DONE] genes_with_no_free_reads={genes_with_no_free_reads}", file=sys.stderr)
    print(f"[DONE] multi_gene_parent_reads={len(multi_gene_reads)}", file=sys.stderr)
    print(f"[DONE] split_parent_reads={len(split_reads)}", file=sys.stderr)
    print(f"[DONE] forced_train_parent_reads={len(all_forced_train_reads)}", file=sys.stderr)
    print(f"[DONE] output_dir={args.out_prefix}", file=sys.stderr)
    print(f"[DONE] split_ratio={ratio_note}", file=sys.stderr)


if __name__ == "__main__":
    main()
