#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import re
import sys
from collections import defaultdict
from typing import Dict, Tuple, List, Optional, Any

import numpy as np
import pysam


def clean_read_id(read_id: str) -> str:
    """
    轻度清洗 read id，避免 /1 /2 或其他后缀导致匹配失败。
    如果你确定 TSV 和 BAM 完全一致，可用 --no-clean-id 关闭。
    """
    cleaned = read_id.strip()
    cleaned = re.sub(r'\/[12]$', '', cleaned)
    cleaned = re.sub(r'[#@]\S+$', '', cleaned)
    return cleaned


def load_regions_from_tsv(
    path: str,
    read_id_col: str = "subread_id",
    start_col: str = "read_start0",
    end_col: str = "read_end0",
    bed_name_col: str = "bed_name",
    parent_read_id_col: str = "parent_read_id",
    split_signal_offset_col: str = "split_signal_offset",
) -> Dict[str, List[Dict[str, Any]]]:
    """
    读取 TSV：
      必须包含 subread_id, read_start0, read_end0, bed_name
      如果存在 parent_read_id / split_signal_offset 也会一并读取

    返回：
      cleaned_subread_id -> [ record1, record2, ... ]
    """
    data: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("输入 TSV 缺少表头。")

        required = {read_id_col, start_col, end_col, bed_name_col}
        missing = required - set(reader.fieldnames)
        if missing:
            raise ValueError(
                f"输入 TSV 缺少必要列: {', '.join(sorted(missing))}\n"
                f"现有列: {', '.join(reader.fieldnames)}"
            )

        has_parent_col = parent_read_id_col in reader.fieldnames
        has_sp_col = split_signal_offset_col in reader.fieldnames

        for ln, row in enumerate(reader, start=2):
            rid = (row.get(read_id_col) or "").strip()
            if not rid:
                print(f"[WARN] 第{ln}行 {read_id_col} 为空，跳过", file=sys.stderr)
                continue

            try:
                s = int(row[start_col])
                e = int(row[end_col])
            except Exception:
                print(
                    f"[WARN] 第{ln}行 {start_col}/{end_col} 不是整数，跳过: "
                    f"{row.get(start_col)} {row.get(end_col)}",
                    file=sys.stderr
                )
                continue

            if e < s:
                s, e = e, s

            bed_name = (row.get(bed_name_col) or "").strip()

            parent_read_id = (row.get(parent_read_id_col) or rid).strip() if has_parent_col else rid

            if has_sp_col:
                sp_raw = (row.get(split_signal_offset_col) or "").strip()
                try:
                    split_signal_offset = int(sp_raw) if sp_raw != "" else 0
                except Exception:
                    split_signal_offset = 0
            else:
                split_signal_offset = 0

            cid = clean_read_id(rid)
            data[cid].append({
                "original_subread_id": rid,
                "parent_read_id": parent_read_id,
                "split_signal_offset": split_signal_offset,
                "start0": s,   # 0-based half-open
                "end0": e,     # 0-based half-open
                "bed_name": bed_name,
                "row": row
            })

    return data


def parse_mv_to_base_intervals_with_ts(
    mv: List[int],
    num_bases_expected: int,
    ts: int,
) -> Optional[Tuple[int, List[Tuple[int, int]], List[Tuple[int, int]]]]:
    """
    按 mv + ts 解析每个碱基对应的信号区间。

    返回：
      stride,
      intervals_trimmed,  # trimmed signal 坐标
      intervals_raw       # 当前 BAM 记录自身 raw signal 坐标（已加 ts）
    """
    if mv is None or len(mv) < 2:
        return None

    stride = int(mv[0])
    moves = np.asarray(mv[1:], dtype=np.int32)

    one_idx = np.flatnonzero(moves == 1)
    if len(one_idx) == 0:
        return None

    def build_intervals_from_starts(starts: np.ndarray) -> List[Tuple[int, int]]:
        if len(starts) == 0:
            return []

        bounds = np.append(starts, len(moves))
        intervals: List[Tuple[int, int]] = []
        for i in range(len(bounds) - 1):
            start = int(bounds[i]) * stride
            end = int(bounds[i + 1]) * stride
            intervals.append((start, end))
        return intervals

    candidates: List[List[Tuple[int, int]]] = []
    candidates.append(build_intervals_from_starts(one_idx))

    if len(one_idx) > 1:
        candidates.append(build_intervals_from_starts(one_idx[1:]))
        candidates.append(build_intervals_from_starts(one_idx[:-1]))

    best_trimmed = None
    for cand in candidates:
        if len(cand) == num_bases_expected:
            best_trimmed = cand
            break

    if best_trimmed is None:
        return None

    intervals_raw = [(ts + s, ts + e) for s, e in best_trimmed]
    return stride, best_trimmed, intervals_raw


def get_parent_info_from_bam(read):
    """
    从 BAM 标签中再次读取 pi/sp，作为兜底或校验。
    正常 read 没有 pi/sp 时：
      parent_read_id = query_name
      split_signal_offset = 0
      is_split_read = 0
    """
    subread_id = read.query_name

    if read.has_tag("pi"):
        parent_read_id = read.get_tag("pi")
        is_split_read = 1
    else:
        parent_read_id = subread_id
        is_split_read = 0

    if read.has_tag("sp"):
        split_signal_offset = int(read.get_tag("sp"))
    else:
        split_signal_offset = 0

    return parent_read_id, split_signal_offset, is_split_read


def main():
    ap = argparse.ArgumentParser(
        description="根据 TSV 中的 subread 碱基区间和 BAM 的 mv + ts + pi/sp tag，提取原始 parent read 的信号区间。"
    )
    ap.add_argument("regions_tsv", help="输入 TSV，建议使用脚本1输出结果")
    ap.add_argument("bam", help="输入 BAM，需包含 mv / ts；split-read 时可包含 pi / sp")
    ap.add_argument("-o", "--output", default="signal_regions.tsv", help="输出 TSV 路径")

    ap.add_argument("--read-id-col", default="subread_id", help="TSV 中 subread id 列名，默认 subread_id")
    ap.add_argument("--start-col", default="read_start0", help="TSV 中起始列名，默认 read_start0")
    ap.add_argument("--end-col", default="read_end0", help="TSV 中终止列名，默认 read_end0")
    ap.add_argument("--bed-name-col", default="bed_name", help="TSV 中 bed_name 列名，默认 bed_name")
    ap.add_argument("--parent-read-id-col", default="parent_read_id", help="TSV 中 parent_read_id 列名")
    ap.add_argument("--split-signal-offset-col", default="split_signal_offset", help="TSV 中 split_signal_offset 列名")

    ap.add_argument("--no-clean-id", action="store_true", help="不清洗 read_id，严格匹配")
    ap.add_argument("--write-not-found", action="store_true", help="输出未在 BAM 中找到的 read_id")
    args = ap.parse_args()

    global clean_read_id
    if args.no_clean_id:
        clean_read_id = lambda x: x.strip()  # type: ignore

    try:
        id2records = load_regions_from_tsv(
            args.regions_tsv,
            read_id_col=args.read_id_col,
            start_col=args.start_col,
            end_col=args.end_col,
            bed_name_col=args.bed_name_col,
            parent_read_id_col=args.parent_read_id_col,
            split_signal_offset_col=args.split_signal_offset_col,
        )
    except Exception as e:
        print(f"[ERROR] 读取 TSV 失败: {e}", file=sys.stderr)
        sys.exit(2)

    if not id2records:
        print("[ERROR] 输入 TSV 未读到有效记录。", file=sys.stderr)
        sys.exit(2)

    target_set = set(id2records.keys())
    found_set = set()

    with pysam.AlignmentFile(args.bam, "rb", check_sq=False) as bf, \
            open(args.output, "w", encoding="utf-8", newline="") as out:

        writer = csv.writer(out, delimiter="\t")
        writer.writerow([
            "subread_id",
            "parent_read_id",
            "is_split_read",
            "split_signal_offset",
            "base_start0",
            "base_end0",
            "base_start1",
            "base_end1",
            "subread_signal_start",
            "subread_signal_end",
            "parent_signal_start",
            "parent_signal_end",
            "stride",
            "ts",
            "query_length",
            "chrom",
            "bed_start",
            "bed_end",
            "bed_name",
            "overlap_ref_start",
            "overlap_ref_end",
            "strand",
            "is_reverse",
            "mapq",
        ])

        processed = 0
        matched = 0
        written = 0

        for read in bf:
            processed += 1
            qn = read.query_name
            if not qn:
                continue

            cid = clean_read_id(qn)
            if cid not in target_set:
                continue

            matched += 1
            found_set.add(cid)

            qseq = read.query_sequence
            if not qseq:
                print(f"[WARN] {qn}: 无 query_sequence，跳过", file=sys.stderr)
                continue

            try:
                mv = list(read.get_tag("mv", with_value_type=False))
            except KeyError:
                print(f"[WARN] {qn}: 缺少 mv tag，跳过", file=sys.stderr)
                continue

            try:
                ts = int(read.get_tag("ts"))
            except KeyError:
                print(f"[WARN] {qn}: 缺少 ts tag，跳过", file=sys.stderr)
                continue

            parsed = parse_mv_to_base_intervals_with_ts(
                mv,
                num_bases_expected=len(qseq),
                ts=ts,
            )
            if parsed is None:
                print(
                    f"[WARN] {qn}: mv 解析失败或与碱基数不匹配 "
                    f"(qlen={len(qseq)})，跳过",
                    file=sys.stderr
                )
                continue

            stride, intervals_trimmed, intervals_raw = parsed

            bam_parent_read_id, bam_split_signal_offset, bam_is_split_read = get_parent_info_from_bam(read)

            for rec in id2records[cid]:
                start0 = rec["start0"]   # 0-based half-open
                end0 = rec["end0"]       # 0-based half-open
                row = rec["row"]
                original_subread_id = rec["original_subread_id"]
                bed_name = rec["bed_name"]

                # 优先用 TSV 里的值；若缺失则用 BAM 里的标签兜底
                parent_read_id = rec.get("parent_read_id") or bam_parent_read_id
                split_signal_offset = rec.get("split_signal_offset", 0)
                if split_signal_offset == 0 and bam_split_signal_offset != 0:
                    split_signal_offset = bam_split_signal_offset

                is_split_read = 1 if split_signal_offset != 0 or parent_read_id != qn or bam_is_split_read == 1 else 0

                if start0 == end0:
                    print(f"[WARN] {original_subread_id}: 区间为空 [{start0}, {end0})，跳过", file=sys.stderr)
                    continue

                if start0 < 0 or end0 < 0 or start0 > end0:
                    print(f"[WARN] {original_subread_id}: 区间非法 [{start0}, {end0})，跳过", file=sys.stderr)
                    continue

                end0_closed = end0 - 1

                if start0 >= len(qseq) or end0_closed >= len(qseq):
                    print(
                        f"[WARN] {original_subread_id}: 碱基区间越界 "
                        f"(qlen={len(qseq)}, region=[{start0}, {end0}))，跳过",
                        file=sys.stderr
                    )
                    continue

                # 当前 subread 自身 raw signal 坐标
                subread_signal_start = intervals_raw[start0][0]
                subread_signal_end = intervals_raw[end0_closed][1]

                # 映射到原始 parent read 的 signal 坐标
                parent_signal_start = split_signal_offset + subread_signal_start
                parent_signal_end = split_signal_offset + subread_signal_end

                writer.writerow([
                    qn,
                    parent_read_id,
                    is_split_read,
                    split_signal_offset,
                    start0,
                    end0,
                    start0 + 1,
                    end0,
                    subread_signal_start,
                    subread_signal_end,
                    parent_signal_start,
                    parent_signal_end,
                    stride,
                    ts,
                    len(qseq),
                    row.get("chrom", ""),
                    row.get("bed_start", ""),
                    row.get("bed_end", ""),
                    bed_name,
                    row.get("overlap_ref_start", ""),
                    row.get("overlap_ref_end", ""),
                    row.get("strand", ""),
                    row.get("is_reverse", ""),
                    row.get("mapq", ""),
                ])
                written += 1

            if matched % 500 == 0:
                print(
                    f"[INFO] processed={processed:,} matched={matched:,} written={written:,}",
                    file=sys.stderr
                )

    print(f"[DONE] processed={processed:,}, matched={matched:,}, written={written:,}", file=sys.stderr)

    if args.write_not_found:
        not_found = [cid for cid in target_set if cid not in found_set]
        nf_path = args.output + ".not_found"
        with open(nf_path, "w", encoding="utf-8") as nf:
            for cid in not_found:
                nf.write(id2records[cid][0]["original_subread_id"] + "\n")
        print(f"[INFO] not_found={len(not_found)} 写入：{nf_path}", file=sys.stderr)
        

if __name__ == "__main__":
    main()