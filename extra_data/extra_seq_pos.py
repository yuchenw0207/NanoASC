#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import pysam
from collections import defaultdict


# CIGAR op codes in pysam
# 0 M
# 1 I
# 2 D
# 3 N
# 4 S
# 5 H
# 6 P
# 7 =
# 8 X
# 9 B (obsolete)

MATCH_LIKE = {0, 7, 8}


def load_bed(bed_path):
    """
    Load BED file.
    BED assumed: chrom start end [name...]
    Coordinates are 0-based half-open.
    """
    bed_by_chr = defaultdict(list)
    with open(bed_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 3:
                continue
            chrom = fields[0]
            start = int(fields[1])
            end = int(fields[2])
            name = fields[3] if len(fields) >= 4 else "."
            if end <= start:
                continue
            bed_by_chr[chrom].append((start, end, name))
    for chrom in bed_by_chr:
        bed_by_chr[chrom].sort()
    return bed_by_chr


def get_softclips(cigartuples):
    """
    Return left and right softclip lengths on BAM stored query orientation.
    """
    left_s = 0
    right_s = 0
    if not cigartuples:
        return left_s, right_s
    if cigartuples[0][0] == 4:
        left_s = cigartuples[0][1]
    if cigartuples[-1][0] == 4:
        right_s = cigartuples[-1][1]
    return left_s, right_s


def ref_interval_to_query_interval(aln, target_start, target_end):
    """
    Map reference interval [target_start, target_end) to query interval
    using reference_start + CIGAR.

    Returns query coordinates on BAM stored query orientation.

    Output:
        q_start, q_end   0-based half-open on stored query orientation
        None, None       if no aligned overlap is found
    """
    cigartuples = aln.cigartuples
    if cigartuples is None:
        return None, None

    ref_pos = aln.reference_start
    query_pos = 0

    q_start = None
    q_end = None

    for op, length in cigartuples:
        if op in MATCH_LIKE:
            block_ref_start = ref_pos
            block_ref_end = ref_pos + length

            ov_start = max(block_ref_start, target_start)
            ov_end = min(block_ref_end, target_end)

            if ov_start < ov_end:
                if q_start is None:
                    q_start = query_pos + (ov_start - block_ref_start)
                q_end = query_pos + (ov_end - block_ref_start)

            ref_pos += length
            query_pos += length

        elif op == 1:   # I
            query_pos += length
        elif op == 4:   # S
            query_pos += length
        elif op == 2 or op == 3:   # D or N
            ref_pos += length
        elif op == 5 or op == 6:   # H or P
            pass
        else:
            pass

    return q_start, q_end


def stored_to_original_query_coords(aln, q_start, q_end):
    """
    Convert coordinates on BAM stored query orientation to coordinates on original read orientation.

    For forward alignments:
        unchanged
    For reverse alignments:
        original_start = read_len - q_end
        original_end   = read_len - q_start
    """
    qlen = aln.query_length
    if qlen is None:
        return None, None

    if not aln.is_reverse:
        return q_start, q_end
    else:
        return qlen - q_end, qlen - q_start


def get_parent_info(aln):
    """
    处理 Dorado split-read 情况：
      - pi: original/parent read id
      - sp: signal sample offset in parent read

    对正常 read：
      - 没有 pi/sp，则 parent_read_id = query_name, split_signal_offset = 0
    """
    subread_id = aln.query_name

    if aln.has_tag("pi"):
        parent_read_id = aln.get_tag("pi")
        is_split_read = 1
    else:
        parent_read_id = subread_id
        is_split_read = 0

    if aln.has_tag("sp"):
        split_signal_offset = int(aln.get_tag("sp"))
    else:
        split_signal_offset = 0

    return subread_id, parent_read_id, split_signal_offset, is_split_read


def main():
    parser = argparse.ArgumentParser(
        description="Extract subread base interval corresponding to target BED regions from BAM, with pi/sp support."
    )
    parser.add_argument("-b", "--bed", required=True, help="Input BED file")
    parser.add_argument("-a", "--bam", required=True, help="Input BAM file (indexed)")
    parser.add_argument("-o", "--out", required=True, help="Output TSV file")
    parser.add_argument(
        "--skip-secondary",
        action="store_true",
        help="Skip secondary alignments"
    )
    parser.add_argument(
        "--skip-supplementary",
        action="store_true",
        help="Skip supplementary alignments"
    )
    parser.add_argument(
        "--mapq",
        type=int,
        default=0,
        help="Minimum MAPQ to keep"
    )

    args = parser.parse_args()

    bed_by_chr = load_bed(args.bed)
    bam = pysam.AlignmentFile(args.bam, "rb")

    with open(args.out, "w") as out:
        out.write(
            "\t".join([
                "subread_id",
                "parent_read_id",
                "split_signal_offset",
                "is_split_read",
                "chrom",
                "bed_start",
                "bed_end",
                "bed_name",
                "overlap_ref_start",
                "overlap_ref_end",
                "strand",
                "is_reverse",
                "mapq",
                "read_length",
                "left_softclip_stored",
                "right_softclip_stored",
                "query_start_stored0",
                "query_end_stored0",
                "read_start0",
                "read_end0",
                "read_start1",
                "read_end1",
            ]) + "\n"
        )

        for chrom, intervals in bed_by_chr.items():
            if chrom not in bam.references:
                continue

            for bed_start, bed_end, bed_name in intervals:
                for aln in bam.fetch(chrom, bed_start, bed_end):
                    if aln.is_unmapped:
                        continue
                    if aln.mapping_quality < args.mapq:
                        continue
                    if args.skip_secondary and aln.is_secondary:
                        continue
                    if args.skip_supplementary and aln.is_supplementary:
                        continue

                    aln_ref_start = aln.reference_start
                    aln_ref_end = aln.reference_end
                    if aln_ref_end is None:
                        continue

                    ov_start = max(bed_start, aln_ref_start)
                    ov_end = min(bed_end, aln_ref_end)
                    if ov_start >= ov_end:
                        continue

                    q_start, q_end = ref_interval_to_query_interval(aln, ov_start, ov_end)
                    if q_start is None or q_end is None:
                        continue

                    read_start0, read_end0 = stored_to_original_query_coords(aln, q_start, q_end)
                    if read_start0 is None or read_end0 is None:
                        continue

                    read_start1 = read_start0 + 1
                    read_end1 = read_end0

                    left_s, right_s = get_softclips(aln.cigartuples)
                    subread_id, parent_read_id, split_signal_offset, is_split_read = get_parent_info(aln)

                    out.write(
                        "\t".join(map(str, [
                            subread_id,
                            parent_read_id,
                            split_signal_offset,
                            is_split_read,
                            chrom,
                            bed_start,
                            bed_end,
                            bed_name,
                            ov_start,
                            ov_end,
                            "-" if aln.is_reverse else "+",
                            int(aln.is_reverse),
                            aln.mapping_quality,
                            aln.query_length,
                            left_s,
                            right_s,
                            q_start,
                            q_end,
                            read_start0,
                            read_end0,
                            read_start1,
                            read_end1,
                        ])) + "\n"
                    )

    bam.close()


if __name__ == "__main__":
    main()