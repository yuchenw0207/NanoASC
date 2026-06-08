import argparse
import os
import re
import numpy as np
from pathlib import Path
from ont_fast5_api.fast5_interface import get_fast5_file
from tqdm import tqdm


def parse_exclude_txt(txt_path: str) -> set:
    """
    解析 txt 文件，提取 read ID
    """
    exclude_reads = set()
    print(f"===== 开始解析 read ID txt 文件: {txt_path} =====")

    with open(txt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            parts = re.split(r"\s+", line)
            read_id = parts[0].strip()
            exclude_reads.add(read_id)

            if len(exclude_reads) % 1000 == 0:
                print(f"已解析 {len(exclude_reads)} 个 read ID")

    print(f"\n(read ID txt 解析完成) 共获取 {len(exclude_reads)} 个 read ID\n")
    return exclude_reads


def parse_fast5_path_txt(fast5_txt_path: str, recursive: bool = False) -> list:
    """
    解析 fast5 路径 txt 文件。

    txt 每行可以是：
      1. 一个 fast5 文件路径
      2. 一个包含 fast5 文件的目录路径

    返回：
      fast5 文件路径列表
    """
    fast5_files = []
    seen = set()

    print(f"===== 开始解析 fast5 路径 txt 文件: {fast5_txt_path} =====")

    with open(fast5_txt_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            path_str = line.strip()

            if not path_str:
                continue

            # 支持注释行
            if path_str.startswith("#"):
                continue

            path = Path(path_str)

            if path.is_file() and path.suffix == ".fast5":
                real_path = str(path.resolve())
                if real_path not in seen:
                    fast5_files.append(path)
                    seen.add(real_path)

            elif path.is_dir():
                if recursive:
                    files = sorted(path.rglob("*.fast5"))
                else:
                    files = sorted(path.glob("*.fast5"))

                print(f"[目录] {path} -> 发现 {len(files)} 个 fast5 文件")

                for f5 in files:
                    real_path = str(f5.resolve())
                    if real_path not in seen:
                        fast5_files.append(f5)
                        seen.add(real_path)

            else:
                print(f"[警告] 第 {line_num} 行路径无效或不是 fast5 文件/目录: {path_str}")

    print(f"\n(fast5 路径解析完成) 共发现 {len(fast5_files)} 个 fast5 文件\n")

    if len(fast5_files) == 0:
        raise FileNotFoundError(f"未从 {fast5_txt_path} 中找到任何 fast5 文件")

    return fast5_files


def save_batch_to_npy(batch_signals: list, output_dir: str, batch_idx: int):
    """
    将一批信号保存为一个 npy 文件
    """
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"batch_{batch_idx:05d}.npy")

    arr = np.array(batch_signals, dtype=object)
    np.save(out_path, arr, allow_pickle=True)

    print(f"\n[保存] {out_path}  shape={arr.shape}")


def extract_filtered_signals_in_batches(
    fast5_files: list,
    exclude_reads: set,
    num: int,
    output_dir: str,
    min_length: int = 5000,
    batch_size: int = 5000,
):
    """
    从多个 fast5 文件中提取符合条件的信号，并按批次保存为多个 npy 文件。

    当前保留逻辑：
      只保留 read_id 在 exclude_reads 中的信号

    如果你想改成“排除这些 read”，把：

      if read_id not in exclude_reads:
          continue

    改成：

      if read_id in exclude_reads:
          continue
    """
    total_saved = 0
    batch_idx = 1
    batch_signals = []

    failed_files = 0

    for f5_path in tqdm(fast5_files, desc="Processing fast5 files"):
        try:
            with get_fast5_file(str(f5_path), "r") as f5:
                for read in f5.get_reads():
                    read_id = read.read_id

                    # 当前逻辑：只提取 read ID txt 中出现的 read
                    if read_id not in exclude_reads:
                        continue

                    raw = read.get_raw_data(scale=False)

                    if len(raw) < min_length:
                        continue

                    batch_signals.append(np.asarray(raw))
                    total_saved += 1

                    if total_saved % 1000 == 0:
                        print(f"\r已收集 {total_saved} 条信号", end="", flush=True)

                    if len(batch_signals) >= batch_size:
                        save_batch_to_npy(batch_signals, output_dir, batch_idx)
                        batch_idx += 1
                        batch_signals = []


        except Exception as e:
            failed_files += 1
            print(f"\n[警告] 读取 fast5 文件失败: {f5_path}")
            print(f"错误信息: {e}")
            continue


    # 保存最后不足一批的部分
    if batch_signals:
        save_batch_to_npy(batch_signals, output_dir, batch_idx)

    print(f"\n共提取并保存 {total_saved} 条符合条件的信号")
    print(f"失败 fast5 文件数: {failed_files}")
    print(f"输出目录: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="从多个 fast5 目录/文件中提取指定 read 的长信号，并按批次保存为 npy")

    parser.add_argument(
        "read_id_txt",
        help="read ID txt 文件路径，每行一个 read ID"
    )

    parser.add_argument(
        "fast5_path_txt",
        help="fast5 路径 txt 文件，每行一个 fast5 文件路径或 fast5 文件夹路径"
    )

    parser.add_argument(
        "-o", "--output_dir",
        default="filtered_signals_batches",
        help="输出目录，默认：filtered_signals_batches"
    )

    parser.add_argument(
        "-l", "--min_length",
        type=int,
        default=5000,
        help="信号最小长度，默认：5000"
    )

    parser.add_argument(
        "-num", "--number",
        type=int,
        default=100000,
        help="最多提取多少条信号，默认：100000"
    )

    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=5000,
        help="每个 npy 保存多少条信号，默认：5000"
    )

    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        help="如果 fast5_path_txt 中包含目录，则递归搜索子目录下的 .fast5 文件"
    )

    args = parser.parse_args()

    exclude_reads = parse_exclude_txt(args.read_id_txt)

    if not exclude_reads:
        print("警告：read ID txt 文件中未找到任何 read ID")

    fast5_files = parse_fast5_path_txt(
        fast5_txt_path=args.fast5_path_txt,
        recursive=args.recursive
    )

    extract_filtered_signals_in_batches(
        fast5_files=fast5_files,
        exclude_reads=exclude_reads,
        num=args.number,
        output_dir=args.output_dir,
        min_length=args.min_length,
        batch_size=args.batch_size,
    )

    print("\n===== 全部流程完成 =====")
    print(f"统计：read ID txt 中 {len(exclude_reads)} 个 read ID")
    print(f"fast5 文件数：{len(fast5_files)}")
    print(f"最多提取：{args.number} 条")
    print(f"输出目录：{args.output_dir}")


if __name__ == "__main__":
    main()