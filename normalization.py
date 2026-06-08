import os
import argparse
import numpy as np
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

def modified_zscore(signal, mad_threshold=3.5, consistency_correction=1.4826):
	median = np.median(signal)
	dev_from_med = np.array(signal) - median
	mad = np.median(np.abs(dev_from_med))
	mad_score = dev_from_med / (consistency_correction * mad)

	x = np.where(np.abs(mad_score) > mad_threshold)
	x = x[0]

	if len(x) > 0:
		for i in range(len(x)):
			if x[i] == 0:
				mad_score[x[i]] = mad_score[x[i] + 1]
			elif x[i] == len(mad_score) - 1:
				mad_score[x[i]] = mad_score[x[i] - 1]
			else:
				mad_score[x[i]] = (mad_score[x[i] - 1] + mad_score[x[i] + 1]) / 2
	return mad_score



def _chunkify_data(data, batch_size):
    n = len(data)
    for start in range(0, n, batch_size):
        yield start, data[start:start + batch_size]


def _process_train_chunk(task):
    data_chunk, cut, length, tile, tile_idx = task

    step = length // tile
    start = tile_idx * step
    segment_arr = []

    for signal in data_chunk:
        if len(signal) < cut + length:
            continue

        signal = signal[cut:]
        end = start + length

        while end <= len(signal):
            segment = modified_zscore(
                signal[end - length:end]
            )

            segment_arr.append(segment.astype(np.float16, copy=False))
            end += length

    if len(segment_arr) == 0:
        return None

    return np.asarray(segment_arr, dtype=np.float16)


def _process_valid_chunk(task):
    data_chunk, cut, length = task
    segment_arr = []

    for signal in data_chunk:
        if len(signal) < cut + length:
            continue

        segment = modified_zscore(
            signal[cut:cut + length]
        )

        segment_arr.append(segment.astype(np.float16, copy=False))

    if len(segment_arr) == 0:
        return None

    return np.asarray(segment_arr, dtype=np.float16)


def save_train_batches(
    data,
    out_dir,
    prefix,
    cut,
    length,
    tile,
    batch_size=5000,
    processes=4,
):
    os.makedirs(out_dir, exist_ok=True)

    ctx = mp.get_context("spawn")
    batch_id = 0
    total_segments = 0

    for raw_batch_start, data_batch in _chunkify_data(data, batch_size):
        print(f"\nProcessing {prefix} raw batch from index {raw_batch_start}, size={len(data_batch)}")

        tasks = []
        sub_chunks = np.array_split(data_batch, processes)

        for tile_idx in range(tile):
            for chunk in sub_chunks:
                if len(chunk) > 0:
                    tasks.append(
                        (
                            chunk,
                            cut,
                            length,
                            tile,
                            tile_idx,
                        )
                    )

        with ProcessPoolExecutor(max_workers=processes, mp_context=ctx) as executor:
            for result in executor.map(_process_train_chunk, tasks):
                if result is None or len(result) == 0:
                    continue

                out_path = os.path.join(out_dir, f"{prefix}_batch_{batch_id:06d}.npy")
                np.save(out_path, result)

                print(
                    f"Saved {out_path}, shape={result.shape}, dtype={result.dtype}"
                )

                total_segments += result.shape[0]
                batch_id += 1

    print(f"\nFinished {prefix}. Total saved segments: {total_segments}")


def save_valid_batches(
    data,
    out_dir,
    prefix,
    cut,
    length,
    batch_size=5000,
    processes=4,
):
    os.makedirs(out_dir, exist_ok=True)

    ctx = mp.get_context("spawn")
    batch_id = 0
    total_segments = 0

    for raw_batch_start, data_batch in _chunkify_data(data, batch_size):
        print(f"\nProcessing {prefix} raw batch from index {raw_batch_start}, size={len(data_batch)}")

        sub_chunks = np.array_split(data_batch, processes)
        tasks = [
            (
                chunk,
                cut,
                length,
            )
            for chunk in sub_chunks
            if len(chunk) > 0
        ]

        with ProcessPoolExecutor(max_workers=processes, mp_context=ctx) as executor:
            for result in executor.map(_process_valid_chunk, tasks):
                if result is None or len(result) == 0:
                    continue

                out_path = os.path.join(out_dir, f"{prefix}_batch_{batch_id:06d}.npy")
                np.save(out_path, result)

                print(
                    f"Saved {out_path}, shape={result.shape}, dtype={result.dtype}"
                )

                total_segments += result.shape[0]
                batch_id += 1

    print(f"\nFinished {prefix}. Total saved segments: {total_segments}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch-wise preprocessing and saving")
    parser.add_argument("--data_folder", "-d", type=str, required=True)
    parser.add_argument("--cut", "-c", type=int, default=1500)
    parser.add_argument("--tiling_fold", "-tf", type=int, default=3)
    parser.add_argument("--length", "-l", type=int, default=3000)
    parser.add_argument("--processes", "-p", type=int, default=4)
    parser.add_argument("--batch_size", "-bs", type=int, default=5000)

    args = parser.parse_args()

    train_path = os.path.join(args.data_folder, "train.npy")
    valid_path = os.path.join(args.data_folder, "valid.npy")

    train_out_dir = os.path.join(args.data_folder, "train_sw_batches")
    valid_out_dir = os.path.join(args.data_folder, "valid_sw_batches")

    print("\nLoad dataset!")
    train_data = np.load(train_path, allow_pickle=True)
    valid_data = np.load(valid_path, allow_pickle=True)

    print(f"Train shape: {train_data.shape}, dtype={train_data.dtype}")
    print(f"Valid shape: {valid_data.shape}, dtype={valid_data.dtype}")

    print("\nPreprocess train dataset batch by batch!")
    save_train_batches(
        data=train_data,
        out_dir=train_out_dir,
        prefix="train_sw",
        cut=args.cut,
        length=args.length,
        tile=args.tiling_fold,
        batch_size=args.batch_size,
        processes=args.processes,
    )

    print("\nPreprocess valid dataset batch by batch!")
    save_valid_batches(
        data=valid_data,
        out_dir=valid_out_dir,
        prefix="valid_sw",
        cut=args.cut,
        length=args.length,
        batch_size=args.batch_size,
        processes=args.processes,
    )

    print("\nAll done!")
