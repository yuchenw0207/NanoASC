import os
import argparse
import numpy as np


def merge_npy_batches(input_dir, output_path, prefix=None, dtype=np.float16):
    files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.endswith(".npy") and (prefix is None or f.startswith(prefix))
    ])

    if len(files) == 0:
        raise RuntimeError(f"No npy files found in {input_dir}")

    print(f"Found {len(files)} batch files.")

    total_rows = 0
    sample_shape = None

    for f in files:
        arr = np.load(f, mmap_mode="r", allow_pickle=True)

        if sample_shape is None:
            sample_shape = arr.shape[1:]
            print(f"Sample shape: {sample_shape}, dtype={arr.dtype}")
        else:
            if arr.shape[1:] != sample_shape:
                raise ValueError(
                    f"Shape mismatch: {f}, got {arr.shape[1:]}, expected {sample_shape}"
                )

        total_rows += arr.shape[0]

    final_shape = (total_rows,) + sample_shape
    print(f"Final merged shape: {final_shape}")

    merged = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=dtype,
        shape=final_shape,
    )

    offset = 0
    for f in files:
        arr = np.load(f, mmap_mode="r", allow_pickle=True)
        n = arr.shape[0]

        merged[offset:offset + n] = arr.astype(dtype, copy=False)

        print(f"Merged {f}, rows={n}, offset={offset}")
        offset += n

    merged.flush()
    print(f"\nSaved merged npy to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge batch npy files into one npy")
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--prefix", type=str, default=None)

    args = parser.parse_args()

    merge_npy_batches(
        input_dir=args.input_dir,
        output_path=args.output_path,
        prefix=args.prefix,
        dtype=np.float16,
    )