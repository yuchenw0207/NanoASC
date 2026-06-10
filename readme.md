# NanoASC：A Deep Learning Model for Intraspecies and Interspecies Adaptive Sequencings

NanoASC is a deep learning model for classifying Oxford Nanopore raw electrical signals as target or non-target for adaptive sequencing. Built on a lightweight 1D residual backbone with SE attention, multi-scale convolutions, and temporal attention pooling, NanoASC supports both intraspecies human gene-panel classification and interspecies mixed-species classification. NanoASC provides stable and competitive raw-signal classification performance across R9 and R10 nanopore datasets, supporting Read Until-style sequence enrichment or depletion.

## Install

### Dependencies

```
python=3.9

numpy=1.24.3

pytorch=1.12.1

cudatoolkit=11.6

scikit-learn=1.2.2

matplotlib-base=3.7.1

tqdm=4.65.0

h5py=3.13.0

pysam=0.23.3

ont-fast5-api=4.1.1

ont_vbz_hdf_plugin=1.0.1

samtools=1.16.1

pod5==0.3.28
```



### Install NanoASC by Conda

1. #### Install Conda

2. #### Download NanoASC code

3. #### Create conda environment

   ```
   conda env create -f environment.yaml
   ```

   

## Start

### Human panel

#### extra_signal_pos.sh

Process one or more panel-mapped BAM files to generate target-region signal-position TSVs, filter qualified non-split reads, and create four-fold train/valid/test TSV splits.

| Parameter          | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| `<panel_bed_path>` | Target panel selector or a direct BED file path. Supported selectors are `148`, `odd`, and `all`. They map to `panel_bed/148_genes.bed`, `panel_bed/odd_cosmic_genes.bed`, and `panel_bed/all_cosmic_genes.bed`. If the value is not one of these selectors, it is treated as a BED file path. |
| `<bam_dir>`        | Directory containing one or more panel-mapped BAM files. Files ending in `.bam` or `.BAM` are processed. |
| `<out_dir>`        | Output directory. The script creates intermediate and final TSV outputs under this directory. |

The input BAM files should contain the tags `mv` and `ts`; split-read cases may also use `pi` and `sp`.

Examples:

Use the built-in `148_gene` panel:

```bash
bash extra_data/extra_signal_pos.sh \
  148 \
  bam_dir \
  output_dir
```

#### extra_data.sh

Extract positive and negative raw signal samples from `pod5` or `fast5` data using four-fold TSV splits and panel-mapped BAM files, then build a four-fold training dataset.

| Parameter         | Description                                                  |
| ----------------- | ------------------------------------------------------------ |
| `<raw_path_txt>`  | Text file containing one `pod5/fast5` file path or raw-signal directory path per line. |
| `<split_dir>`     | Four-fold split directory. It should contain `fold0/fold1/fold2/fold3`; `fold_0/fold_1/fold_2/fold_3` are also accepted. Each fold directory must contain `train.tsv`, `valid.tsv`, and `test.tsv`. |
| `<out_dir>`       | Root output directory. The script creates `pos/` and `neg/` under this directory. |
| `<panel_bam_dir>` | Directory containing panel-mapped BAM files. Unmapped read IDs from these BAM files are used as the negative-sample source. |
| `<pod5/fast5>`    | Raw signal type. Must be either `pod5` or `fast5`.           |

Examples:

```bash
bash extra_data/extra_data.sh \
  raw_pod5_paths.txt \
  split_dir \
  out_dir \
  panel_bam_dir \
  pod5
```

#### normalization.sh

Run `normalization.py` on all four positive/negative sample folds, then merge the generated batches with `merge_data.py` into preprocessed `.npy` files for training and validation.

| Parameter                          | Description                                                  |
| ---------------------------------- | ------------------------------------------------------------ |
| `<pos_root>`                       | Root directory for positive sample folds. It should contain `fold0`, `fold1`, `fold2`, and `fold3`; each fold should contain `train.npy` and `valid.npy`. |
| `<neg_root>`                       | Root directory for negative sample folds. It should contain `fold0`, `fold1`, `fold2`, and `fold3`; each fold should contain `train.npy` and `valid.npy`. |
| `[normalization.py extra args...]` | Additional arguments passed to `normalization.py`, such as `-l`, `-tf`, `-p`, and `-bs`. |

Examples:

```bash
bash extra_data/normalization.sh \
  data/pos \
  data/neg \
  -l 3000 \
  -tf 3 \
  -p 8 \
  -bs 5000
```

#### train.sh

Run `train.py` on all four positive/negative sample folds and save each fold's model and training log to its own output directory.

| Parameter                  | Description                                                  |
| -------------------------- | ------------------------------------------------------------ |
| `<pos_root>`               | Root directory for positive sample folds. It should contain `fold0`, `fold1`, `fold2`, and `fold3`. |
| `<neg_root>`               | Root directory for negative sample folds. It should contain `fold0`, `fold1`, `fold2`, and `fold3`. |
| `<out_root>`               | Root output directory for training results. The script writes to `<out_root>/fold0` through `<out_root>/fold3`. |
| `[train.py extra args...]` | Additional arguments passed to `train.py`, such as `-e`, `-b`, `-lr`, `-nw`, and `-g`. |

Examples:

```bash
bash extra_data/train.sh \
  data/pos \
  data/neg \
  results/train \
  -e 200 \
  -b 512 \
  -lr 0.001 \
  -nw 2 \
  -g 0
```

### species 

#### read_raw_signal.py

Read raw signals from `fast5` or `pod5` files, optionally filter by read ID, and save shuffled `train.npy`, `valid.npy`, and `test.npy` splits.

| Parameter              | Description                                                  |
| ---------------------- | ------------------------------------------------------------ |
| `--file_dir, -dir`     | Directory containing raw `fast5` or `pod5` files. The script searches recursively. Required. |
| `--input_type, -type`  | Raw input file type. Must be `fast5` or `pod5`. Required.    |
| `--output, -o`         | Output directory for `train.npy`, `valid.npy`, and `test.npy`. Required. |
| `--read_ids, -ids`     | Optional text file containing read IDs to keep. The first whitespace-separated column is used. |
| `--min_length, -len`   | Minimum raw signal length to keep. Default: `4500`.          |
| `--train_size, -train` | Requested number of training signals. Default: `20000`.      |
| `--valid_size, -valid` | Requested number of validation signals. Default: `10000`.    |
| `--test_size, -test`   | Requested number of test signals. Default: `10000`.          |
| `--seed`               | Random seed used before splitting. Default: `42`.            |

If fewer reads are available than the requested total, the script falls back to a 2:1:1 train/valid/test split using all accepted reads.

Examples:

Read `pod5` files without a read-ID filter

```bash
python extra_data/read_raw_signal.py \
  -dir data/raw_pod5 \
  -type pod5 \
  -o data/raw_split \
  -len 4500 \
  -train 20000 \
  -valid 10000 \
  -test 10000
```

## Scripts

### normalization.py and merge_data.py

Run the preprocessing workflow for one fold: `normalization.py` normalizes raw `train.npy` and `valid.npy` signals into batch files, then `merge_data.py` merges those batches into `train_preprocessed.npy` and `valid_preprocessed.npy` for model training.

#### Step1:normalization.py

```bash
python normalization.py --data_folder <fold_dir> [options]
```

| Parameter            | Description                                                  |
| -------------------- | ------------------------------------------------------------ |
| `--data_folder, -d`  | Input fold directory containing `train.npy` and `valid.npy`. Required. |
| `--cut, -c`          | Number of signal points to skip from the beginning of each read. Default: `1500`. |
| `--tiling_fold, -tf` | Number of staggered training-window offsets. `step = length // tiling_fold`. Default: `3`. |
| `--length, -l`       | Segment length for each normalized signal window. Default: `3000`. |
| `--processes, -p`    | Number of worker processes. Default: `4`.                    |
| `--batch_size, -bs`  | Number of raw reads processed per outer batch. Default: `5000`. |

Training data is converted into multiple `length`-sized windows after `cut`, using staggered offsets controlled by `tiling_fold`. Validation data uses the first `length` points after `cut`.

#### Step2:merge_data.py

```bash
python merge_data.py --input_dir <batch_dir> --output_path <merged.npy> [options]
```

| Parameter       | Description                                                  |
| --------------- | ------------------------------------------------------------ |
| `--input_dir`   | Directory containing batch `.npy` files generated by `normalization.py`. Required. |
| `--output_path` | Output path for the merged `.npy` file. Required.            |
| `--prefix`      | Optional filename prefix filter. Only `.npy` files starting with this prefix are merged. |

`merge_data.py` checks that all batch files have matching sample shapes, allocates the final merged array with `np.lib.format.open_memmap`, and writes the result as `float16`.

#### Example

Preprocess one positive fold.

```bash
python normalization.py \
  -d data/pos/fold0 \
  -c 0 \
  -l 3000 \
  -tf 3 \
  -p 8 \
  -bs 5000

python merge_data.py \
  --input_dir data/pos/fold0/train_sw_batches \
  --output_path data/pos/fold0/train_preprocessed.npy

python merge_data.py \
  --input_dir data/pos/fold0/valid_sw_batches \
  --output_path data/pos/fold0/valid_preprocessed.npy
```

### train.py

Train the `NanoASC` model using preprocessed positive and negative training/validation datasets, then evaluate the best model on raw positive and negative test reads.

| Parameter               | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| `--pos_data_folder, -p` | Positive sample fold directory. It must contain `train_preprocessed.npy`, `valid_preprocessed.npy`, and `test.npy`. Required. |
| `--neg_data_folder, -n` | Negative sample fold directory. It must contain `train_preprocessed.npy`, `valid_preprocessed.npy`, and `test.npy`. Required. |
| `--output, -o`          | Output directory for logs, plots, and model checkpoint. Required. |
| `--cut, -c`             | Number of raw test-signal points to skip before inference. Default: `0`. |
| `--length, -l`          | Raw test-signal segment length used during final evaluation. Default: `3000`. |
| `--batch_size, -b`      | Total training batch size. The script uses half for positive samples and half for negative samples. Default: `1024`. |
| `--epochs, -e`          | Maximum number of training epochs. Default: `300`.           |
| `--learning_rate, -lr`  | Adam learning rate. Default: `1e-3`.                         |
| `--tolerance, -t`       | Number of validation checks without loss improvement before reducing the learning rate or early stopping. Default: `10`. |
| `--interm, -i`          | Optional checkpoint path used to initialize the model.       |
| `--num_workers, -nw`    | `DataLoader` worker count. Default: `0`.                     |
| `--gpu_ids, -g`         | CUDA device IDs to expose through `CUDA_VISIBLE_DEVICES`. If omitted, PyTorch chooses CUDA when available, otherwise CPU. |

After training, the script reloads `<output_dir>/model.pth` and reports accuracy, precision, recall, F1 score, and average inference time on `test.npy`.

Example:

```bash
python train.py \
  -p data/pos/fold0 \
  -n data/neg/fold0 \
  -o results/fold0_resume \
  -i results/fold0/model.pth \
  -g 0
```

### test.py

Evaluate a trained `NanoASC` checkpoint on positive and negative raw `test.npy` reads and report classification metrics.

| Parameter               | Description                                                  |
| ----------------------- | ------------------------------------------------------------ |
| `--pos_data_folder, -p` | Positive sample fold directory containing `test.npy`. Required. |
| `--neg_data_folder, -n` | Negative sample fold directory containing `test.npy`. Required. |
| `--model_state, -ms`    | Path to the trained model checkpoint, usually `model.pth`. Required. |
| `--output, -o`          | Output directory for `test.txt`. Required.                   |
| `--batch_size, -b`      | Inference batch size. Default: `512`.                        |
| `--cut, -c`             | Number of raw signal points to skip before inference. Default: `0`. |
| `--length, -len`        | Signal segment length used for inference. Default: `3000`.   |
| `--gpu_ids, -g`         | CUDA device IDs to expose through `CUDA_VISIBLE_DEVICES`. If omitted, PyTorch chooses CUDA when available, otherwise CPU. |

Each accepted read is sliced as `read[cut:cut + length]`, normalized with modified z-score normalization, and passed to the model. Reads shorter than `cut + length` are rejected and counted in the log.

Example:

```bash
python tester.py \
  -p data/pos/fold0 \
  -n data/neg/fold0 \
  -ms results/fold0/model.pth \
  -o results/fold0_test \
  -b 512 \
  -c 0 \
  -len 3000 \
  -g 0
```

