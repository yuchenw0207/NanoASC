import argparse
import csv
import functools
import logging
import os
from multiprocessing.pool import ThreadPool
from pathlib import Path
import time
import typing

import grpc
import numpy as np
import read_until
import torch
from torch import nn

from models.model import NanoASC


def build_model():
    return NanoASC(
        n_conv_neurons=[64, 64, 128, 256, 512],
        n_fc_neurons=512,
        depth=17,
        n_classes=2,
        shortcut=True,
        se_reduction=16,
        se_stages=(2, 3, 4),
        ms_dilations=(1, 2, 4),
        ms_stages=(2, 3, 4),
        ms_kernel_size=3,
        attn_hidden=128,
        head_dropout=0.2,
    )


def modified_zscore(signal, mad_threshold=3.5, consistency_correction=1.4826):
    median = np.median(signal)
    dev_from_med = np.array(signal) - median
    mad = np.median(np.abs(dev_from_med))

    if mad == 0:
        mad = 1e-8

    mad_score = dev_from_med / (consistency_correction * mad)

    x = np.where(np.abs(mad_score) > mad_threshold)[0]
    if len(x) > 0:
        for i in range(len(x)):
            if x[i] == 0:
                mad_score[x[i]] = mad_score[x[i] + 1]
            elif x[i] == len(mad_score) - 1:
                mad_score[x[i]] = mad_score[x[i] - 1]
            else:
                mad_score[x[i]] = (mad_score[x[i] - 1] + mad_score[x[i] + 1]) / 2

    return mad_score.astype(np.float32, copy=False)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("Read until API demonstration.")
    parser.add_argument("--host", default="127.0.0.1", help="MinKNOW server host.")
    parser.add_argument(
        "--port", type=int, default=8000, help="MinKNOW gRPC server port."
    )
    parser.add_argument(
        "--ca-cert",
        type=Path,
        default=None,
        help="Path to alternate CA certificate for connecting to MinKNOW.",
    )
    parser.add_argument("--workers", default=1, type=int, help="Worker threads.")
    parser.add_argument(
        "--analysis_delay",
        type=int,
        default=1,
        help="Period to wait before starting analysis.",
    )
    parser.add_argument(
        "--run_time", type=int, default=30, help="Period to run the analysis."
    )
    parser.add_argument(
        "--unblock_duration",
        type=float,
        default=0.1,
        help="Time in seconds to apply unblock voltage.",
    )
    parser.add_argument(
        "--one_chunk",
        default=False,
        action="store_true",
        help="Minimum read chunk size to receive.",
    )
    parser.add_argument(
        "--min_chunk_size",
        type=int,
        default=2000,
        help=(
            "Minimum read chunk size to receive. NOTE: this functionality is "
            "currently disabled; read chunks received will be unfiltered."
        ),
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Number of read chunks to request from the client per batch.",
    )
    parser.add_argument(
        "--debug",
        help="Print all debugging information.",
        action="store_const",
        dest="log_level",
        const=logging.DEBUG,
        default=logging.WARNING,
    )
    parser.add_argument(
        "--verbose",
        help="Print verbose messaging.",
        action="store_const",
        dest="log_level",
        const=logging.INFO,
    )
    parser.add_argument(
        "--model_state",
        type=str,
        required=True,
        help="Path of the model checkpoint (.pth file).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory.",
    )
    parser.add_argument(
        "--gpu_ids",
        type=str,
        default=None,
        help="Specify the GPU to use; if omitted, use all GPUs or CPU.",
    )
    return parser


def simple_analysis(
    model,
    device,
    client: read_until.ReadUntilClient,
    output: str,
    batch_size: int = 512,
    delay: float = 1,
    throttle: float = 0.1,
    unblock_duration: float = 0.1,
):
    """Run adaptive-sampling decisions on recent read chunks."""
    logger = logging.getLogger("Analysis")
    logger.warning(
        "Initialising simple analysis. "
        "Enable --verbose or --debug logging to see more."
    )
    logger.info("Starting analysis of reads in %ss.", delay)
    time.sleep(delay)

    os.makedirs(output, exist_ok=True)
    sampling_file = open(
        os.path.join(output, "adaptive_sampling.csv"),
        mode="w",
        newline="",
        encoding="utf-8",
    )
    sampling_writer = csv.writer(sampling_file)
    sampling_writer.writerow(
        ["batch_time", "read_number", "channel", "read_id", "num_samples", "decision"]
    )
    sampling_file.flush()

    target_counter = 0
    non_target_counter = 0
    short_counter = 0
    control_counter = 0

    model.eval()
    try:
        while client.is_running:
            time_begin = time.time()
            read_batch = client.get_read_chunks(batch_size=batch_size, last=True)
            read_list, inputs = [], []

            for channel, read in read_batch:
                raw_data = np.frombuffer(read.raw_data, client.signal_dtype)
                read.raw_data = bytes("", "utf8")
                signal_length = len(raw_data)

                # Channels 257-512 are used as a control group.
                if channel > 256:
                    control_counter += 1
                    client.stop_receiving_read(channel, read.number)
                    row = [
                        time_begin,
                        read.number,
                        channel,
                        read.id,
                        signal_length,
                        "control",
                    ]
                    sampling_writer.writerow(row)
                    continue

                if signal_length < 4500:
                    short_counter += 1
                    row = [
                        time_begin,
                        read.number,
                        channel,
                        read.id,
                        signal_length,
                        "short",
                    ]
                    sampling_writer.writerow(row)
                else:
                    read_list.append((channel, read, signal_length))
                    normal_data = modified_zscore(raw_data[-3000:])
                    inputs.append(normal_data)

            if len(inputs) > 0:
                input_tensor = torch.as_tensor(
                    np.asarray(inputs, dtype=np.float32),
                    dtype=torch.float32,
                    device=device,
                )

                with torch.no_grad():
                    outputs = model(input_tensor).max(dim=1).indices.detach().cpu().numpy()

                target_reads, non_target_reads = [], []
                for index, (channel, read, signal_length) in enumerate(read_list):
                    if int(outputs[index]) == 1:
                        target_counter += 1
                        target_reads.append((channel, read.number))
                        row = [
                            time_begin,
                            read.number,
                            channel,
                            read.id,
                            signal_length,
                            "stop_receiving",
                        ]
                        sampling_writer.writerow(row)
                    else:
                        non_target_counter += 1
                        non_target_reads.append((channel, read.number))
                        row = [
                            time_begin,
                            read.number,
                            channel,
                            read.id,
                            signal_length,
                            "unblock",
                        ]
                        sampling_writer.writerow(row)

                if target_reads:
                    client.stop_receiving_batch(target_reads)
                if non_target_reads:
                    client.unblock_read_batch(non_target_reads, duration=unblock_duration)

            sampling_file.flush()
            time_end = time.time()
            if time_begin + throttle > time_end:
                time.sleep(throttle + time_begin - time_end)
            if len(read_batch) > 0:
                print(
                    "batch time: {}, batch size: {}, target reads: {}, "
                    "non-target reads: {}, short reads: {}, control group reads: {}".format(
                        time_end - time_begin,
                        len(read_batch),
                        target_counter,
                        non_target_counter,
                        short_counter,
                        control_counter,
                    )
                )
    finally:
        sampling_file.close()

    return target_counter


def run_workflow(
    client: read_until.ReadUntilClient,
    analysis_worker: typing.Callable[[], None],
    n_workers: int,
    run_time: float,
    runner_kwargs: typing.Optional[typing.Dict] = None,
):
    """Run an analysis function against a ReadUntilClient."""
    logger = logging.getLogger("Manager")

    if not runner_kwargs:
        runner_kwargs = {}

    results = []
    pool = ThreadPool(n_workers)
    logger.info("Creating %s workers", n_workers)
    try:
        client.run(**runner_kwargs)

        for _ in range(n_workers):
            results.append(pool.apply_async(analysis_worker))
        pool.close()

        time.sleep(run_time)
        logger.info("Sending reset")
        client.reset()
        pool.join()
    except KeyboardInterrupt:
        logger.info("Caught ctrl-c, terminating workflow.")
        client.reset()

    collected = []
    for result in results:
        try:
            res = result.get(3)
        except TimeoutError:
            logger.warning("Worker function did not exit successfully.")
            collected.append(None)
        except Exception:
            logger.exception("Worker raised exception:")
        else:
            logger.info("Worker exited successfully.")
            collected.append(res)
    pool.terminate()
    return collected


def main(argv=None):
    args = get_parser().parse_args(argv)

    logging.basicConfig(
        format="[%(asctime)s - %(name)s] %(message)s",
        datefmt="%H:%M:%S",
        level=args.log_level,
    )

    if args.gpu_ids:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_ids

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model()
    model = nn.DataParallel(model).to(device)
    model.load_state_dict(torch.load(args.model_state, map_location=device))
    model.eval()

    print(f"Run in {device} {args.gpu_ids}")

    channel_credentials = None
    if args.ca_cert is not None:
        channel_credentials = grpc.ssl_channel_credentials(
            root_certificates=args.ca_cert.read_bytes()
        )

    read_until_client = read_until.ReadUntilClient(
        mk_host=args.host,
        mk_port=args.port,
        mk_credentials=channel_credentials,
        one_chunk=args.one_chunk,
        filter_strands=True,
    )

    analysis_worker = functools.partial(
        simple_analysis,
        model,
        device,
        client=read_until_client,
        output=args.output,
        batch_size=args.batch_size,
        delay=args.analysis_delay,
        unblock_duration=args.unblock_duration,
    )

    results = run_workflow(
        read_until_client,
        analysis_worker,
        args.workers,
        args.run_time,
        runner_kwargs={"min_chunk_size": args.min_chunk_size},
    )

    for idx, result in enumerate(results):
        logging.info("Worker %s received %s target reads", idx + 1, result)


if __name__ == "__main__":
    main()
