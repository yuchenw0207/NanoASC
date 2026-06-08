import os
import torch
import time
import argparse
from torch import nn
import numpy as np
from models.newmodel1 import ReadCurrentTargetRegion

MAD_SCORE = 3
REPEATED = False
#from models.tfmda_replaced_model import ReadCurrentTimeFreqFusionGated
def sliding_window_normalization(signal, length=3000, window_length=400, step=400):
    if not isinstance(signal, np.ndarray):
        signal = np.array(signal)

    if len(signal) < length:
        raise ValueError(f"输入信号长度必须为{length}，实际为 {len(signal)}")

    n = len(signal)
    normalized_signal = np.zeros_like(signal, dtype=np.float64)

    # 计算窗口数
    num_windows = (n - window_length) // step + 1

    last_mean = None
    last_std = None

    for i in range(num_windows):
        start = i * step
        end = start + window_length
        if end > n:
            end = n

        window = signal[start:end]
        mean = np.mean(window)
        std = np.std(window)
        if std < 1e-10:
            std = 1e-10

        normalized_window = (window - mean) / std
        normalized_signal[start:end] = normalized_window

        # 记录最后一个完整窗口的统计量
        last_mean = mean
        last_std = std

    # 处理最后不足一个窗口长度的尾巴
    # 找到最后一个窗口的结束位置
    last_start = (num_windows - 1) * step
    last_end = last_start + window_length
    if last_end < n:
        # 如果你想用“最后一个完整窗口”的统计量：
        mean = last_mean
        std = last_std

        if std < 1e-10:
            std = 1e-10

        tail_start = last_end
        normalized_signal[tail_start:] = (signal[tail_start:] - mean) / std

    return normalized_signal

def modified_zscore(data, consistency_correction=1.4826):
    """
    原始 MAD modified z-score 归一化方法。
    """
    data = np.asarray(data, dtype=np.float32)

    median = np.median(data)
    dev_from_med = data - median
    mad = np.median(np.abs(dev_from_med))

    # 防止 mad = 0 导致除零
    if mad == 0:
        mad = 1e-8

    mad_score = dev_from_med / (consistency_correction * mad)

    x = np.where(np.abs(mad_score) > MAD_SCORE)[0]

    while True:
        if len(x) > 0:
            for i in range(len(x)):
                idx = x[i]
                if idx == 0:
                    mad_score[idx] = mad_score[idx + 1]
                elif idx == len(mad_score) - 1:
                    mad_score[idx] = mad_score[idx - 1]
                else:
                    mad_score[idx] = (mad_score[idx - 1] + mad_score[idx + 1]) / 2
        else:
            break

        if REPEATED:
            x = np.where(np.abs(mad_score) > MAD_SCORE)[0]
        else:
            break

    return np.asarray(mad_score, dtype=np.float32)

def cut_patchs(signal, seq_length, stride, patch_size):
	split_signal = np.zeros((patch_size, seq_length), dtype="float32")
	for i in range(seq_length):
		split_signal[:, i] = signal[(i*stride):(i*stride)+patch_size]
	return split_signal


def myprint(string, log):
	log.write(string+'\n')
	print(string)


def inference(inputs, model, label, device):
	true_pred, false_pred = 0, 0
	x = np.array(inputs)
	x_time = torch.FloatTensor(x[:, :3000]).to(device)
	#x_freq = torch.FloatTensor(x[:, 3000:]).to(device)
	outputs = model(x_time)
	preds = outputs.max(dim=1).indices
	for y in preds:
		if int(y.item()) == label:
			true_pred += 1
		else:
			false_pred += 1
	return true_pred, false_pred 




def test(model, reads, label, batch_size, cut, length,
			  patches, seq_length, stride, patch_size, log, device):
	model.to(device)
	model.eval()
	with torch.no_grad():
		rejected_reads, accepted_reads, batch_count = 0, 0, 0
		true_pred, false_pred = 0, 0
		inputs = []

		start_time = time.time()
		for read in reads:
			if len(read) < cut + length:
				rejected_reads += 1
				continue
			accepted_reads += 1
			read = modified_zscore(read[cut:cut+length])
			#read = sliding_window_normalization(read[cut:cut+length])
			#read = extract_time_freq_features_real(read)
			if patches:
				read = cut_patchs(read, seq_length, stride, patch_size)
			inputs.append(read)

			if accepted_reads % batch_size == 0 and accepted_reads != 0:
				batch_count += 1
				t, f = inference(inputs, model, label, device)
				true_pred += t
				false_pred += f
				inputs = []

		if len(inputs) > 0:
			batch_count += 1
			t, f = inference(inputs, model, label, device)
			true_pred += t
			false_pred += f
			inputs = []

		if label == 1:
			myprint('accepted pos reads: {}, rejected pos reads: {}, TP: {}, FN: {}'.format(
				accepted_reads, rejected_reads, true_pred, false_pred), log)
		else:
			myprint('accepted neg reads: {}, rejected neg reads: {}, TN: {}, FP: {}'.format(
				accepted_reads, rejected_reads, true_pred, false_pred), log)
		total_time = time.time() - start_time
	return true_pred, false_pred, total_time / batch_count



if __name__ == '__main__':
	# Get command arguments
	parser = argparse.ArgumentParser(description="Test model")
	parser.add_argument("--pos_data_folder", '-p', type=str, required=True, help="Path to the positive dataset folder that contains train, valid, test files (.npy)")
	parser.add_argument("--neg_data_folder", '-n', type=str, required=True, help="Path to the negative dataset folder that contains train, valid, test files (.npy)")
	parser.add_argument("--model_state", '-ms', type=str, required=True, help="Path of the model (a pth file)")
	parser.add_argument("--output", '-o', type=str, required=True, help="The output path")
	parser.add_argument("--batch_size", '-b', type=int, default=512, help="Batch size, default 512")
	parser.add_argument("--cut", '-c', type=int, default=0, help="Electrical signal length to be cut, default 0")
	parser.add_argument("--length", '-len', type=int, default=3000, help="The length of each signal segment, default 3000")
	parser.add_argument("--patches", '-patches', action='store_true', help="Convert electrical signals into patches, default False")
	parser.add_argument("--seq_length", '-sl', type=int, default=299, help="Sequence length after patch, default 299")
	parser.add_argument("--stride", '-s', type=int, default=10, help="Patch step size, default 10")
	parser.add_argument("--patch_size", '-ps', type=int, default=16, help="The size of patch, default 16")
	parser.add_argument("--gpu_ids", '-g', type=str, default=None, help="Specify the GPU to use, if not specified, use all GPUs or CPU, default None")
	args = parser.parse_args()

	# Create output folder
	if not os.path.exists(args.output):
		os.makedirs(args.output)
	log = open(os.path.join(args.output, 'test.txt'), mode='w', encoding='utf-8')

	# Print parameter information
	for arg in vars(args):
		myprint(f"{arg}: {getattr(args, arg)}", log)

	# Load model

	model = ReadCurrentTargetRegion(
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
	# model = ReadCurrentTimeFreqFusionGated([32, 64, 128, 256, 512], n_fc_neurons=1024, n_classes=2, depth=29, shortcut=True, fusion_dim=256,dropout=0.25,gate_hidden=128)
	# model = LSTM(args.patch_size, 512, 2, True)
	# model = CNN_LSTM(512, 2, True)
	# model = Transformer(args.patch_size, args.seq_length, 512, 2048, 8, 6, 0.1, use_bias=True)
	# model = CNN_Transformer(512, 2048, 8, 6, 0.1, use_bias=True)

	# Use GPU or CPU
	if args.gpu_ids:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = nn.DataParallel(model).to(device)
	myprint(f"test in {device} {args.gpu_ids}", log)

	# Load model state
	model.load_state_dict(torch.load(args.model_state))
	myprint(f"Load model state from {args.model_state}", log)

	# Load dataset and testing
	reads = np.load(os.path.join(args.pos_data_folder, "test.npy"), allow_pickle=True)
	myprint(f"Load positive test data from {os.path.join(args.pos_data_folder, 'test.npy')}, shape: {reads.shape}", log)
	tp, fn, pos_infer_time = test(model, reads, 1, args.batch_size, args.cut, args.length,
		args.patches, args.seq_length, args.stride, args.patch_size, log, device)
	
	reads = np.load(os.path.join(args.neg_data_folder, "test.npy"), allow_pickle=True)
	myprint(f"Load negative test data from {os.path.join(args.neg_data_folder, 'test.npy')}, shape: {reads.shape}", log)
	tn, fp, neg_infer_time = test(model, reads, 0, args.batch_size, args.cut, args.length,
		args.patches, args.seq_length, args.stride, args.patch_size, log, device)

	# Calculate evaluation index values
	accuracy = round((tp + tn) * 100 / (tp + tn + fp + fn), 2)
	precision = round(tp * 100 / (tp + fp), 2)
	recall = round(tp * 100 / (tp + fn), 2)
	f1_score = round((2 * precision * recall) / (precision + recall), 2)
	aver_infer_time = round((pos_infer_time + neg_infer_time) / 2, 4)
	myprint(f"accuracy: {accuracy}, precision: {precision}, recall: {recall}, f1_score: {f1_score}, average inference time: {aver_infer_time}", log)
	log.close()
