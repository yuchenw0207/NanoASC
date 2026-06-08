import torch
from torch import nn
import torch.nn.functional as F


class SEBlock1d(nn.Module):
    """
    Squeeze-and-Excitation for 1D feature maps
    x: [B, C, T]
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Conv1d(channels, hidden, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.fc(w)
        return x * w


class MultiScaleConv1d(nn.Module):
    """
    Multi-branch 1D convolution with different dilations.
    Output channels are split across branches, then concatenated.
    """
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 3,
        dilations=(1, 2, 4),
        stride: int = 1,
        bias: bool = False,
    ):
        super().__init__()
        self.dilations = tuple(dilations)
        n = len(self.dilations)

        base = out_ch // n
        rem = out_ch % n
        split = [base + (1 if i < rem else 0) for i in range(n)]

        branches = []
        for d, c_out in zip(self.dilations, split):
            pad = ((kernel_size - 1) // 2) * d
            branches.append(
                nn.Conv1d(
                    in_ch,
                    c_out,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=pad,
                    dilation=d,
                    bias=bias,
                )
            )
        self.branches = nn.ModuleList(branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outs = [conv(x) for conv in self.branches]
        return torch.cat(outs, dim=1)


class BasicConvResBlock(nn.Module):
    """
    Residual block:
    conv1 -> bn1 -> relu -> conv2(or multiscale) -> bn2 -> se -> add -> relu
    """
    def __init__(
        self,
        input_dim,
        n_filters,
        kernel_size=3,
        padding=1,
        stride=1,
        shortcut=False,
        downsample=None,
        se_reduction=None,
        ms_dilations=None,
        ms_kernel_size=3,
    ):
        super().__init__()
        self.downsample = downsample
        self.shortcut = shortcut

        self.conv1 = nn.Conv1d(
            input_dim,
            n_filters,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(n_filters)
        self.relu = nn.ReLU(inplace=True)

        if ms_dilations is None:
            self.conv2 = nn.Conv1d(
                n_filters,
                n_filters,
                kernel_size=kernel_size,
                padding=padding,
                stride=1,
                bias=False,
            )
        else:
            self.conv2 = MultiScaleConv1d(
                in_ch=n_filters,
                out_ch=n_filters,
                kernel_size=ms_kernel_size,
                dilations=tuple(ms_dilations),
                stride=1,
                bias=False,
            )

        self.bn2 = nn.BatchNorm1d(n_filters)
        self.se = nn.Identity() if se_reduction is None else SEBlock1d(n_filters, reduction=se_reduction)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # SE before residual addition
        out = self.se(out)

        if self.shortcut:
            if self.downsample is not None:
                residual = self.downsample(x)
            out = out + residual

        out = self.relu(out)
        return out


class AttentionPooling1d(nn.Module):
    """
    Learnable temporal attention pooling.
    Input:  [B, C, T]
    Output: [B, C]
    """
    def __init__(self, channels: int, attn_hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Conv1d(channels, attn_hidden, kernel_size=1, bias=True),
            nn.Tanh(),
            nn.Conv1d(attn_hidden, 1, kernel_size=1, bias=True),
        )

    def forward(self, x: torch.Tensor):
        # x: [B, C, T]
        score = self.attn(x)               # [B, 1, T]
        weight = torch.softmax(score, dim=-1)
        pooled = torch.sum(x * weight, dim=-1)  # [B, C]
        return pooled, weight


class NanoASC(nn.Module):
    """
    A task-oriented ReadCurrent variant for target-region adaptive sequencing.

    Main changes vs original-style ReadCurrent:
    1) gentler stem to preserve early waveform details
    2) SE + multi-scale residual blocks from stage2/3/4
    3) attention pooling instead of simple flattening after coarse pooling
    """
    @staticmethod
    def build_classifier_head(input_dim, n_fc_neurons, n_classes, dropout=0.2):
        return nn.Sequential(
            nn.Linear(input_dim, n_fc_neurons),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(n_fc_neurons, n_fc_neurons),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(n_fc_neurons, n_classes),
        )

    def __init__(
        self,
        n_conv_neurons,
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
        build_head=True,
    ):
        super().__init__()

        if len(n_conv_neurons) != 5:
            raise ValueError(f"n_conv_neurons must have length 5, got {len(n_conv_neurons)}")

        def _se(stage_id):
            return se_reduction if (se_reduction is not None and stage_id in set(se_stages)) else None

        def _ms(stage_id):
            return ms_dilations if (ms_dilations is not None and stage_id in set(ms_stages)) else None

        if depth == 9:
            n_conv_block_1, n_conv_block_2, n_conv_block_3, n_conv_block_4 = 1, 1, 1, 1
        elif depth == 17:
            n_conv_block_1, n_conv_block_2, n_conv_block_3, n_conv_block_4 = 2, 2, 2, 2
        elif depth == 29:
            n_conv_block_1, n_conv_block_2, n_conv_block_3, n_conv_block_4 = 5, 5, 2, 2
        elif depth == 49:
            n_conv_block_1, n_conv_block_2, n_conv_block_3, n_conv_block_4 = 8, 8, 5, 3
        else:
            raise ValueError(f"Unsupported depth: {depth}")

        layers = []

        # ------------------------------------------------------------------
        # Stem: gentler downsampling than the original stride=3 design
        # For nanopore 3000-pt signal, preserve early local details better
        # ------------------------------------------------------------------
        layers.append(nn.Conv1d(1, n_conv_neurons[0], kernel_size=19, stride=1, padding=9, bias=False))
        layers.append(nn.BatchNorm1d(n_conv_neurons[0]))
        layers.append(nn.ReLU(inplace=True))

        layers.append(nn.Conv1d(n_conv_neurons[0], n_conv_neurons[1], kernel_size=7, stride=2, padding=3, bias=False))
        layers.append(nn.BatchNorm1d(n_conv_neurons[1]))
        layers.append(nn.ReLU(inplace=True))

        # stage 1
        layers.append(
            BasicConvResBlock(
                input_dim=n_conv_neurons[1],
                n_filters=n_conv_neurons[1],
                kernel_size=3,
                padding=1,
                shortcut=shortcut,
                se_reduction=_se(1),
                ms_dilations=_ms(1),
                ms_kernel_size=ms_kernel_size,
            )
        )
        for _ in range(n_conv_block_1 - 1):
            layers.append(
                BasicConvResBlock(
                    input_dim=n_conv_neurons[1],
                    n_filters=n_conv_neurons[1],
                    kernel_size=3,
                    padding=1,
                    shortcut=shortcut,
                    se_reduction=_se(1),
                    ms_dilations=_ms(1),
                    ms_kernel_size=ms_kernel_size,
                )
            )
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        # stage 2
        ds = nn.Sequential(
            nn.Conv1d(n_conv_neurons[1], n_conv_neurons[2], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(n_conv_neurons[2]),
        )
        layers.append(
            BasicConvResBlock(
                input_dim=n_conv_neurons[1],
                n_filters=n_conv_neurons[2],
                kernel_size=3,
                padding=1,
                shortcut=shortcut,
                downsample=ds,
                se_reduction=_se(2),
                ms_dilations=_ms(2),
                ms_kernel_size=ms_kernel_size,
            )
        )
        for _ in range(n_conv_block_2 - 1):
            layers.append(
                BasicConvResBlock(
                    input_dim=n_conv_neurons[2],
                    n_filters=n_conv_neurons[2],
                    kernel_size=3,
                    padding=1,
                    shortcut=shortcut,
                    se_reduction=_se(2),
                    ms_dilations=_ms(2),
                    ms_kernel_size=ms_kernel_size,
                )
            )
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        # stage 3
        ds = nn.Sequential(
            nn.Conv1d(n_conv_neurons[2], n_conv_neurons[3], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(n_conv_neurons[3]),
        )
        layers.append(
            BasicConvResBlock(
                input_dim=n_conv_neurons[2],
                n_filters=n_conv_neurons[3],
                kernel_size=3,
                padding=1,
                shortcut=shortcut,
                downsample=ds,
                se_reduction=_se(3),
                ms_dilations=_ms(3),
                ms_kernel_size=ms_kernel_size,
            )
        )
        for _ in range(n_conv_block_3 - 1):
            layers.append(
                BasicConvResBlock(
                    input_dim=n_conv_neurons[3],
                    n_filters=n_conv_neurons[3],
                    kernel_size=3,
                    padding=1,
                    shortcut=shortcut,
                    se_reduction=_se(3),
                    ms_dilations=_ms(3),
                    ms_kernel_size=ms_kernel_size,
                )
            )
        layers.append(nn.MaxPool1d(kernel_size=3, stride=2, padding=1))

        # stage 4
        ds = nn.Sequential(
            nn.Conv1d(n_conv_neurons[3], n_conv_neurons[4], kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(n_conv_neurons[4]),
        )
        layers.append(
            BasicConvResBlock(
                input_dim=n_conv_neurons[3],
                n_filters=n_conv_neurons[4],
                kernel_size=3,
                padding=1,
                shortcut=shortcut,
                downsample=ds,
                se_reduction=_se(4),
                ms_dilations=_ms(4),
                ms_kernel_size=ms_kernel_size,
            )
        )
        for _ in range(n_conv_block_4 - 1):
            layers.append(
                BasicConvResBlock(
                    input_dim=n_conv_neurons[4],
                    n_filters=n_conv_neurons[4],
                    kernel_size=3,
                    padding=1,
                    shortcut=shortcut,
                    se_reduction=_se(4),
                    ms_dilations=_ms(4),
                    ms_kernel_size=ms_kernel_size,
                )
            )

        self.backbone = nn.Sequential(*layers)
        self.pool = AttentionPooling1d(n_conv_neurons[4], attn_hidden=attn_hidden)

        self.feature_dim = n_conv_neurons[4]
        self.build_head = build_head
        self.fc_layers = (
            self.build_classifier_head(self.feature_dim, n_fc_neurons, n_classes, dropout=head_dropout)
            if build_head else None
        )

        self.__init_weights()

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode="fan_in")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False, return_attention=False):
        """
        x:
          - [B, L]
          - [B, 1, L]

        return_features=True:
          return pooled embedding [B, C]

        return_attention=True:
          return logits/features and attention weights [B, 1, T]
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)
        elif x.dim() == 3 and x.size(1) == 1:
            pass
        else:
            raise ValueError(f"Expected (B,L) or (B,1,L), got {tuple(x.shape)}")

        feat_map = self.backbone(x)              # [B, C, T]
        pooled, attn = self.pool(feat_map)       # pooled: [B, C]

        if return_features or self.fc_layers is None:
            if return_attention:
                return pooled, attn
            return pooled

        out = self.fc_layers(pooled)
        if return_attention:
            return out, attn
        return out