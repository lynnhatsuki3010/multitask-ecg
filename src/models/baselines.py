"""
Published-architecture baselines, retrained under this project's exact protocol.

Review-2 MC3 asks for at least one competing architecture retrained on the same
13 labels and the same folds, because every published number quoted in the paper
so far comes from a different label set on a different split and cannot support
a "better than" claim.

Baselines are implemented as *backbones*: each returns the same
{"global_features": (B, D), "sequence_features": (B, T, D)} contract as
src/models/backbones.py, so the multitask heads, loss, sampler, augmentation,
calibration and metrics stay identical to the proposed model. The comparison
then isolates the representation learner, which is the only thing a backbone
swap should change.

Every class below is ported from the authors' released code, not written from a
paper description. Deviations forced by the task change are listed per class and
must be reported in the paper - an unlabelled adaptation presented under the
original name is exactly the unverifiable citation MC1 penalised.
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn


class SELayer(nn.Module):
    """Squeeze-and-excitation, ported verbatim from ECGTransForm's models.py."""

    def __init__(self, channel: int, reduction: int = 4) -> None:
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    """Residual block with SE gating - the "channel recalibration module" (CRM).

    Ported verbatim, including a detail that looks like a typo but is what the
    published model actually runs: the source writes
    `nn.Conv1d(inplanes, planes, stride)`, and Conv1d's third positional
    argument is `kernel_size`, not `stride`. With the default stride=1 both
    convolutions in this block are therefore 1x1. Reproduced as-is - "fixing" it
    would benchmark a model the authors never published.
    """

    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1,
                 downsample: nn.Module | None = None, reduction: int = 4) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)   # kernel_size=stride, see docstring
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.se(self.bn2(self.conv2(out)))
        if self.downsample is not None:
            residual = self.downsample(x)
        return self.relu(out + residual)


class ECGTransFormBackbone(nn.Module):
    """ECGTransForm, ported from the authors' released implementation.

    Source
    ------
    Ahmed et al., "ECGTransForm: Empowering adaptive ECG arrhythmia
    classification framework with bidirectional transformer", Biomedical Signal
    Processing and Control, 2023.
    Code: https://github.com/emadeldeen24/ECGTransForm (models.py)

    Chosen as the MC3 baseline because it is published in the target journal and
    each of its three components is a direct alternative to a component of the
    proposed model.

    Structure, following the source exactly
    ---------------------------------------
    1. Three parallel convolutions, kernels 5/9/11, all stride `stride`, each
       mapping input_channels -> mid_channels. Their outputs are **averaged**
       (`torch.mean(torch.stack(...), 2)`), not concatenated, so the block's
       width stays mid_channels.
    2. BN -> ReLU -> MaxPool(2, 2, padding=1) -> Dropout.
    3. conv_block2: Conv1d(mid, mid*2, k=8, padding=4) -> BN -> ReLU -> MaxPool.
    4. conv_block3: Conv1d(mid*2, final_out, k=8, padding=4) -> BN -> ReLU -> MaxPool.
    5. CRM: three stacked SEBasicBlocks at 128 channels.
    6. Bidirectional transformer: a standard 3-layer nn.TransformerEncoder with
       batch_first=True, applied twice with **shared weights** - once to the
       tensor and once to it flipped along the last axis - and summed. Note the
       consequence of batch_first on a (B, C, T) tensor: the 128 channels are
       the sequence and the time axis is the embedding, so `trans_dim` is a
       *time* length and `num_heads` must divide it.
    7. AdaptiveAvgPool1d(1) over the time axis -> (B, 128) feature vector, which
       is what the source feeds to its classifier.

    Authors' hyperparameters (configs/data_configs.py, MIT and PTB alike):
    mid_channels 32, final_out_channels 128, trans_dim 25, num_heads 5,
    stride 1, dropout 0.2, filter_sizes [5, 9, 11], 3 CRM blocks, 3 transformer
    layers, feature_dim 128.

    Deviations forced by this task - REPORT THESE
    ---------------------------------------------
    1. `input_channels` is 12 here, 1 in the source (it classifies single-lead
       segmented beats).
    2. Input length is 5000 (10 s at 500 Hz) against the source's 186/188-sample
       beats. Length after the three pooling stages is floor(L/2)+1 applied
       three times, and that value *is* `trans_dim`, so `num_heads` must divide
       it. At stride 1 a 5000-sample input gives trans_dim 626, whose only
       useful factor is 2. `stride` is therefore a config knob: stride 10 gives
       trans_dim 64 and allows nhead 8. Set it explicitly and report it.
    3. The classifier is replaced by this project's multitask heads (13 labels
       over three task groups). `global_features` here is exactly the vector the
       source sends to its own classifier, so the swap happens at the same point.
    4. `width_mult` scales mid/final channels. Leave it at 1.0 for the faithful
       model; a >1 run answers "is the baseline just smaller?" and should be
       reported as a separate row, never as "ECGTransForm".

    Points 1-3 are unavoidable consequences of the different task. Call this
    "ECGTransForm retrained under our protocol", not a reproduction of the
    published numbers.
    """

    def __init__(
        self,
        num_leads: int = 12,
        mid_channels: int = 32,
        final_out_channels: int = 128,
        stride: int = 10,
        num_heads: int = 8,
        num_transformer_layers: int = 3,
        num_crm_blocks: int = 3,
        dropout: float = 0.2,
        input_length: int = 5000,
        width_mult: float = 1.0,
    ) -> None:
        super().__init__()
        mid_channels = int(mid_channels * width_mult)
        final_out_channels = int(final_out_channels * width_mult)

        filter_sizes = [5, 9, 11]
        self.conv1 = nn.Conv1d(num_leads, mid_channels, kernel_size=filter_sizes[0],
                               stride=stride, bias=False, padding=filter_sizes[0] // 2)
        self.conv2 = nn.Conv1d(num_leads, mid_channels, kernel_size=filter_sizes[1],
                               stride=stride, bias=False, padding=filter_sizes[1] // 2)
        self.conv3 = nn.Conv1d(num_leads, mid_channels, kernel_size=filter_sizes[2],
                               stride=stride, bias=False, padding=filter_sizes[2] // 2)

        self.bn = nn.BatchNorm1d(mid_channels)
        self.relu = nn.ReLU()
        self.mp = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.do = nn.Dropout(dropout)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels * 2, kernel_size=8, stride=1,
                      bias=False, padding=4),
            nn.BatchNorm1d(mid_channels * 2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv1d(mid_channels * 2, final_out_channels, kernel_size=8, stride=1,
                      bias=False, padding=4),
            nn.BatchNorm1d(final_out_channels),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
        )

        self.inplanes = final_out_channels
        self.crm = self._make_layer(SEBasicBlock, final_out_channels, num_crm_blocks)

        trans_dim = self._trans_dim(input_length, stride)
        if trans_dim % num_heads != 0:
            raise ValueError(
                f"num_heads={num_heads} must divide trans_dim={trans_dim}. "
                f"trans_dim is the time length after three poolings and is set by "
                f"input_length={input_length} and stride={stride}; change stride "
                f"(10 gives 64 for a 5000-sample input) or num_heads."
            )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=trans_dim, nhead=num_heads, batch_first=True)
        # One encoder, applied to both directions - weights are shared in the source.
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_transformer_layers)

        self.trans_dim = trans_dim
        self.output_dim = final_out_channels

    @staticmethod
    def _trans_dim(input_length: int, stride: int) -> int:
        """Time length reaching the transformer, i.e. the encoder's d_model.

        Stem conv (k=5, pad=2, stride s), then three times: MaxPool(2, 2,
        padding=1) which maps L -> L//2 + 1. The two later conv blocks use k=8
        with padding=4, which *lengthens* the signal by one sample each - easy
        to miss, and the reason this is computed rather than assumed.

        Sanity check against the source: input_length=186, stride=1 returns 25,
        the `trans_dim` in configs/data_configs.py.
        """
        length = (input_length + 2 * (5 // 2) - 5) // stride + 1
        length = length // 2 + 1                     # stem MaxPool
        for _ in range(2):
            length = length + 1                      # conv, k=8 pad=4
            length = length // 2 + 1                 # MaxPool
        return length

    def _make_layer(self, block, planes: int, blocks: int, stride: int = 1) -> nn.Module:
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.conv1(x_in)
        x2 = self.conv2(x_in)
        x3 = self.conv3(x_in)

        x_concat = torch.mean(torch.stack([x1, x2, x3], 2), 2)
        x_concat = self.do(self.mp(self.relu(self.bn(x_concat))))

        x = self.conv_block2(x_concat)
        x = self.conv_block3(x)
        x = self.crm(x)                                   # (B, C, T)

        forward_pass = self.transformer_encoder(x)
        backward_pass = self.transformer_encoder(torch.flip(x, [2]))
        x = forward_pass + backward_pass                  # (B, C, T)

        # (B, C, T) -> (B, T, C): time becomes the sequence and channels the
        # feature width, which is the layout this project's heads expect. The
        # mean over T is exactly the AdaptiveAvgPool1d(1) the source feeds to
        # its classifier, so the head swap happens at the same tensor.
        seq = x.transpose(1, 2)
        return {
            "global_features": seq.mean(dim=1),
            "sequence_features": seq,
        }


# ═════════════════════════════════════════════════════════════════════════════
# PTB-XL benchmark baselines (Strodthoff et al., IEEE JBHI 2021)
# ═════════════════════════════════════════════════════════════════════════════
#
# xresnet1d101 and inception1d are the two strongest models of the PTB-XL
# benchmark (github.com/helme/ecg_ptbxl_benchmarking). That repository is
# GPL-3.0, so nothing below is copied from it: the classes are written for this
# project from the architecture its code specifies (layer counts, widths, kernel
# sizes, stem, shortcut and initialisation rules), which a reader can check
# against code/models/xresnet1d.py and code/models/inception1d.py there.
#
# Deviations forced by this project's protocol - REPORT THESE
# -----------------------------------------------------------
# 1. Input. The benchmark trains on random 2.5 s crops at 100 Hz and averages
#    sliding-window predictions at test time. Here the input is the full 10 s at
#    500 Hz, average-pooled by `input_downsample` (default 5, i.e. to 100 Hz) so
#    every kernel keeps the receptive field in seconds that it was designed for.
# 2. Head. The benchmark's head is concat pooling (avg + max) -> BN -> dropout ->
#    Linear(128) -> ReLU -> BN -> dropout -> Linear. Here the backbone stops at the
#    last feature map and this project's multitask heads take over, with the
#    "plain" MI head so that no anatomical lead grouping is added to the baseline.
# 3. Optimisation. The benchmark uses fastai's one-cycle schedule (lr 1e-2, 50
#    epochs, BCE). Here the baselines get their own epoch and learning-rate budget
#    under AdamW + cosine and the shared loss; see the CV protocol document.


class _ConvBNAct(nn.Sequential):
    """Conv1d -> BatchNorm1d (-> ReLU). BN starts at weight 1 (or 0) and bias 1e-3."""

    def __init__(self, ni: int, nf: int, ks: int, stride: int = 1, act: bool = True,
                 zero_bn: bool = False) -> None:
        conv = nn.Conv1d(ni, nf, ks, stride=stride, padding=(ks - 1) // 2, bias=False)
        nn.init.kaiming_normal_(conv.weight)
        bn = nn.BatchNorm1d(nf)
        nn.init.constant_(bn.weight, 0.0 if zero_bn else 1.0)
        nn.init.constant_(bn.bias, 1e-3)
        super().__init__(conv, bn, *([nn.ReLU(inplace=True)] if act else []))


class _XBottleneck(nn.Module):
    """Bottleneck residual block: 1x1 -> k (stride) -> 1x1 x4, zero-initialised last BN.

    The shortcut average-pools first when the block downsamples, then matches
    channels with a 1x1 conv + BN - the "ResNet-D" shortcut of the xresnet family.
    """

    def __init__(self, ni: int, width: int, stride: int, ks: int, expansion: int = 4) -> None:
        super().__init__()
        nf = width * expansion
        self.convs = nn.Sequential(
            _ConvBNAct(ni, width, 1),
            _ConvBNAct(width, width, ks, stride=stride),
            _ConvBNAct(width, nf, 1, act=False, zero_bn=True),
        )
        short = []
        if stride != 1:
            short.append(nn.AvgPool1d(2, ceil_mode=True))
        if ni != nf:
            short.append(_ConvBNAct(ni, nf, 1, act=False))
        self.shortcut = nn.Sequential(*short)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.convs(x) + self.shortcut(x))


class XResNet1d101Backbone(nn.Module):
    """xresnet1d101 as configured in the PTB-XL benchmark.

    Stem: three conv-BN-ReLU layers (12 -> 32 -> 32 -> 64, kernel 5, first stride 2)
    and max-pool 3/2. Four stages of bottleneck blocks, [3, 4, 23, 3], all of width
    64 with expansion 4 (the benchmark narrows the vision widths 64-128-256-512 to
    64 throughout), kernel 5, stride 1 in the first stage and 2 after. Output 256
    channels.
    """

    def __init__(self, num_leads: int = 12, input_downsample: int = 5, kernel_size: int = 5,
                 layers=(3, 4, 23, 3), width: int = 64, expansion: int = 4) -> None:
        super().__init__()
        self.down = nn.AvgPool1d(input_downsample) if input_downsample > 1 else nn.Identity()
        self.stem = nn.Sequential(
            _ConvBNAct(num_leads, 32, kernel_size, stride=2),
            _ConvBNAct(32, 32, kernel_size),
            _ConvBNAct(32, 64, kernel_size),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        blocks, ni = [], 64
        for i, n in enumerate(layers):
            for j in range(n):
                blocks.append(_XBottleneck(ni, width, stride=(2 if (i > 0 and j == 0) else 1),
                                           ks=kernel_size, expansion=expansion))
                ni = width * expansion
        self.blocks = nn.Sequential(*blocks)
        self.output_dim = ni

    def forward(self, x_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        seq = self.blocks(self.stem(self.down(x_in))).transpose(1, 2)    # (B, T, C)
        return {"global_features": seq.mean(dim=1), "sequence_features": seq}


class _InceptionBlock(nn.Module):
    """Bottleneck 1x1 -> three parallel convs (kernels kss) + maxpool/1x1 branch -> BN-ReLU."""

    def __init__(self, ni: int, nb_filters: int, kss, bottleneck: int) -> None:
        super().__init__()
        self.bottleneck = nn.Conv1d(ni, bottleneck, 1, bias=False)
        self.convs = nn.ModuleList(
            [nn.Conv1d(bottleneck, nb_filters, k, padding=(k - 1) // 2, bias=False) for k in kss])
        self.pool_branch = nn.Sequential(nn.MaxPool1d(3, stride=1, padding=1),
                                         nn.Conv1d(ni, nb_filters, 1, bias=False))
        self.bn_relu = nn.Sequential(nn.BatchNorm1d(nb_filters * (len(kss) + 1)), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = self.bottleneck(x)
        return self.bn_relu(torch.cat([c(b) for c in self.convs] + [self.pool_branch(x)], dim=1))


class Inception1dBackbone(nn.Module):
    """inception1d (InceptionTime) as configured in the PTB-XL benchmark.

    Six inception blocks, kernels 39/19/9 (from kernel_size 40), bottleneck 32, 32
    filters per branch (128 channels out), with a 1x1 conv + BN residual shortcut
    added after every third block. No temporal downsampling inside the network.
    """

    def __init__(self, num_leads: int = 12, input_downsample: int = 5, kernel_size: int = 40,
                 depth: int = 6, bottleneck: int = 32, nb_filters: int = 32) -> None:
        super().__init__()
        assert depth % 3 == 0
        kss = [k - 1 if k % 2 == 0 else k for k in (kernel_size, kernel_size // 2, kernel_size // 4)]
        out = nb_filters * (len(kss) + 1)
        self.down = nn.AvgPool1d(input_downsample) if input_downsample > 1 else nn.Identity()
        self.blocks = nn.ModuleList(
            [_InceptionBlock(num_leads if d == 0 else out, nb_filters, kss, bottleneck) for d in range(depth)])
        self.shortcuts = nn.ModuleList(
            [nn.Sequential(nn.Conv1d(num_leads if d == 0 else out, out, 1, bias=False), nn.BatchNorm1d(out))
             for d in range(depth // 3)])
        self.act = nn.ReLU()
        self.output_dim = out

    def forward(self, x_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.down(x_in)
        res = x
        for d, block in enumerate(self.blocks):
            x = block(x)
            if d % 3 == 2:
                x = self.act(x + self.shortcuts[d // 3](res))
                res = x
        seq = x.transpose(1, 2)                                            # (B, T, C)
        return {"global_features": seq.mean(dim=1), "sequence_features": seq}
