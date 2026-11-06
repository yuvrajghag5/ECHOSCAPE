"""
Three segmentation networks, written from scratch, compared on the same data.

    unet        U-Net         (2015)  convolutional encoder-decoder with skips
    deeplab     DeepLabV3+    (2018)  dilated convolutions + multi-scale pooling
    segformer   SegFormer     (2021)  transformer encoder + all-MLP decoder

Three different families, deliberately: the comparison asks which way of
"seeing" a radar tile works best, not which of three similar nets wins. All
three are sized to roughly the same number of parameters (~2 M) so that the
difference between them is the architecture and not its size.

None is imported from a library, and none starts from pretrained weights. This
is a Deep Learning course and the architectures are what is assessed - and
pretrained weights were learned on photographs, which radar is not.

Every network has the same contract:

    input   (B, 3, 128, 128)   VV, VH, VV-VH, scaled to 0-1
    output  (B, 10, 128, 128)  one score per class, including 0 = unlabelled

Class 0 is predicted but never trained on - the loss ignores it. Keeping the
channel means the class ids in the output line up with config.CLASSES, so
nothing has to be renumbered on the way to a map.

The prototype builds a network with build_model(name, **kwargs), using the
name and kwargs saved inside the checkpoint, so it always rebuilds exactly the
network that was trained.

    .venv\\Scripts\\python.exe training\\model.py        # shapes and sizes of all three
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================== U-Net (2015)
# Layers unchanged since the first trained run. Only the head grew, from 7 to
# 10 outputs, when the three merged-away Dynamic World classes came back - so
# the archived 6-class weights no longer load, by design.

class Block(nn.Module):
    """Two 3x3 convolutions, each followed by batch-norm and a ReLU.

    Batch-norm matters more than usual here: radar in decibels has a very
    different spread per channel, and without it the first layers spend their
    time undoing that instead of learning.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNet(nn.Module):
    """Four down, four up, with skip connections.

    The skips are the point of a U-Net. Pooling four times takes a 128-pixel
    tile down to 8 pixels, which is what lets the model see a whole landscape
    at once - but it also throws away exactly where each boundary was. The
    skip connections hand that detail back on the way up, so the output can be
    both context-aware and sharp at the edges.

    base 16 = 1.94 M parameters (the size design.md specifies); base 32 = 7.76 M,
    which overfitted within six epochs on the hand labels.
    """

    def __init__(self, in_channels=3, classes=10, base=32):
        super().__init__()
        self.kwargs = {"in_channels": in_channels, "classes": classes, "base": base}
        widths = [base, base * 2, base * 4, base * 8]

        self.downs = nn.ModuleList()
        channels = in_channels
        for width in widths:
            self.downs.append(Block(channels, width))
            channels = width
        self.pool = nn.MaxPool2d(2)

        self.middle = Block(widths[-1], widths[-1] * 2)

        self.ups = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        channels = widths[-1] * 2
        for width in reversed(widths):
            self.ups.append(nn.ConvTranspose2d(channels, width, 2, stride=2))
            # width * 2 because the skip connection is concatenated on
            self.up_blocks.append(Block(width * 2, width))
            channels = width

        self.head = nn.Conv2d(base, classes, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.middle(x)

        for up, block, skip in zip(self.ups, self.up_blocks, reversed(skips)):
            x = up(x)
            x = torch.cat([skip, x], dim=1)
            x = block(x)

        return self.head(x)


# ========================================================== DeepLabV3+ (2018)

def conv_bn_relu(in_channels, out_channels, kernel=3, stride=1, dilation=1):
    """A convolution, batch-norm and ReLU - the unit everything below is built from."""
    padding = dilation * (kernel // 2)
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, stride=stride,
                  padding=padding, dilation=dilation, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ResBlock(nn.Module):
    """A residual block: two 3x3 convolutions plus a shortcut around them.

    The shortcut lets the block learn a *correction* to its input rather than
    a whole new picture, which is what makes deeper encoders trainable.
    `dilation` spreads the 3x3 kernel's taps apart, so the block sees further
    without shrinking the image - DeepLab's central idea.
    """

    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.main = nn.Sequential(
            conv_bn_relu(in_channels, out_channels, 3, stride, dilation),
            nn.Conv2d(out_channels, out_channels, 3, padding=dilation,
                      dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.shortcut = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        return F.relu(self.main(x) + self.shortcut(x))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling: look at the same place at several scales.

    Parallel branches see the feature map through 3x3 kernels dilated by
    different amounts - a small neighbourhood, a wider one, a wider still -
    plus one branch that averages the whole tile. Their outputs are stacked
    and mixed. A field, the farm around it and the region around that are all
    in view at once.

    The rates are small on purpose. DeepLab's published rates (6, 12, 18) were
    set for 512-pixel images, where the map at this depth is 32x32. Our tiles
    are 128 pixels, so the map here is 8x8: a rate of 12 would only sample the
    zero padding beyond the edge.
    """

    def __init__(self, in_channels, out_channels, rates):
        super().__init__()
        self.branches = nn.ModuleList([conv_bn_relu(in_channels, out_channels, 1)])
        for rate in rates:
            self.branches.append(conv_bn_relu(in_channels, out_channels, 3, dilation=rate))
        # Whole-tile average. No batch-norm here: after pooling there is one
        # value per channel, and batch-norm on a single value is undefined
        # when a batch holds one tile (as it does in the prototype).
        self.pooled = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.ReLU(inplace=True),
        )
        self.project = nn.Sequential(
            conv_bn_relu(out_channels * (len(rates) + 2), out_channels, 1),
            nn.Dropout(0.1),
        )

    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        pooled = F.interpolate(self.pooled(x), size=x.shape[-2:],
                               mode="bilinear", align_corners=False)
        return self.project(torch.cat(outputs + [pooled], dim=1))


class DeepLabV3Plus(nn.Module):
    """DeepLabV3+: a residual encoder, ASPP on top, and a light decoder.

    Where the U-Net shrinks the tile four times and rebuilds it step by step,
    DeepLab shrinks it to 1/16 and then STOPS shrinking: the last stage keeps
    the 8x8 size and uses dilated convolutions to keep widening its view.
    ASPP then reads that map at several scales. The decoder upsamples once to
    1/4, mixes in one early, sharp feature map (the "+" in V3+), and upsamples
    to full size.

        128 -stem-> 64 -layer1-> 32 (kept for the decoder)
                    -layer2-> 16 -layer3-> 8 -layer4 (dilated)-> 8 -ASPP-> 8
        8 -> 32, joined with layer1 -> decoder -> 128
    """

    def __init__(self, in_channels=3, classes=10, base=24, aspp_channels=96,
                 low_channels=24, rates=(2, 4, 6)):
        super().__init__()
        self.kwargs = {"in_channels": in_channels, "classes": classes, "base": base,
                       "aspp_channels": aspp_channels, "low_channels": low_channels,
                       "rates": tuple(rates)}
        w = [base, base * 2, base * 4, base * 8]

        self.stem = nn.Sequential(conv_bn_relu(in_channels, w[0], 3, stride=2),
                                  conv_bn_relu(w[0], w[0]))                      # 1/2
        self.layer1 = nn.Sequential(ResBlock(w[0], w[0], stride=2),
                                    ResBlock(w[0], w[0]))                        # 1/4
        self.layer2 = nn.Sequential(ResBlock(w[0], w[1], stride=2),
                                    ResBlock(w[1], w[1]))                        # 1/8
        self.layer3 = nn.Sequential(ResBlock(w[1], w[2], stride=2),
                                    ResBlock(w[2], w[2]))                        # 1/16
        self.layer4 = nn.Sequential(ResBlock(w[2], w[3], dilation=2),
                                    ResBlock(w[3], w[3], dilation=2))            # 1/16, wider view

        self.aspp = ASPP(w[3], aspp_channels, rates)
        self.low = conv_bn_relu(w[0], low_channels, 1)
        self.decoder = nn.Sequential(conv_bn_relu(aspp_channels + low_channels, aspp_channels),
                                     conv_bn_relu(aspp_channels, aspp_channels))
        self.head = nn.Conv2d(aspp_channels, classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        x = self.stem(x)
        low = self.layer1(x)
        x = self.layer4(self.layer3(self.layer2(low)))
        x = self.aspp(x)
        x = F.interpolate(x, size=low.shape[-2:], mode="bilinear", align_corners=False)
        x = self.decoder(torch.cat([x, self.low(low)], dim=1))
        return F.interpolate(self.head(x), size=size, mode="bilinear", align_corners=False)


# ============================================================ SegFormer (2021)

class PatchEmbed(nn.Module):
    """Turn an image (or feature map) into a sequence of tokens.

    A strided convolution cuts the map into patches; each patch becomes one
    token, a vector the transformer can attend between. "Overlapping": the
    kernel is wider than the stride, so neighbouring patches share pixels and
    no boundary falls exactly on a patch edge.
    """

    def __init__(self, in_channels, dim, kernel, stride):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, dim, kernel, stride=stride, padding=kernel // 2)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.proj(x)
        height, width = x.shape[-2:]
        return self.norm(x.flatten(2).transpose(1, 2)), height, width


class EfficientAttention(nn.Module):
    """Self-attention: every token looks at every other and decides which matter.

    Plain attention compares all N tokens with all N - at the first stage that
    is 1,024 x 1,024. SegFormer shrinks the keys and values first (`reduction`
    pools them by that factor on each side), so each token is compared with a
    coarse summary of the map instead: 1,024 x 16 at the first stage.
    """

    def __init__(self, dim, heads, reduction):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim * 2)
        self.proj = nn.Linear(dim, dim)
        self.reduction = reduction
        if reduction > 1:
            self.shrink = nn.Conv2d(dim, dim, reduction, stride=reduction)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, height, width):
        batch, tokens, dim = x.shape
        per_head = dim // self.heads
        q = self.q(x).reshape(batch, tokens, self.heads, per_head).transpose(1, 2)

        source = x
        if self.reduction > 1:
            grid = x.transpose(1, 2).reshape(batch, dim, height, width)
            source = self.norm(self.shrink(grid).flatten(2).transpose(1, 2))
        kv = self.kv(source).reshape(batch, -1, 2, self.heads, per_head).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        weights = (q @ k.transpose(-2, -1)) * self.scale
        out = weights.softmax(dim=-1) @ v
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, dim))


class MixFFN(nn.Module):
    """The feed-forward part of a transformer block, with a 3x3 convolution inside.

    Transformers have no built-in sense of where a token is. Most add a fixed
    position code, which breaks when the image size changes. SegFormer instead
    puts a small depthwise convolution in here: each token mixes with its
    immediate neighbours, which is enough for the network to know position.
    """

    def __init__(self, dim, ratio):
        super().__init__()
        hidden = dim * ratio
        self.fc1 = nn.Linear(dim, hidden)
        self.local = nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x, height, width):
        x = self.fc1(x)
        batch, tokens, hidden = x.shape
        x = self.local(x.transpose(1, 2).reshape(batch, hidden, height, width))
        return self.fc2(F.gelu(x.flatten(2).transpose(1, 2)))


class TransformerBlock(nn.Module):
    """Attention then feed-forward, each wrapped in a layer-norm and a shortcut."""

    def __init__(self, dim, heads, reduction, ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attention = EfficientAttention(dim, heads, reduction)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = MixFFN(dim, ratio)

    def forward(self, x, height, width):
        x = x + self.attention(self.norm1(x), height, width)
        return x + self.ffn(self.norm2(x), height, width)


class SegFormer(nn.Module):
    """SegFormer: a four-stage transformer encoder and an all-MLP decoder.

    Convolutions only ever see a fixed neighbourhood; attention lets any part
    of the tile influence any other from the first stage on. The encoder makes
    four maps at 1/4, 1/8, 1/16 and 1/32 of the tile. The decoder is nothing
    but 1x1 convolutions: bring every map to the same width and to the 1/4
    size, stack them, mix, classify, upsample.

    Shaped like the published MiT-B0, the smallest SegFormer, slimmed in the
    last two stages to sit near the other two networks' size.

        128 -> 32x32 tokens -> 16x16 -> 8x8 -> 4x4
        all four -> 32x32 -> mix -> 128
    """

    def __init__(self, in_channels=3, classes=10, dims=(32, 64, 128, 192),
                 depths=(2, 2, 2, 2), heads=(1, 2, 4, 6), reductions=(8, 4, 2, 1),
                 ratio=4, decoder_dim=128):
        super().__init__()
        self.kwargs = {"in_channels": in_channels, "classes": classes, "dims": tuple(dims),
                       "depths": tuple(depths), "heads": tuple(heads),
                       "reductions": tuple(reductions), "ratio": ratio,
                       "decoder_dim": decoder_dim}

        self.embeds, self.stages, self.norms = nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        channels = in_channels
        for i, dim in enumerate(dims):
            kernel, stride = (7, 4) if i == 0 else (3, 2)
            self.embeds.append(PatchEmbed(channels, dim, kernel, stride))
            self.stages.append(nn.ModuleList(
                TransformerBlock(dim, heads[i], reductions[i], ratio) for _ in range(depths[i])))
            self.norms.append(nn.LayerNorm(dim))
            channels = dim

        self.lateral = nn.ModuleList(nn.Conv2d(dim, decoder_dim, 1) for dim in dims)
        self.fuse = nn.Sequential(
            nn.Conv2d(decoder_dim * len(dims), decoder_dim, 1, bias=False),
            nn.BatchNorm2d(decoder_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )
        self.head = nn.Conv2d(decoder_dim, classes, 1)

    def forward(self, x):
        size = x.shape[-2:]
        maps = []
        for embed, blocks, norm in zip(self.embeds, self.stages, self.norms):
            tokens, height, width = embed(x)
            for block in blocks:
                tokens = block(tokens, height, width)
            x = norm(tokens).transpose(1, 2).reshape(tokens.shape[0], -1, height, width)
            maps.append(x)

        target = maps[0].shape[-2:]
        stacked = [F.interpolate(lateral(m), size=target, mode="bilinear", align_corners=False)
                   for lateral, m in zip(self.lateral, maps)]
        x = self.head(self.fuse(torch.cat(stacked, dim=1)))
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)


# ================================================================ one way in

MODELS = {"unet": UNet, "deeplab": DeepLabV3Plus, "segformer": SegFormer}

# The settings each network is trained with. unet keeps base 16, the size
# design.md specifies and the run already done; the other two were sized to
# land near it (see the table printed by running this file).
DEFAULTS = {"unet": {"base": 16}, "deeplab": {}, "segformer": {}}


def build_model(name, **overrides):
    """Build a network by name. Its exact settings end up in `net.kwargs`.

    Save `name` and `net.kwargs` beside the weights, and build_model(name,
    **kwargs) rebuilds exactly the network that was trained - which is what the
    prototype does.
    """
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {', '.join(MODELS)}")
    return MODELS[name](**{**DEFAULTS[name], **overrides})


def count_parameters(net):
    return sum(p.numel() for p in net.parameters())


if __name__ == "__main__":
    dummy = torch.zeros(2, 3, 128, 128)
    print(f"{'model':<11}{'parameters':>12}   output")
    for name in MODELS:
        net = build_model(name).eval()
        with torch.no_grad():
            shape = tuple(net(dummy).shape)
        print(f"{name:<11}{count_parameters(net):>12,}   {shape}")
