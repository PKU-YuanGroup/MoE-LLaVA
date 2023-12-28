import re

import torch
from einops import rearrange
from timm.models.vision_transformer import Block
from torch import nn


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)

class BaseConv2D(nn.Module):
    def __init__(self, channels, groups=1, eps=1e-6):
        super().__init__()
        self.conv = nn.Sequential(
            nn.GroupNorm(num_groups=groups, num_channels=channels, eps=eps, affine=True),  # LayerNorm
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
        )
    def forward(self, x):
        h = w = int(x.shape[1]**0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = x + self.conv(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        return x


class SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_in_block, num_out_block, num_heads=32, mlp_ratio=2.6875, groups=32, eps=1e-6):
        super().__init__()
        self.proj_in = nn.Sequential(nn.Linear(in_channels, out_channels),
                                  nn.GELU(),
                                  nn.Linear(out_channels, out_channels))

        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.block_in = nn.Sequential(
            *([BaseConv2D(out_channels, groups, eps), Block(out_channels, num_heads, mlp_ratio)] * num_in_block)
        ) if num_in_block > 0 else nn.Identity()

        self.down2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=0) if num_out_block > 0 else nn.Identity()
        self.block_out = nn.Sequential(
            *([BaseConv2D(out_channels, groups, eps), Block(out_channels, num_heads, mlp_ratio)] * num_out_block)
        ) if num_out_block > 0 else nn.Identity()

        self.proj_out = nn.Sequential(nn.Linear(out_channels, out_channels),
                                  nn.GELU(),
                                  nn.Linear(out_channels, out_channels))

    def forward(self, x):
        x = self.proj_in(x)

        h = w = int(x.shape[1]**0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.down1(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.block_in(x)

        h = w = int(x.shape[1]**0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.down2(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.block_out(x)

        x = self.proj_out(x)
        return x




class Cheap_SimpleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_in_block, num_out_block, num_heads=32, mlp_ratio=4, groups=32, eps=1e-6):
        super().__init__()
        self.proj_in = nn.Sequential(nn.Linear(in_channels, in_channels),
                                  nn.GELU(),
                                  nn.Linear(in_channels, in_channels))

        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.block_in = nn.Sequential(
            *([BaseConv2D(in_channels, groups, eps), Block(in_channels, num_heads, mlp_ratio)] * num_in_block)
        ) if num_in_block > 0 else nn.Identity()

        self.down2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=0) if num_out_block > 0 else nn.Identity()
        self.block_out = nn.Sequential(
            *([BaseConv2D(in_channels, groups, eps), Block(in_channels, num_heads, mlp_ratio)] * num_out_block)
        ) if num_out_block > 0 else nn.Identity()

        self.proj_out = nn.Sequential(nn.Linear(in_channels, out_channels),
                                  nn.GELU(),
                                  nn.Linear(out_channels, out_channels))

    def forward(self, x):
        x = self.proj_in(x)

        h = w = int(x.shape[1]**0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.down1(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.block_in(x)

        h = w = int(x.shape[1]**0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.down2(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.block_out(x)

        x = self.proj_out(x)
        return x

if __name__ == '__main__':
    config = type('Args', (), {
        "hidden_size": 4096,
        "mm_hidden_size": 1024
    })()
    projector_type = 'simple_in1_out1'
    pattern = r"simple_in(\d+)_out(\d+)"

    match = re.search(pattern, projector_type)
    num_in_block = int(match.group(1))
    num_out_block = int(match.group(2))

    x = torch.randn(2, 256, 1024)
    # simple = SimpleBlock(config.mm_hidden_size, config.hidden_size, num_in_block, num_out_block)
    simple = Cheap_SimpleBlock(config.mm_hidden_size, config.hidden_size, num_in_block, num_out_block)
    y = simple(x)
    print(y.shape)
    params_count = sum(p.numel() for p in simple.parameters() if p.requires_grad)
    print(round(params_count/1000000, 2))


    # simple_in1_out1   822.2    # 256 -> 36
    # simple_in1_out0   362.87   # 256 -> 64
    # qformer4_36       952.57   # 256 -> 36
    # qformer2_64       503.75   # 256 -> 64

    # cheap_simple_in1_out1   76.58   # 256 -> 36
    # cheap_simple_in1_out0   45.11   # 256 -> 64
    # cheap_qformer4_36       90.3    # 256 -> 36
    # cheap_qformer2_64       56.74   # 256 -> 64

    # pool_mlp2x_gelu         20.98   # 256 -> 64
