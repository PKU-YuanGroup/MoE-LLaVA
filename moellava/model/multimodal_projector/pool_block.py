import re

import torch
from einops import rearrange
from timm.models.vision_transformer import Block
from torch import nn

class Pool_Block(nn.Module):
    def __init__(self, projector_type, config):
        super(Pool_Block, self).__init__()
        self.proj_in = nn.AvgPool2d(kernel_size=2, stride=2)
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        self.block = nn.Sequential(*modules)

    def forward(self, x):
        h = w = int(x.shape[1] ** 0.5)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.block(x)

        return x

if __name__ == '__main__':
    config = type('Args', (), {
        "hidden_size": 4096,
        "mm_hidden_size": 1024
    })()
    projector_type = 'mlp2x_gelu'

    x = torch.randn(2, 256, 1024)
    simple = Pool_Block(projector_type, config)
    y = simple(x)
    print(y.shape)
    params_count = sum(p.numel() for p in simple.parameters() if p.requires_grad)
    print(round(params_count/1000000, 2))


