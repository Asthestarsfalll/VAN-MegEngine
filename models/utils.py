import math

import megengine as mge
import megengine.functional as F
import megengine.module as M
import numpy as np


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = mge.tensor(1 - drop_prob, dtype=x.dtype)
    size = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + mge.random.normal(mean=0, std=1, size=size)
    random_tensor = F.floor(random_tensor)  # binarize
    output = x / keep_prob * random_tensor
    return output

class DropPath(M.Module):

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


def to_2tuple(n):
    return (n, n)

def init_weights(m):
    if isinstance(m, M.Linear):
        # trunc_normal_(m.weight, std=.02)
        if isinstance(m, M.Linear) and m.bias is not None:
            M.init.zeros_(m.bias)
    elif isinstance(m, M.LayerNorm):
        M.init.zeros_(m.bias)
        M.init.ones_(m.weight)
    elif isinstance(m, M.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        M.init.normal_(m.weight, 0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            M.init.zeros_(m.bias)
