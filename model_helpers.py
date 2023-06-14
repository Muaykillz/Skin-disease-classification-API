from fastai.vision.all import params, imagenet_stats
from fastai.vision.models import mobilenet_v2
from fastcore.foundation import L
import torch.nn as nn

def _mobilenet_v2_split(m: nn.Module):
    return L(m[0][0][:7], m[0][0][7:], m[1:]).map(params)

_mobilenet_v2_meta = {'cut': -1, 'split': _mobilenet_v2_split, 'stats': imagenet_stats}
model_meta = {}
model_meta[mobilenet_v2] = {**_mobilenet_v2_meta}
