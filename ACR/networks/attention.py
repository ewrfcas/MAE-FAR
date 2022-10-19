import numpy as np
import torch.nn as nn
from torch.nn import functional as F


def extract_patches(x, kernel=3, padding=None, stride=1):
    if padding is None:
        padding = 1 if kernel > 1 else 0
    if padding > 0:
        x = nn.ReplicationPad2d(padding)(x)
    x = x.permute(0, 2, 3, 1)
    all_patches = x.unfold(1, kernel, stride).unfold(2, kernel, stride)
    return all_patches


class GroupConvAttention(nn.Module):
    def __init__(self):
        super(GroupConvAttention, self).__init__()

    def forward(self, ori_x, scores, alpha):
        # get shapes [B,C,H,W]
        [batch, channel, ori_height, ori_width] = list(ori_x.size())
        if 48 < ori_height < 64:
            height, width = 64, 64
        elif 32 < ori_height < 48:
            height, width = 48, 48
        else:
            height, width = ori_height, ori_width
        x = F.interpolate(ori_x, size=(height, width), mode='bilinear', align_corners=False)

        [_, _, hw] = scores.shape
        hs = int(np.sqrt(hw))
        ws = int(np.sqrt(hw))
        rate = int(height / hs)

        # value for back features
        vksize = int(rate * 2)  # must be rate*2 for transposeconv
        vpadding = rate // 2
        if rate % 2 != 0:
            vpadding += 1
        value = extract_patches(x, kernel=vksize, padding=vpadding, stride=rate)
        if rate == 1:
            value = value[:, :-1, :-1, ...]
        value = value.contiguous().reshape(batch, hs * ws, channel, vksize, vksize)  # [B,HW,C,K,K]

        # groupconv for attention (qk)v: B*[C,H,W]·B*[HW,C,K,K]->B*[C,H,W]
        scores = scores.permute(0, 2, 1)  # [B,HW,HW(softmax)]->[B,HW(softmax),HW]
        # [1,B*C,H,W]->[1,B*HW,H,W]->[B,HW,H,W]
        scores_ = scores.reshape(1, batch * hs * ws, hs, ws)  # [1,B*HW,H,W]
        value = value.reshape(batch * hs * ws, channel, vksize, vksize)  # [B*HW,C,K,K]
        y = F.conv_transpose2d(scores_, value, stride=rate, padding=vpadding, groups=batch) / 4.
        if rate % 2 != 0:
            y = nn.ReplicationPad2d([0, 1, 0, 1])(y)
        y = y.contiguous().reshape(batch, channel, height, width)  # [B,C,H,W]

        if height != ori_height:
            y = F.interpolate(y, size=(ori_height, ori_width), mode='bilinear', align_corners=False)

        return ori_x + y * alpha
