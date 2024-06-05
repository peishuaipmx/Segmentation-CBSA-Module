
from functools import partial
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class PatchEmbed(nn.Module):
    """
    2D Image to Patch Embedding
    """
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.flat = nn.Flatten(start_dim=2)

    def forward(self, x):
        (_,_,w,_) = x.size()
        num_patches = (w//self.patch_size) *(w//self.patch_size)
        x1 = self.flat(x.unfold(-2, 2, 2).unfold(-1, 2, 2))
        x2 = x1.unfold(2, num_patches, num_patches)
        x2 = torch.transpose(x2, dim0=-2, dim1=-1)
        return x2

class Hidden_conv_block(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 drop_out_ratio=0.):
        super(Hidden_conv_block, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, 1, 0),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(drop_out_ratio),
        )

    def forward(self, x):
        a = self.layer(x)
        return a

class Attention(nn.Module):
    def __init__(self,
                 in_channel,
                 out_channel,
                 patch_size=2,
                 attn_drop_ratio=0.25,
                 ):
        super(Attention, self).__init__()
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.patchEmbedding = PatchEmbed(patch_size)
        self.ConvQ = Hidden_conv_block(in_channel, self.out_channel, attn_drop_ratio)
        self.ConvK = Hidden_conv_block(in_channel, self.out_channel, attn_drop_ratio)
        self.dropout = nn.Dropout(attn_drop_ratio)
        self.softmax = nn.Softmax(dim=-1)
        self.patch_size = patch_size
        self.act = nn.GELU()
        self.norm = nn.BatchNorm2d(self.out_channel)

    def forward(self, x):
        (_,_,w,_) = x.size()
        patch_num = w//self.patch_size
        scale = patch_num ** -1
        q = self.patchEmbedding(self.ConvQ(x)) * scale
        k = self.patchEmbedding(self.ConvK(x)) * scale
        attn = q @ torch.transpose(k, dim0=-2, dim1=-1)
        attn = torch.sum(attn, dim=-2) * scale
        attn = self.softmax(attn)
        attn = attn.unfold(2, patch_num, patch_num)
        attn = F.interpolate(attn, scale_factor=2)
        out = torch.mul(attn, x)

        return self.norm(out) + x
