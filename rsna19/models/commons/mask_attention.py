import torch
import torch.nn as nn
import torch.nn.functional as F


class MaskAttention(nn.Module):
    def __init__(self, in_dim):
        super(MaskAttention, self).__init__()
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, attention_mask):
        batchsize, C, width, height = x.size()
        attention_mask = F.interpolate(attention_mask, (width, height), mode='area')
        att_out = self.value_conv(x)  # B x C x W x H
        att_out = att_out * attention_mask

        out = self.gamma * att_out + x
        return out


class MaskGWAP(nn.Module):
    def __init__(self):
        super(MaskGWAP, self).__init__()

    def forward(self, x, attention_mask):
        batchsize, C, width, height = x.size()
        attention_mask = F.interpolate(attention_mask, (width, height), mode='area')
        x = attention_mask * x
        gwap = torch.sum(x, dim=(2, 3))
        return gwap


class PoolAttentionConcat(nn.Module):
    def __init__(self):
        super(PoolAttentionConcat, self).__init__()
        self.mask_attention = MaskGWAP()

    def forward(self, x, attention_mask):
        avg_pool = F.avg_pool2d(x, x.shape[2:]).squeeze()
        attention_out = self.mask_attention(x, attention_mask)
        avg_max_pool = torch.cat((avg_pool, attention_out), 1)
        x = avg_max_pool.view(avg_max_pool.size(0), -1)
        return x
