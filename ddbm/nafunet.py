from abc import abstractmethod

import math

# TODO: EVERYTHING !!!!!!
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .unet import Upsample, Downsample, TimestepBlock
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    zero_module,
    normalization,
    linear,
    SiLU,
    timestep_embedding,
    LayerNorm
)


# Fusion SAR with Optical Featrue
# We need to handle Optical Feature (Q) and SAR feature (K, V)
# Based on QKVAttention (unet.py by NVIDIA)
# Q from Optical, KV from SAR. need to concat it before send it to SFBlock

class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class SqueezeExcite(nn.Module):
    def __init__(self, channels, reduction=4, dims=2):
        super().__init__()
        reduced = max(1, channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1) if dims == 2 else nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            conv_nd(dims, channels, reduced, 1),
            nn.SiLU(),
            conv_nd(dims, reduced, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        pooled = self.pool(x)
        return x * self.fc(pooled.to(x.dtype))

# TODO:
class RGBBlock(nn.Module):
    def __init__(self, channels, num_heads=4, dims=2):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.channels  = channels
        self.num_heads = num_heads
        self.head_dim  = channels // num_heads  # 채널 수를 헤드 개수로 나눠서 각 헤드 차원 설정
        
        self.norm_q  = normalization(1)
        self.norm_kv = normalization(2)

        self.q_proj = conv_nd(dims, 1, channels, 1)
        self.k_proj = conv_nd(dims, 2, channels, 1)
        self.v_proj = conv_nd(dims, 2, channels, 1)
        
        self.mlp = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )
        
        self.out_proj = zero_module(conv_nd(dims, channels, channels, 1))

    def forward(self, feat_opt):
        B, _, H, W = feat_opt.shape   # C=3
        HW = H * W

        b_channel  = feat_opt[:, 2:3, :, :]
        RG_channel = feat_opt[:, 0:2, :, :]

        q = self.q_proj(self.norm_q(b_channel)).to(feat_opt.dtype)
        k = self.k_proj(self.norm_kv(RG_channel)).to(feat_opt.dtype)
        v = self.v_proj(self.norm_kv(RG_channel)).to(feat_opt.dtype)

        q = q.view(B, self.channels, HW)
        k = k.view(B, self.channels, HW)
        v = v.view(B, self.channels, HW)

        q_h = q.view(B, self.num_heads, self.head_dim, HW)
        k_h = k.view(B, self.num_heads, self.head_dim, HW)
        v_h = v.view(B, self.num_heads, self.head_dim, HW)

        scale = self.head_dim ** -0.5
        attn = th.matmul(q_h, k_h.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)
        out_h = th.matmul(attn, v_h)

        out = out_h.view(B, self.channels, HW)

        out = out.permute(0, 2, 1).reshape(B*HW, self.channels)
        out = self.mlp(out)
        out = out.view(B, HW, self.channels).permute(0, 2, 1)

        out = out.view(B, self.channels, H, W) + feat_opt
        return self.out_proj(out)



class MBConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        expansion=6,
        reduction=4,
        dropout=0.0,
        dims=2,
    ):
        super().__init__()
        self.use_residual = (in_channels == out_channels)
        hidden = in_channels * expansion

        # expansion
        self.expand = conv_nd(dims, in_channels, hidden, 1)
        # depthwise
        self.dw = conv_nd(dims, hidden, hidden, 3, padding=1, groups=hidden)
        # channel‑attention
        self.se = SqueezeExcite(hidden, reduction, dims)
        # projection
        self.project = conv_nd(dims, hidden, out_channels, 1)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        dtype = x.dtype
        # keep original for residual
        x_in = x

        # Pointwise expansion
        h = self.expand(x).to(dtype)
        h = self.act(h)

        # Depthwise
        h = self.dw(h).to(dtype)
        h = self.act(h)

        # SE
        h = self.se(h).to(dtype)

        # Projection
        h = self.project(h).to(dtype)
        h = self.dropout(h)

        return x_in + h if self.use_residual else h



class SFBlock(nn.Module):

    def __init__(self, channels, num_heads=4, mlp_ratio=2, dims=2):
        super().__init__()
        assert channels % num_heads == 0
        self.num_heads = num_heads
        self.head_dim  = (channels * (dims == 2 and 1 or 1))  # we'll split features below
        # 1) Norm + 1×1 conv
        self.norm_q  = normalization(channels)
        self.q_proj  = conv_nd(dims, channels, channels, 1)
        self.norm_kv = normalization(channels)
        self.k_proj  = conv_nd(dims, channels, channels, 1)
        self.v_proj  = conv_nd(dims, channels, channels, 1)
        # 2) MLP on channel embeddings
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Linear(hidden, channels),
        )
        # 3) Final 1×1 conv
        self.out_proj = zero_module(conv_nd(dims, channels, channels, 1))

    def forward(self, feat_opt, feat_sar):
        B, C, H, W = feat_opt.shape
        HW = H * W

        # 1) Norm + conv1×1
        q = self.norm_q(feat_opt).to(feat_opt.dtype)
        k = self.norm_kv(feat_sar).to(feat_opt.dtype)
        v = self.norm_kv(feat_sar).to(feat_opt.dtype)

        q = self.q_proj(q)    # [B, C, H, W]
        k = self.k_proj(k)    # [B, C, H, W]
        v = self.v_proj(v)    # [B, C, H, W]

        # 2) Flatten spatial dims → feature dim
        #    shape becomes [B, C, HW]
        q = q.reshape(B, C, HW)
        k = k.reshape(B, C, HW)
        v = v.reshape(B, C, HW)

        # 3) Multi-head channel attention
        #    Sequence length = C, embed dim = HW // num_heads
        head_dim = HW // self.num_heads
        assert head_dim * self.num_heads == HW, "HW must be divisible by num_heads"

        # Split heads on feature axis
        # q_h: [B, num_heads, C, head_dim]
        q_h = q.reshape(B, C, self.num_heads, head_dim) \
               .permute(0, 2, 1, 3)
        k_h = k.reshape(B, C, self.num_heads, head_dim) \
               .permute(0, 2, 1, 3)
        v_h = v.reshape(B, C, self.num_heads, head_dim) \
               .permute(0, 2, 1, 3)

        # Scaled dot-product over channels
        scale = head_dim ** -0.5
        # attn: [B, num_heads, C, C]
        attn = (q_h @ k_h.transpose(-2, -1)) * scale
        attn = F.softmax(attn, dim=-1)

        # apply to V
        # out_h: [B, num_heads, C, head_dim]
        out_h = attn @ v_h

        # 4) Merge heads → [B, C, HW]
        out = out_h.permute(0, 2, 1, 3)   # [B, C, num_heads, head_dim]
        out = out.reshape(B, C, HW)

        # 5) MLP over channel embeddings
        #    We treat each of the HW positions independently:
        #    out.permute -> [B*HW, C], MLP, then back
        out = out.permute(0, 2, 1).reshape(B*HW, C)  # [B*HW, C]
        out = self.mlp(out)                          # [B*HW, C]
        out = out.reshape(B, HW, C).permute(0, 2, 1)  # [B, C, HW]

        # 6) reshape to (B,C,H,W), residual + final conv
        out = out.reshape(B, C, H, W) + feat_opt
        return self.out_proj(out)

class NAFBlock(nn.Module):
    def __init__(
        self,
        channels,
        dropout=0.0,
        dims=2,
        use_checkpoint=False,
        emb_channels=None,
        use_scale_shift_norm=False,
    ):
        super().__init__()
        assert channels % 2 == 0, "channels must be divisible by 2 for SimpleGate"
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.norm1 = normalization(channels)
        self.mbconv = MBConv(channels, channels, dims=dims, dropout=dropout)

        self.norm2 = normalization(channels)
        hidden = channels * 2
        self.ffn = nn.Sequential(
            conv_nd(dims, channels, hidden, 3, padding=1),
            conv_nd(dims, hidden, hidden, 3, padding=1),
        )
        self.sg = SimpleGate()

    def forward(self, x, emb=None):
        return checkpoint(self._forward, (x, emb), self.parameters(), self.use_checkpoint)

    def _forward(self, x, emb=None):
        h = self.norm1(x).to(x.dtype)
        h = self.mbconv(h)
        x1 = x + h

        h2 = self.norm2(x1).to(x.dtype)
        h2 = self.ffn(h2)

        h3 = self.sg(h2)

        return h3 + x1




class BlueScaler(nn.Module):
    def __init__(self, num_channels=13, blue_idx=1):

        super().__init__()
        assert 0 <= blue_idx < num_channels
        self.num_channels = num_channels
        self.blue_idx = blue_idx
        self.scale_blue = nn.Parameter(th.tensor(1.0))

    def forward(self, x):

        B, C, H, W = x.shape
        assert C == self.num_channels, \
            f"Expected {self.num_channels} channels but got {C}"
        
        scale = x.new_ones((1, C, 1, 1))
        scale[:, self.blue_idx, :, :] = self.scale_blue
        
        return x * scale
        

    

class NAFUNetModel(nn.Module):
    def __init__(
        self,
        in_channels=13,
        sar_channels=2,
        out_channels=13,
        model_channels=22,
        channel_mult=(1,2,4,8),
        num_naf_blocks=1,
        num_heads_per_level=(1,1,2,4),
        dropout=0.0,
        dims=2,
        use_checkpoint=False,
        conv_resample=True,
        use_fp16=False,
    ):
        super().__init__()
        self.dtype = th.float16 if use_fp16 else th.float32

        self.scaler = BlueScaler()
        self.emb_channels = model_channels

        self.opt_embed = conv_nd(dims, in_channels, model_channels, 3, padding=1)
        self.sar_embed = conv_nd(dims, sar_channels, model_channels, 3, padding=1)

        self.channel_list = [model_channels * m for m in channel_mult]
        self.num_levels   = len(self.channel_list)

        self.encoder_opt      = nn.ModuleList()
        self.encoder_sar      = nn.ModuleList()
        self.fusion_blocks    = nn.ModuleList()
        self.downsamples_opt  = nn.ModuleList()
        self.downsamples_sar  = nn.ModuleList()

        for lvl, ch in enumerate(self.channel_list):
            self.encoder_opt.append(nn.Sequential(*[
                NAFBlock(ch, dropout, dims, use_checkpoint, emb_channels=self.emb_channels)
                for _ in range(num_naf_blocks)
            ]))
            self.encoder_sar.append(nn.Sequential(*[
                NAFBlock(ch, dropout, dims, use_checkpoint, emb_channels=self.emb_channels)
                for _ in range(num_naf_blocks)
            ]))

            self.fusion_blocks.append(
                SFBlock(ch, num_heads=num_heads_per_level[lvl], dims=dims)
            )

            if lvl < self.num_levels-1:
                self.downsamples_opt.append(
                    Downsample(ch, use_conv=conv_resample, dims=dims)
                )
                self.downsamples_sar.append(
                    Downsample(ch, use_conv=conv_resample, dims=dims)
                )

        mid_ch = self.channel_list[-1]
        self.middle_block = NAFBlock(mid_ch, dropout, dims, use_checkpoint, emb_channels=self.emb_channels)

        self.decoder   = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for lvl, ch in enumerate(reversed(self.channel_list)):
            self.decoder.append(
                NAFBlock(ch, dropout, dims, use_checkpoint, emb_channels=self.emb_channels)
            )
            if lvl < self.num_levels-1:
                self.upsamples.append(
                    Upsample(ch, use_conv=conv_resample, dims=dims)
                )

        self.out = zero_module(conv_nd(dims, model_channels, out_channels, 1))

        if use_fp16:
            self.convert_to_fp16()

    def convert_to_fp16(self):
        for m in self.modules():
            if isinstance(m, LayerNorm):
                m.float()
            else:
                m.half()

    def convert_to_fp32(self):
        for m in self.modules():
            m.float()

    def forward(self, x, t, opt, sar):
        self.dtype = opt
        t   = t.to(self.dtype)
        opt = opt.to(self.dtype)
        sar = sar.to(self.dtype)

        t_emb = timestep_embedding(t, dim=self.emb_channels).to(self.dtype)
        
        # TODO:
        # opt = self.scaler(opt)

        h_opt = self.opt_embed(opt)
        h_sar = self.sar_embed(sar)

        for lvl in range(self.num_levels):

            for blk in self.encoder_opt[lvl]:
                h_opt = blk(h_opt, t_emb)
            for blk in self.encoder_sar[lvl]:
                h_sar = blk(h_sar, t_emb)

            h_opt = self.fusion_blocks[lvl](h_opt, h_sar)
            if lvl < self.num_levels-1:
                h_opt = self.downsamples_opt[lvl](h_opt)
                h_sar = self.downsamples_sar[lvl](h_sar)

        h = self.middle_block(h_opt, t_emb)
        for i, dec in enumerate(self.decoder):
            h = dec(h, t_emb)
            if i < len(self.upsamples):
                h = self.upsamples[i](h)

        return self.out(h)
