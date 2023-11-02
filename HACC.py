import warnings
warnings.filterwarnings("ignore")
from functools import partial
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from torch import FloatTensor
from torch.nn.parameter import Parameter
import numpy as np
from torch.nn.modules.module import Module
from scipy.spatial.distance import pdist, squareform
from torch.autograd import Variable
import numpy as np


class CBG(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2)
        self.conv = nn.Conv2d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, groups=groups)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.act(output)
        return output


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1, groups=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut,kSize, stride=stride, padding=padding, bias=False,
                              dilation=d, groups=groups)

    def forward(self, input):
        output = self.conv(input)
        return output


class MDP(nn.Module):
    def __init__(self, nIn, nOut, stride=1, k=4, r_lim=9, down_method='esp'):
        super().__init__()
        self.stride = stride
        n = int(nOut / k)
        n1 = nOut - (k - 1) * n
        assert down_method in ['avg', 'esp'], 'One of these is suppported (avg or esp)'
        assert n == n1, "n(={}) and n1(={}) should be equal for Depth-wise Convolution ".format(n, n1)
        self.proj_1x1 = CBG(nIn, n, 1, stride=1, groups=1)

        map_receptive_ksize = {3: 1, 5: 2, 7: 3, 9: 4, 11: 5, 13: 6, 15: 7, 17: 8}
        self.k_sizes = list()
        for i in range(k):
            ksize = int(3 + 2 * i)
            ksize = ksize if ksize <= r_lim else 3
            self.k_sizes.append(ksize)
        self.k_sizes.sort()
        self.spp_dw = nn.ModuleList()
        for i in range(k):
            d_rate = map_receptive_ksize[self.k_sizes[i]]
            self.spp_dw.append(CDilated(n, n, kSize=3, stride=stride, groups=1, d=d_rate))

    def forward(self, input):
        output1 = self.proj_1x1(input)
        output = [self.spp_dw[0](output1)]
        for k in range(1, len(self.spp_dw)):
            out_k = self.spp_dw[k](output1)
            out_k = out_k + output[0]
            output.append(out_k)
        output.append(output1)
        expanded = torch.cat(output, 1)
        return expanded


class FPN(nn.Module):
    """
    Feature Pyramid Network.

    This is an implementation of - Feature Pyramid Networks for Object
    Detection (https://arxiv.org/abs/1612.03144)

    Args:
        in_channels (List[int]):
            number of input channels per scale

        out_channels (int):
            number of output channels (used at each scale)

        num_outs (int):
            number of output scales

        start_level (int):
            index of the first input scale to use as an output scale

        end_level (int, default=-1):
            index of the last input scale to use as an output scale
    """

    def __init__(self, in_channels, out_channels, num_outs, start_level=0, end_level=-1,
                 extra_convs_on_inputs=True, bn=True):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs

        self.fp16_enabled = False

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level < inputs, no extra level is allowed
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level

        self.extra_convs_on_inputs = extra_convs_on_inputs

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for i in range(self.start_level, self.backbone_end_level):
            l_conv = FPN_Conv2d(in_channels[i], out_channels,
                            1, bn=bn, bias=not bn, same_padding=True)

            fpn_conv = FPN_Conv2d(out_channels, out_channels, 3,
                              bn=bn, bias=not bn, same_padding=True)

            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        # add extra conv layers (e.g., RetinaNet)
        self.init_weights()
    # default init_weights for conv(msra) and norm in ConvModule

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, inputs):

        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [lateral_conv(inputs[i + self.start_level])
                    for i, lateral_conv in enumerate(self.lateral_convs)]

        # build top-down path
        used_backbone_levels = len(laterals)
        # num = 0
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # print(prev_shape)
            # print(laterals[i].shape)
            # print(laterals[i-1].shape)
            # print(num)
            # num += 1
            laterals[i - 1] += F.interpolate(laterals[i],
                                             size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        outs = [self.fpn_convs[i](laterals[i])
                for i in range(used_backbone_levels)]

        return tuple(outs)


class FPN_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, NL='GELU', same_padding=False, bn=True, bias=True):
        super(FPN_Conv2d, self).__init__()
        padding = int((kernel_size - 1) // 2) if same_padding else 0

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding=padding, bias=bias, groups=4)

        self.bn = nn.BatchNorm2d(out_channels) if bn else None
        if NL == 'relu':
            self.relu = nn.ReLU()
        elif NL == 'GELU':
            self.relu = nn.GELU()
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size,
               W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous(
    ).view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size,
                     window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - \
            coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(
            1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - \
            1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index",
                             relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C //
                                  self.num_heads).permute(2, 0, 3, 1, 4)
        # make torchscript happy (cannot use tensor as tuple)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(
            2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N,
                             N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            # nW, window_size, window_size, 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1,
                                             self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        # nW*B, window_size, window_size, C
        x_windows = window_partition(shifted_x, self.window_size)
        # nW*B, window_size*window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)

        # W-MSA/SW-MSA
        # nW*B, window_size*window_size, C
        attn_windows = self.attn(x_windows, mask=self.attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1,
                                         self.window_size, self.window_size, C)
        shifted_x = window_reverse(
            attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(
                self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops


class PatchMerging(nn.Module):
    r""" Patch Merging Layer.

    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f"input_resolution={self.input_resolution}, dim={self.dim}"

    def flops(self):
        H, W = self.input_resolution
        flops = H * W * self.dim
        flops += (H // 2) * (W // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (
                                     i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(
                                     drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(
                input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        x1 = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x, x1

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=384, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        # patch  / 4
        patches_resolution = [img_size[0] //
                              patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        Ho, Wo = self.patches_resolution
        flops = Ho * Wo * self.embed_dim * self.in_chans * \
            (self.patch_size[0] * self.patch_size[1])
        if self.norm is not None:
            flops += Ho * Wo * self.embed_dim
        return flops


class HGNN(nn.Module):
    def __init__(self, in_ch, n_hid, n_class, dropout=0.5, momentum=0.1):
        super(HGNN, self).__init__()
        self.dropout = dropout
        # self.batch_normalzation1 = nn.BatchNorm1d(in_ch, momentum=momentum)
        self.hgc1 = HGNN_conv(in_ch, n_hid)
        self.hgc2 = HGNN_conv(n_hid, n_class)

    def forward(self, x, G):
        # print(x.shape)
        # x = self.batch_normalzation1(x)
        x = F.relu(self.hgc1(x, G))
        # print(x.shape)
        x = F.dropout(x, self.dropout)
        x = self.hgc2(x, G)
        # print(x.shape)
        return x


class HGNN_conv(nn.Module):
    def __init__(self, in_ft, out_ft, bias=True):
        super(HGNN_conv, self).__init__()

        self.weight = Parameter(torch.Tensor(in_ft, out_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, x: torch.Tensor, G: torch.Tensor):
        x = x.matmul(self.weight)
        if self.bias is not None:
            x = x + self.bias
        x = G.matmul(x)
        return x


def Eu_dis(x):
    """
    Calculate the distance among each raw of x
    :param x: N X D
                N: the object number
                D: Dimension of the feature
    :return: N X N distance matrix
    """
    x = np.mat(x)  # 杞垚np
    aa = np.sum(np.multiply(x, x), 1)  # N X 1
    ab = x * x.T  # N X N
    dist_mat = aa + aa.T - 2 * ab
    dist_mat[dist_mat < 0] = 0
    dist_mat = np.sqrt(dist_mat)  # 瀵规暟缁勬瘡涓厓绱犺繑鍥炰竴涓潪璐熷钩鏂规牴
    dist_mat = np.maximum(dist_mat, dist_mat.T)
    return dist_mat


def _generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    H = np.array(H)
    n_edge = H.shape[1]
    # the weight of the hyperedge
    W = np.ones(n_edge)
    # the degree of the node
    DV = np.sum(H * W, axis=1)
    # the degree of the hyperedge
    DE = np.sum(H, axis=0)

    invDE = np.mat(np.diag(np.power(DE, -1)))
    DV2 = np.mat(np.diag(np.power(DV, -0.5)))
    W = np.mat(np.diag(W))
    H = np.mat(H)
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2
        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G


def generate_G_from_H(H, variable_weight=False):
    """
    calculate G from hypgraph incidence matrix H
    :param H: hypergraph incidence matrix H
    :param variable_weight: whether the weight of hyperedge is variable
    :return: G
    """
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G


def construct_H_with_KNN_from_distance(dis_mat, k_neig, is_probH=False, m_prob=1):
    """
    construct hypregraph incidence matrix from hypergraph node distance matrix
    :param dis_mat: node distance matrix
    :param k_neig: K nearest neighbor
    :param is_probH: prob Vertex-Edge matrix or binary
    :param m_prob: prob
    :return: N_object X N_hyperedge
    """
    # print(dis_mat.shape)
    n_obj = dis_mat.shape[0]
    # construct hyperedge from the central feature space of each node
    n_edge = n_obj
    H = np.zeros((n_obj, n_edge))
    for center_idx in range(n_obj):  # 涓績鑺傜偣鐨勬暟閲忓凡缁忕‘瀹�
        dis_mat[center_idx, center_idx] = 0  # 瀵硅绾夸负0锛岃妭鐐硅嚜鐩稿叧涓�0
        dis_vec = dis_mat[center_idx]  # 鑾峰彇杩欎竴琛�
        # print(dis_vec.shape)
        # argsort 锛� 瀵规暟缁勪腑鐨勫厓绱犺繘琛屼粠灏忓埌澶ф帓搴忥紝骞惰繑鍥炵浉搴斿簭鍒楀厓绱犵殑鏁扮粍涓嬫爣銆�
        res_vec = list(reversed(np.argsort(dis_vec)))  # 鐩稿叧鍒嗘暟浠庡ぇ鍒板皬鎺掑簭浜�
        nearest_idx = np.array(res_vec).squeeze()   #

        avg_dis = np.average(dis_vec)  # 鍙栫浉鍏冲垎鏁扮殑骞冲潎
        # 鍓峃涓珮鐩稿叧鐨勮妭鐐逛腑娌℃湁涓績鐐圭殑璇濓紝灏辨妸鏈€鍚庝竴涓浉鍏宠妭鐐硅涓轰腑蹇冪偣
        if not np.any(nearest_idx[:k_neig] == center_idx):
            nearest_idx[k_neig - 1] = center_idx

        # print(nearest_idx[:k_neig])
        for node_idx in nearest_idx[:k_neig]: # 灏嗗墠N涓妭鐐逛箣闂寸殑杩炴帴鍏崇郴璁颁负1锛岀粍鎴愪竴鏉¤秴杈�
            if is_probH:  # 鐢ㄦ鐜囪繕鏄洿鎺ョ敤1
                H[node_idx, center_idx] = np.exp(-dis_vec[node_idx]
                                                 ** 2 / (m_prob * avg_dis) ** 2)
                # print(H[node_idx, center_idx])
        else:
            # print("#")
            H[node_idx, center_idx] = 1.0
        # print(H)
    return H


class SimilarityAdj(Module):
    def __init__(self, in_features, out_features):
        super(SimilarityAdj, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight0 = Parameter(FloatTensor(in_features, out_features))
        self.weight1 = Parameter(FloatTensor(in_features, out_features))
        self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        # stdv = 1. / sqrt(self.weight0.size(1))
        nn.init.xavier_uniform_(self.weight0)
        nn.init.xavier_uniform_(self.weight1)

    def forward(self, input, seq_len):
        # To support batch operations
        soft = nn.Softmax(1)
        # print(input.shape) # [2, 512, 512]
        theta = torch.matmul(input, self.weight0)
        # print(theta.shape) # [2, 512, 32]
        phi = torch.matmul(input, self.weight1)
        # print(phi.shape) # [2, 512, 32]
        phi2 = phi.permute(0, 2, 1)
        sim_graph = torch.matmul(theta, phi2)
        # print(sim_graph.shape) # [2, 576, 576]

        theta_norm = torch.norm(theta, p=2, dim=2, keepdim=True)  # B*T*1
        # print(theta_norm.shape) # [2, 576, 1]
        phi_norm = torch.norm(phi, p=2, dim=2, keepdim=True)  # B*T*1
        # print(phi_norm.shape) # [2, 576, 1]
        x_norm_x = theta_norm.matmul(phi_norm.permute(0, 2, 1))
        # print(x_norm_x.shape) # [2, 576, 576]
        sim_graph = sim_graph / (x_norm_x + 1e-20)

        # print(sim_graph.shape)
        output = torch.zeros_like(sim_graph)
        for i in range(len(seq_len)):
            tmp = sim_graph[i, :seq_len[i], :seq_len[i]]
            adj2 = tmp
            # print(adj2.shape)
            # adj2 = F.threshold(adj2, 0.7, 0)
            adj2 = soft(adj2)
            # print(adj2.shape)
            output[i, :seq_len[i], :seq_len[i]] = adj2
        # print(output.shape) # [2, 512, 512]

        # 鐠烘繄顬囬惌鈺呮█
        input = input.cpu().detach().numpy()
        dis_mats = []
        for i in range(len(input)):
            dis_mat = Eu_dis(input[i])
            dis_mats.append(dis_mat)

        dis_mats = torch.Tensor(np.array(dis_mats))
        # print(sim_graph.shape,  dis_mats.shape)
        output1 = torch.zeros_like(dis_mats)
        for i in range(len(seq_len)):
            tmp = dis_mats[i, :seq_len[i], :seq_len[i]]
            adj2 = tmp
            # print(adj2.shape)
            # adj2 = F.threshold(adj2, 0.7, 0)
            adj2 = soft(adj2)
            # print(adj2.shape)
            output1[i, :seq_len[i], :seq_len[i]] = adj2
        # print(output.shape) # [2, 512, 512]

        dis_mats = np.array(output1)

        # print(sim_graph.shape,  dis_mats.shape)
        # 閻╃ǹ鍙ч崗宕囬兇閻晠妯€
        sim_graph = output.cpu().detach().numpy()
        # print(sim_graph) # [2, 576, 576]

        # print(sim_graph[0][0][:50], dis_mats[0][0][:50])
        # sim_graph = sim_graph +  list(np.array(dis_mats)*0.1)  #
        # sim_graph = sim_graph +  list(np.array(dis_mats)*0.2)  #
        sim_graph = sim_graph + dis_mats*0.5  #

        # sim_graph =  nn.Sigmoid(sim_graph) +  nn.Sigmoid(dis_mats)

        hypergraph = []
        for i in range(len(sim_graph)):
            # print(sim_graph[i].shape)
            H = construct_H_with_KNN_from_distance(sim_graph[i], 10, False, 1)
            # print(H[0])
            hypergraph.append(H)

        # print("#################",H.shape)
        # print(len(hypergraph), hypergraph[0].shape) # [2,576,576]
        G = generate_G_from_H(hypergraph, variable_weight=False)
        # print(len(H))
        # print(G[0].shape) # [2,576,576]
        for i in range(len(G)):
            G[i] = G[i].A
        # G[1] = G[1].A
        # print(type(hypergraph[0]), len(hypergraph))
        # print(type(G[0]), len(G))
        # print(len(hypergraph), hypergraph[0].shape)
        G = torch.Tensor(G).cuda()

        return G

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'


class SwinTransformer(nn.Module):
    def __init__(self, img_size=384, patch_size=4, in_chans=3, num_classes=1000,
                 embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
                 window_size=7, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(
                                   depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=PatchMerging if (
                                   i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        in_channels = [192, 384, 768, 1536]
        self.neck1 = FPN(in_channels, out_channels=256, num_outs=4)

        self.MDP = MDP(nIn = 2880, nOut = 1024)

        self.conv1 = nn.Conv1d(in_channels=1280, out_channels=576, kernel_size=1, padding=0)
        self.simAdj = SimilarityAdj(576, 576)

        self.gc1 = HGNN(576, 576, 576)
        self.gc2 = HGNN(576, 576, 576)
        self.sigmoid = nn.Sigmoid()

        self.out = nn.Sequential(
            nn.GELU(),
            nn.Linear(576, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()


    def forward(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        out = []
        for layer in self.layers:
            x, x_last = layer(x)
            B, L, C = x_last.shape
            H = W = int(L**0.5)
            xf = x_last.view(B, C, H, W)
            out.append(xf)

        '''
        torch.Size([2, 192, 96, 96])
        torch.Size([2, 384, 48, 48])
        torch.Size([2, 768, 24, 24])
        torch.Size([2, 1536, 12, 12])
        '''

        x = torch.cat([out[2], F.interpolate(out[3], (out[2].shape[2], out[2].shape[3]), mode='nearest'), \
                F.interpolate(out[1], (out[2].shape[2], out[2].shape[3]), mode='nearest'), \
                F.interpolate(out[0], (out[2].shape[2], out[2].shape[3]), mode='nearest')], dim=1)

        x = self.MDP(x)

        x = x.reshape(x.shape[0], x.shape[1], x.shape[2]*x.shape[3])
        x = self.conv1(x)

        seq_len = torch.sum(torch.max(torch.abs(x), dim=2)[0] > 0, 1)
        adj = self.simAdj(x, seq_len)
        x1 = torch.relu(self.gc1(x, adj))
        x2 = torch.relu(self.gc2(x1, adj))
        wei = self.sigmoid(x2)
        x2 = x*wei

        x = x2.mean(dim=1)
        x = self.out(x)
        return x


# swin_large_patch4_window12_384
def SwinTransformer_cc(pretrained=True, **kwargs):
    model = SwinTransformer(img_size=384, patch_size=4, in_chans=3, num_classes=1000,
                            embed_dim=192, depths=[2, 2, 18, 2], num_heads=[6, 12, 24, 48],
                            window_size=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                            drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                            norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                            use_checkpoint=True, **kwargs)
    # model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            './pre_models/swin_large_patch4_window12_384_22k.pth')
        model.load_state_dict(checkpoint["model"], strict=False)
        print("load swin transformer pretrained")
    return model


if __name__ == "__main__":
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'

    net = SwinTransformer_cc(pretrained=False).cuda()
    # print(net)
    input = torch.randn(2, 3, 384, 384).cuda()
    x = net(input)
    print(x.shape, x)
    # from thop import profile
    # flops, params = profile(net, inputs=(input, ))
    # # summary(net,(3,64 ,64 ),batch_size=4)
    # print('Total params: %.2fM' % (params/1000000.0))
    # print('Total flops: %.2fG' % (flops/1000000000.0))
