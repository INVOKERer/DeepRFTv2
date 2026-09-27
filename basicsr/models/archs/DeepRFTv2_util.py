# --------------------------------------------------------
# Reversible Column Networks
# Copyright (c) 2022 Megvii Inc.
# Licensed under The Apache License 2.0 [see LICENSE for details]
# Written by Yuxuan Cai
# --------------------------------------------------------
import os.path

import cv2
import kornia.filters

from basicsr.models.archs.modules.revcol_function import *
from basicsr.utils.sv_blur import BatchBlur_SV, BatchBlur_SV2, BatchBlur_nopad
from basicsr.models.losses.losses import *
from basicsr.models.archs.baseblock import *
from basicsr.models.archs.utils_deblur import *
from collections import OrderedDict
from einops import repeat
from basicsr.models.archs.local_arch import Local_Base
from basicsr.models.archs.my_module import code_extra_mean_var, code_extra_mean_varX, kernel_extra_conv_tail_mean_var, num_kernel_extra_conv_tail_mean_var
from basicsr.models.archs.Flow_arch import KernelPrior
from basicsr.models.archs.EVSSM import EVS_nogrid, EVS_noflip

class DownSample(nn.Module):
    def __init__(self, n_feat, out_feat, mode='conv'):
        super().__init__()
        if mode == 'conv':
            self.body = nn.Sequential(nn.Conv2d(n_feat, out_feat, kernel_size=2, stride=2, padding=0, bias=False))
        elif mode == 'shuffle_conv':
            self.body = nn.Sequential(
                nn.PixelUnshuffle(2),
                nn.Conv2d(n_feat*4, out_feat, kernel_size=3, stride=1, padding=1, bias=False)
                                     )
        elif mode == 'conv_shuffle':
            self.body = nn.Sequential(nn.Conv2d(n_feat, out_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                     nn.PixelUnshuffle(2))
        elif mode == 'bilinear':
            self.body = nn.Sequential(nn.Upsample(scale_factor=0.5, mode='bilinear', align_corners=False),
                                      nn.Conv2d(n_feat, out_feat, 3, stride=1, padding=1, bias=False))
    def forward(self, x):
        return self.body(x)
class UpSample(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, n_feat, out_feat, mode='conv'):
        super().__init__()
        if mode == 'conv':
            self.body = nn.ConvTranspose2d(n_feat, out_feat, kernel_size=2, stride=2)
        elif mode == 'conv_shuffle':
            self.body = nn.Sequential(nn.Conv2d(n_feat, out_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))
        elif mode == 'conv1x1_shuffle':
            self.body = nn.Sequential(nn.Conv2d(n_feat, out_feat * 4, kernel_size=1, stride=1, padding=0, bias=False),
                                  nn.PixelShuffle(2))
        elif mode == 'shuffle_conv':
            self.body = nn.Sequential(nn.PixelShuffle(2),
                                      nn.Conv2d(n_feat // 4, out_feat, kernel_size=3, stride=1, padding=1, bias=False)
                                  )
        elif mode == 'bilinear':
            self.body = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                  nn.Conv2d(n_feat, out_feat, 3, stride=1, padding=1, bias=False))

    def forward(self, x):
        return self.body(x)

# from torch.cuda.amp import autocast, GradScaler
class ApplyCoeffs(nn.Module):
    def __init__(self):
        super(ApplyCoeffs, self).__init__()
        self.degree = 3

    def forward(self, coeff, full_res_input):
        '''
            Affine:
            r = a11*r + a12*g + a13*b + a14
            g = a21*r + a22*g + a23*b + a24
            ...
        '''

        R = torch.sum(full_res_input * coeff[:, 0:3, :, :], dim=1, keepdim=True) + coeff[:, 3:4, :, :]
        G = torch.sum(full_res_input * coeff[:, 4:7, :, :], dim=1, keepdim=True) + coeff[:, 7:8, :, :]
        B = torch.sum(full_res_input * coeff[:, 8:11, :, :], dim=1, keepdim=True) + coeff[:, 11:12, :, :]

        # torch.sum --->  Conv2D(in=3, out=1, k=3, s=1, p=1)

        return torch.cat([R, G, B], dim=1)


class Fusion_Decoder(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        # n = 2**(2-level)
        n = 2 ** (3 - level)
        h, w = train_size[0] // n, train_size[1] // n
        # self.fourier_refine = MLP_UHD([h, w, channels[level+1]], 1) if level in [0, 1, 2] else nn.Identity()
        self.down = nn.Sequential(
            nn.Conv2d(channels[level + 1], channels[level], kernel_size=2, stride=2),
            LayerNorm(channels[level]),
        ) if level in [0, 1, 2] else nn.Identity()

        # if not first_col:
        # self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()
        # self.up = UpsampleX(channels[level], channels[level+1]) if level in [0, 1, 2] else nn.Identity()
        self.up = nn.Sequential(nn.Conv2d(channels[level] * 2, channels[level] * 4, 1, bias=False),
                                nn.PixelShuffle(2),
                                # LayerNorm(channels[level])
                                ) if level in [0, 1, 2, 3] else nn.Identity()
        # self.fuse = Fuse(channels[level], ffn_expansion_factor=ffn_expansion_factor) if level in [0, 1, 2] else nn.Identity()
        # self.fuse = FusNext(channels[level], ffn_expansion_factor=2) if level in [0, 1, 2] else nn.Identity()
        # self.fuse = Fusext(channels[level], ffn_expansion_factor=2) if level in [0, 1, 2] else nn.Identity()
        # self.fuse = FuseV5(channels[level], ffn_expansion_factor=ffn_expansion_factor) if level in [0, 1, 2] else nn.Identity()
        # self.prompt_emb = PromptGenBlock(channels[level], prompt_dim=channels[level],
        #                                  prompt_len=5, prompt_size=[h, w], num_heads=num_heads) if level in [0, 1, 2] else nn.Identity()

    def forward(self, *args):

        c_down, c_up = args

        if self.level == 3:
            # print(c_down.shape)
            x = self.up(c_down)
        else:
            # print(c_up.shape, c_down.shape)
            x = self.down(c_up) + self.up(c_down)
            # x = self.fuse(self.down(c_up), self.up(c_down))
            # c_up = self.down(c_up)
            # prompt = self.prompt_emb(c_up)
            # x = self.fuse(c_up+prompt, self.up(c_down))
            # x = self.down(c_up) + self.up(c_down)
        return x
class Fusion_Decoder_3level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        # n = 2**(2-level)
        n = 2 ** (3 - level)
        h, w = train_size[0] // n, train_size[1] // n
        # self.fourier_refine = MLP_UHD([h, w, channels[level+1]], 1) if level in [0, 1, 2] else nn.Identity()
        self.down = nn.Sequential(
            nn.Conv2d(channels[level + 1], channels[level], kernel_size=2, stride=2),
            LayerNorm(channels[level]),
        ) if level in [0, 1] else nn.Identity()

        # if not first_col:
        # self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()
        # self.up = UpsampleX(channels[level], channels[level+1]) if level in [0, 1, 2] else nn.Identity()
        self.up = nn.Sequential(nn.Conv2d(channels[level] * 2, channels[level] * 4, 1, bias=False),
                                nn.PixelShuffle(2),
                                # LayerNorm(channels[level])
                                ) if level in [0, 1, 2] else nn.Identity()
        # self.fuse = Fuse(channels[level], ffn_expansion_factor=ffn_expansion_factor) if level in [0, 1, 2] else nn.Identity()
        # self.fuse = FusNext(channels[level], ffn_expansion_factor=2) if level in [0, 1, 2] else nn.Identity()
        # self.fuse = Fusext(channels[level], ffn_expansion_factor=2) if level in [0, 1, 2] else nn.Identity()
        # self.fuse = FuseV5(channels[level], ffn_expansion_factor=ffn_expansion_factor) if level in [0, 1, 2] else nn.Identity()
        # self.prompt_emb = PromptGenBlock(channels[level], prompt_dim=channels[level],
        #                                  prompt_len=5, prompt_size=[h, w], num_heads=num_heads) if level in [0, 1, 2] else nn.Identity()

    def forward(self, *args):

        c_down, c_up = args

        if self.level == 3:
            # print(c_down.shape)
            x = self.up(c_down)
        else:
            # print(c_up.shape, c_down.shape)
            x = self.down(c_up) + self.up(c_down)
            # x = self.fuse(self.down(c_up), self.up(c_down))
            # c_up = self.down(c_up)
            # prompt = self.prompt_emb(c_up)
            # x = self.fuse(c_up+prompt, self.up(c_down))
            # x = self.down(c_up) + self.up(c_down)
        return x

class Fusion_Simple_Decoder_3level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        # n = 2**(2-level)
        n = 2 ** (3 - level)
        h, w = train_size[0] // n, train_size[1] // n
        # self.fourier_refine = MLP_UHD([h, w, channels[level+1]], 1) if level in [0, 1, 2] else nn.Identity()
        # self.down = nn.Sequential(
        #     nn.Conv2d(channels[level + 1], channels[level], kernel_size=2, stride=2),
        #     LayerNorm(channels[level]),
        # ) if level in [0, 1] else nn.Identity()

        # if not first_col:
        # self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()
        # self.up = UpsampleX(channels[level], channels[level+1]) if level in [0, 1, 2] else nn.Identity()
        self.up = nn.Sequential(nn.Conv2d(channels[level] * 2, channels[level] * 4, 1, bias=False),
                                nn.PixelShuffle(2),
                                # LayerNorm(channels[level])
                                ) if level in [0, 1] else nn.Identity()
        # self.fuse = Fuse(channels[level], ffn_expansion_factor=ffn_expansion_factor) if level in [0, 1, 2] else nn.Identity()
        # self.fuse = FusNext(channels[level], ffn_expansion_factor=2) if level in [0, 1, 2] else nn.Identity()
        # self.fuse = Fusext(channels[level], ffn_expansion_factor=2) if level in [0, 1, 2] else nn.Identity()
        # self.fuse = FuseV5(channels[level], ffn_expansion_factor=ffn_expansion_factor) if level in [0, 1, 2] else nn.Identity()
        # self.prompt_emb = PromptGenBlock(channels[level], prompt_dim=channels[level],
        #                                  prompt_len=5, prompt_size=[h, w], num_heads=num_heads) if level in [0, 1, 2] else nn.Identity()

    def forward(self, *args):

        c_down, c_up = args
        x = self.up(c_down)

        return x

class Fusion_Encoder(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            # LayerNorm(channels[level]),
        ) if level in [1, 2, 3] else nn.Identity()
        if not first_col:
            # self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()
            self.up = nn.Sequential(nn.Conv2d(channels[level + 1], channels[level] * 4, 1, bias=False),
                                    nn.PixelShuffle(2),
                                    LayerNorm(channels[level])
                                    ) if level in [0, 1, 2] else nn.Identity()
            # self.fuse = FusNext(channels[level], ffn_expansion_factor=2) if level in [0, 1, 2] else nn.Identity()

    def forward(self, *args):

        c_down, c_up = args

        if self.first_col:
            # print(self.first_col, c_down.shape)
            x = self.down(c_down)
            return x

        if self.level == 3:
            x = self.down(c_down)
        else:
            x = self.up(c_up) + self.down(c_down)
        return x
class Fusion_Encoder_nofuse(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66, downsample_mode='conv') -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = DownSample(channels[level - 1], channels[level], mode=downsample_mode) if level in [1, 2, 3] else nn.Identity()


    def forward(self, *args):

        c_down, c_up = args
        x = self.down(c_down)
        return x
class Fusion_Encoder_nodownup(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col

        if not self.first_col:
            self.down = nn.Sequential(
                nn.Conv2d(channels[level] + channels[level], channels[level], kernel_size=1, stride=1),
                # LayerNorm(channels[level]),
            ) if level in [0, 1, 2] else nn.Identity()



    def forward(self, *args):

        c_down, c_up = args
        # h, w = c_down.shape[-2:]
        # if h != self.ker_size or w != self.ker_size:
        #     c_down = self.resize(c_down)
            # ks = (self.ker_size - h) // 2
            # dim = (ks, ks, ks, ks)
            # c_down = torch.nn.functional.pad(c_down, dim, "constant")
        if not self.first_col and c_up is not None:
            x = self.down(torch.cat([c_down, c_up], dim=1)) # concat
        else:
            x = c_down
        # x = c_down
        # x = self.down(c_down)
        return x
class Fusion_ResNet_nodownup(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66, skip_connect=False) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.skip_connect = skip_connect
        if skip_connect:
            if not self.first_col:
                self.down = nn.Sequential(
                    nn.Conv2d(channels[level] + channels[level], channels[level], kernel_size=1, stride=1),
                    # LayerNorm(channels[level]),
                ) if level in [0, 1, 2] else nn.Identity()



    def forward(self, *args):

        c_down, c_up = args
        # h, w = c_down.shape[-2:]
        # if h != self.ker_size or w != self.ker_size:
        #     c_down = self.resize(c_down)
            # ks = (self.ker_size - h) // 2
            # dim = (ks, ks, ks, ks)
            # c_down = torch.nn.functional.pad(c_down, dim, "constant")
        if self.skip_connect and not self.first_col and c_up is not None:
            x = self.down(torch.cat([c_down, c_up], dim=1)) # concat
        else:
            x = c_down
        # x = c_down
        # x = self.down(c_down)
        return x
class Fusion_Encoder_nofuse_3level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66, downsample_mode='conv') -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        # self.down = nn.Sequential(
        #     nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
        #     # LayerNorm(channels[level]),
        # ) if level in [1, 2] else nn.Identity()

        self.down = nn.Sequential(
            DownSample(channels[level - 1], channels[level], mode=downsample_mode),
            # LayerNorm(channels[level]),
        ) if level in [1, 2] else nn.Identity()

    def forward(self, *args):

        c_down, c_up = args
        x = self.down(c_down)
        return x

class Fusion_Encoder_3level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            # LayerNorm(channels[level]),
        ) if level in [1, 2] else nn.Identity()
        if not first_col:
            # self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()
            self.up = nn.Sequential(nn.Conv2d(channels[level + 1], channels[level] * 4, 1, bias=False),
                                    nn.PixelShuffle(2),
                                    LayerNorm(channels[level])
                                    ) if level in [0, 1] else nn.Identity()

    def forward(self, *args):

        c_down, c_up = args
        x = self.down(c_down)
        if not self.first_col and c_up is not None:
            x = x + self.up(c_up)
        return x
class MFusion_Encoder_3level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66, downsample='downsample', downsample_mode='conv') -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.downsample = downsample
        # self.down = nn.Sequential(
        #     nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
        #     # LayerNorm(channels[level]),
        # ) if level in [1, 2] else nn.Identity()
        self.down = nn.Sequential(
            DownSample(channels[level - 1], channels[level], mode=downsample_mode),
            # LayerNorm(channels[level]),
        ) if level in [1, 2] else nn.Identity()
        if downsample == 'concat':
            self.concat_conv = nn.Sequential(
                nn.Conv2d(channels[level] * 2, channels[level], kernel_size=1, stride=1),
                # LayerNorm(channels[level]),
            )

    def forward(self, *args):

        c_down, c_up = args
        if type(c_up) == int:
            x = self.down(c_down)
        else:

            if self.downsample == 'concat':
                # print(c_down.shape, c_up.shape)
                x = self.concat_conv(torch.cat([self.down(c_down), c_up], dim=1))
            else:
                x = self.down(c_down) + c_up
            # x = self.fuse_block(x1, x2)
        # else:
        #     x1 = self.down(c_down)
        #     x2 = self.down_(c_up)
        #     x = x1 + x2

        # if not self.first_col and c_up is not None:
        #     x = x + self.down_(c_up)
        return x
class MFusion_Encoder_4level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66, downsample='downsample', downsample_mode='conv') -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.downsample = downsample
        self.down = DownSample(channels[level - 1], channels[level], mode=downsample_mode)\
            if level in [1, 2, 3] else nn.Identity()

        if downsample == 'concat':
            # self.down_ = nn.Sequential(
            #     nn.Conv2d(channels[level], channels[level], kernel_size=2, stride=2),
            #     # LayerNorm(channels[level]),
            # ) if level in [0] else nn.Identity()
            self.concat_conv = nn.Sequential(
                nn.Conv2d(channels[level] * 2, channels[level], kernel_size=1, stride=1),
                # LayerNorm(channels[level]),
            )

    def forward(self, *args):

        c_down, c_up = args
        # print(type(c_up), c_up.shape)
        if type(c_up) == int or c_up is None:
            x = self.down(c_down)
        else:

            if self.downsample == 'concat':
                # print(c_down.shape, c_up.shape)
                x = self.concat_conv(torch.cat([self.down(c_down), c_up], dim=1))
            else:
                x = self.down(c_down) + c_up
            # x = self.fuse_block(x1, x2)
        # else:
        #     x1 = self.down(c_down)
        #     x2 = self.down_(c_up)
        #     x = x1 + x2

        # if not self.first_col and c_up is not None:
        #     x = x + self.down_(c_up)
        return x
class MaxMidFusion_Encoder_3level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            # LayerNorm(channels[level]),
        ) if level in [1, 2] else nn.Identity()
        # if not first_col:
            # self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()

        if level in [1, 2]:
            self.down_ = nn.Sequential(
                nn.Conv2d(channels[level-1], channels[level-1] * 4, 1, bias=False),
                nn.PixelShuffle(2)
            )
        else:
            self.down_ = nn.Sequential(
                nn.Conv2d(channels[level], channels[level] * 4, 1, bias=False),
                nn.PixelShuffle(2)
            )

    def forward(self, *args):

        c_max, c_mid = args
        if type(c_mid) == int:
            x = self.down(c_max)
        else:
            x = self.down(c_max + self.down_(c_mid))
        # if not self.first_col and c_up is not None:
        #     x = x + self.down_(c_up)
        return x

class MidMaxFusion_Encoder_3level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.channels = channels
        self.down = nn.Sequential(
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            # LayerNorm(channels[level]),
        ) if level in [1, 2] else nn.Identity()
        # if not first_col:
            # self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()

        if level in [1, 2]:
            # print(channels[level-1])
            self.down_ = nn.Sequential(
                nn.Conv2d(channels[level-1], channels[level-1], kernel_size=2, stride=2),
                # LayerNorm(channels[level]),
            )
        else:
            self.down_ = nn.Sequential(
                nn.Conv2d(channels[level], channels[level], kernel_size=2, stride=2),
                # LayerNorm(channels[level]),
            )

    def forward(self, *args):

        c_mid, c_max = args
        # print(self.channels, c_mid.shape, c_max.shape, self.down_(c_max).shape)
        if type(c_max) == int:
            x = self.down(c_mid)
        else:
            x = self.down(c_mid + self.down_(c_max))
        # if not self.first_col and c_up is not None:
        #     x = x + self.down_(c_up)
        return x

class Fusion_Encoder5Level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            # LayerNorm(channels[level]),
        ) if level in [1, 2, 3, 4] else nn.Identity()
        if not first_col:
            # self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()
            self.up = nn.Sequential(nn.Conv2d(channels[level + 1], channels[level] * 4, 1, bias=False),
                                    nn.PixelShuffle(2),
                                    LayerNorm(channels[level])
                                    ) if level in [0, 1, 2, 3] else nn.Identity()
            # self.fuse = FusNext(channels[level], ffn_expansion_factor=2) if level in [0, 1, 2] else nn.Identity()

    def forward(self, *args):

        c_down, c_up = args

        if self.first_col:
            # print(self.first_col, c_down.shape)
            x = self.down(c_down)
            return x

        if self.level == 4:
            x = self.down(c_down)
        else:
            x = self.up(c_up) + self.down(c_down)
        return x
class SimpleFusion_Encoder5Level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
            nn.Conv2d(channels[level - 1], channels[level], kernel_size=2, stride=2),
            # LayerNorm(channels[level]),
        ) if level in [1, 2, 3, 4] else nn.Identity()


    def forward(self, *args):
        c_down, c_up = args
        x = self.down(c_down)

        return x
class SimpleFusion_Decoder3Level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66, upsample_mode='conv') -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        # print(channels)
        # self.up = nn.Sequential(nn.Conv2d(channels[level], channels[level+1]*4, 1, bias=False),
        #                         nn.PixelShuffle(2),
        #                         # LayerNorm(channels[level])
        #                         ) if level in [0, 1] else nn.Identity()
        self.up = nn.Sequential(UpSample(channels[level], channels[level + 1], mode=upsample_mode),
                                # nn.PixelShuffle(2),
                                # LayerNorm(channels[level])
                                ) if level in [0, 1] else nn.Identity()
    def forward(self, *args):
        c_down, c_up = args
        x = self.up(c_down)

        return x
class SimpleFusion_KernelUpDecoder3Level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        # print(channels)
        self.up = nn.Sequential(nn.Conv2d(channels[level]//2, channels[level+1]*2, 1, bias=False),
                                nn.PixelShuffle(2),
                                # LayerNorm(channels[level])
                                ) if level in [0, 1] else nn.Identity()
        self.alpha = nn.Parameter(
            torch.cat([torch.ones([1, channels[level+1] // 2, 1, 1]), torch.ones([1, channels[level+1] // 2, 1, 1])],
                      dim=1),
            requires_grad=True) if level in [0, 1] else nn.Identity()
        self.beta = nn.Parameter(
            torch.cat(
                [torch.ones([1, channels[level] // 2, 1, 1]) * 1., torch.ones([1, channels[level] // 2, 1, 1])],
                dim=1), requires_grad=True) if level in [0, 1] else nn.Identity()
        self.conv = nn.Sequential(
            nn.Conv2d(channels[level + 1], channels[level + 1], kernel_size=1, stride=1)
        ) if level in [0, 1] else nn.Identity()

    def forward(self, *args):
        c_down, c_up = args
        x = c_down
        if self.level in [0,1]:
            # print(x.shape)
            x = x * self.beta
            ker_mag, ker_phase = torch.chunk(x, 2, dim=1)

            ker_mag, ker_phase = ker_mag, ker_phase * torch.pi
            H, W = ker_mag.shape[-2:]
            kernels_deblur, _ = inverse_logarithmic_ifft_kernel(ker_mag, ker_phase, kernel_size=[H, W], sub=0.)

            x = self.up(kernels_deblur)

            B, C = x.shape[:2]
            _, mag_log, phase = logarithmic_fft_kernel(x, [B, C, H * 2, W * 2])
            x = self.conv(torch.cat([mag_log, phase / torch.pi], dim=1) * self.alpha)
        return x

class SimpleFusion_Decoder5Level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        # print(channels)
        self.up = nn.Sequential(nn.Conv2d(channels[level], channels[level+1]*4, 1, bias=False),
                                nn.PixelShuffle(2),
                                # LayerNorm(channels[level])
                                ) if level in [0, 1, 2, 3] else nn.Identity()


    def forward(self, *args):
        c_down, c_up = args
        # print(c_down.shape, c_up.shape)
        x = self.up(c_down)

        return x
class SimpleFusion_Decoder4Level(nn.Module):
    def __init__(self, level, channels, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66, upsample_mode='conv') -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        # print(channels)
        self.up = UpSample(channels[level], channels[level+1], mode=upsample_mode) if level in [0, 1, 2] else nn.Identity()


    def forward(self, *args):
        c_down, c_up = args
        # print(c_down.shape, c_up.shape)
        x = self.up(c_down)

        return x
class Fusion_Kernel(nn.Module):
    def __init__(self, level, channels, first_col, kernel_size=19, num_heads=1, ffn_expansion_factor=2.66) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
            nn.Conv2d(kernel_size**2, kernel_size**2, kernel_size=2, stride=2),
            # LayerNorm(channels[level]),
        ) if level in [1, 2] else nn.Identity()
        if not first_col:
            # self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()
            self.up = nn.Sequential(nn.Conv2d(kernel_size**2, kernel_size**2 * 4, 1, bias=False),
                                    nn.PixelShuffle(2),
                                    # LayerNorm(channels[level])
                                    ) if level in [0, 1] else nn.Identity()

    def forward(self, *args):

        c_down, c_up = args

        if self.first_col:
            # print(self.first_col, c_down.shape)
            x = self.down(c_down)
            return x

        if self.level == 2:
            x = self.down(c_down)
        else:
            x = self.up(c_up) + self.down(c_down)
        return x
class Fusion_KernelV2(nn.Module):
    def __init__(self, level, channels_down, channels_up, first_col, train_size=[256, 256], num_heads=1, ffn_expansion_factor=2.66, kernel_size=19) -> None:
        super().__init__()

        self.level = level
        self.first_col = first_col
        self.down = nn.Sequential(
            nn.Conv2d(channels_down[level - 1], channels_down[level], kernel_size=2, stride=2),
            # LayerNorm(channels[level]),
        ) if level in [1, 2] else nn.Identity()
        if not first_col:
            # self.up = UpSampleConvnext(1, channels[level+1], channels[level]) if level in [0, 1, 2] else nn.Identity()
            self.up = nn.Sequential(nn.Conv2d(channels_up[level + 1], channels_up[level] * 4, 1, bias=False),
                                    nn.PixelShuffle(2),
                                    LayerNorm(channels_up[level])
                                    ) if level in [0, 1] else nn.Identity()

            # self.fuse = FusNext(channels[level], ffn_expansion_factor=2) if level in [0, 1, 2] else nn.Identity()

    def forward(self, *args):

        c_down, c_up = args

        if self.first_col:
            # print(self.first_col, c_down.shape)
            x = self.down(c_down)
            return x

        if self.level == 2:
            x = self.down(c_down)
        else:
            x = self.up(c_up) + self.down(c_down)
        return x
class Level_Decoder(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        self.fusion = Fusion_Decoder(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)

        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size,
                              dp_rate=dp_rate)
        self.blocks = nn.Sequential(*modules)


    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)

        x = self.blocks(x)
        # print(x.shape)
        return x
class Level_Encoder_KernelPrior(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[96, 96], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        self.kernel_size = kernel_size
        train_size = [train_size[0] // (2 ** level), train_size[1] // (2 ** level)]
        patch_size = [patch_size[0] // (2 ** level), patch_size[1] // (2 ** level)]
        # self.flow = KernelPrior(n_blocks=5, input_size=kernel_size ** 2, hidden_size=25, n_hidden=1, kernel_size=kernel_size)
        # self.Conv_tail = kernel_extra_conv_tail_mean_var(channels[level], self.kernel_size * self.kernel_size)
        # self.fusion = SimpleFusion_Encoder5Level(level, channels, first_col)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        self.fusion = SimpleFusion_Encoder5Level(level, channels, first_col, train_size, num_heads[level],
                                                 ffn_expansion_factor=ffn_expansion_factor)

        modules = get_ker_modules(baseblock[level], level, channels, layers, num_heads, train_size, kernel_size, patch_size)
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c, kernel = args

        x = self.fusion(x, c)
        # x, c = args
        # print(x.shape)
        # B, C, H, W = x.shape
        # kernel_code, kernel_var = self.Conv_tail(kernel)
        # kernel_code = (kernel_code - torch.mean(kernel_code, dim=[2, 3], keepdim=True)) / torch.std(kernel_code,
        #                                                                                             dim=[2, 3],
        #                                                                                             keepdim=True)
        # # code uncertainty
        # sigma = kernel_var
        # kernel_code_uncertain = kernel_code * torch.sqrt(1 - torch.square(sigma)) + torch.randn_like(
        #     kernel_code) * sigma
        # # print(kernel_code_uncertain.shape, x.shape)
        # kernel = generate_k(self.flow, kernel_code_uncertain.reshape(kernel_code.shape[0] * kernel_code.shape[1], -1))
        # kernel = kernel.reshape(kernel_code.shape[0], kernel_code.shape[1], self.kernel_size, self.kernel_size)
        # # kernel_blur = kernel
        #
        # kernel = kernel.permute(0, 2, 3, 1).reshape(B, self.kernel_size * self.kernel_size, H, W)
        x = self.blocks([x, kernel])
        # x = self.fusion(x, c)
        return x

class Level_Decoder_KernelPrior(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[96, 96], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        # print(channels, num_heads, layers)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        train_size = [train_size[0] // (2 ** (4-level)), train_size[1] // (2 ** (4-level))]
        patch_size = [patch_size[0] // (2 ** (4 - level)), patch_size[1] // (2 ** (4 - level))]
        self.fusion = SimpleFusion_Decoder5Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        # if level == 0:
        #     modules = [nn.Identity()]
        # else:
        modules = get_ker_modules(baseblock[level], level, channels, layers, num_heads[level], train_size, kernel_size=kernel_size, patch_size=patch_size)

        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c, kernel = args
        # print(x.shape, c.shape)
        x = self.blocks([x, kernel])
        x = self.fusion(x, c)
        # print(x.shape)
        return x



class Level_Decoder_KernelFeaturePrior(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[96, 96], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        # print(channels, num_heads, layers)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        train_size = [train_size[0] // (2 ** (4-level)), train_size[1] // (2 ** (4-level))]
        patch_size = [patch_size[0] // (2 ** (4 - level)), patch_size[1] // (2 ** (4 - level))]
        self.fusion = SimpleFusion_Decoder5Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        # if level == 0:
        #     modules = [nn.Identity()]
        # else:
        modules = get_ker_modules(baseblock[level], level, channels, layers, num_heads[level], train_size, kernel_size=kernel_size, patch_size=patch_size)

        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c, kernel, kernel_feature = args
        # print(x.shape, c.shape)
        x = self.blocks([x, kernel, kernel_feature])
        x = self.fusion(x, c)
        # print(x.shape)
        return x
class Level_Encoder_KernelFeaturePrior(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[96, 96], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        self.kernel_size = kernel_size
        train_size = [train_size[0] // (2 ** level), train_size[1] // (2 ** level)]
        patch_size = [patch_size[0] // (2 ** level), patch_size[1] // (2 ** level)]
        # self.flow = KernelPrior(n_blocks=5, input_size=kernel_size ** 2, hidden_size=25, n_hidden=1, kernel_size=kernel_size)
        # self.Conv_tail = kernel_extra_conv_tail_mean_var(channels[level], self.kernel_size * self.kernel_size)
        # self.fusion = SimpleFusion_Encoder5Level(level, channels, first_col)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        self.fusion = SimpleFusion_Encoder5Level(level, channels, first_col, train_size, num_heads[level],
                                                 ffn_expansion_factor=ffn_expansion_factor)

        modules = get_ker_modules(baseblock[level], level, channels, layers, num_heads, train_size, kernel_size, patch_size)
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c, kernel, kernel_feature = args

        x = self.fusion(x, c)
        # x, c = args
        # print(x.shape)
        # B, C, H, W = x.shape
        # kernel_code, kernel_var = self.Conv_tail(kernel)
        # kernel_code = (kernel_code - torch.mean(kernel_code, dim=[2, 3], keepdim=True)) / torch.std(kernel_code,
        #                                                                                             dim=[2, 3],
        #                                                                                             keepdim=True)
        # # code uncertainty
        # sigma = kernel_var
        # kernel_code_uncertain = kernel_code * torch.sqrt(1 - torch.square(sigma)) + torch.randn_like(
        #     kernel_code) * sigma
        # # print(kernel_code_uncertain.shape, x.shape)
        # kernel = generate_k(self.flow, kernel_code_uncertain.reshape(kernel_code.shape[0] * kernel_code.shape[1], -1))
        # kernel = kernel.reshape(kernel_code.shape[0], kernel_code.shape[1], self.kernel_size, self.kernel_size)
        # # kernel_blur = kernel
        #
        # kernel = kernel.permute(0, 2, 3, 1).reshape(B, self.kernel_size * self.kernel_size, H, W)
        x = self.blocks([x, kernel, kernel_feature])
        # x = self.fusion(x, c)
        return x
class Level_Encoder_3level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        self.kernel_size = kernel_size
        channels_fusion_down = channels
        channels_fusion_up = [channels[0], channels[1], kernel_size**2]
        self.fusion = Fusion_KernelV2(level, channels_fusion_down, channels_fusion_up, first_col)
        if baseblock == 'fnaf':
            self.blocks = nn.Sequential(*[FNAFBlock(channels[level]) for _ in range(layers[level])])
        elif baseblock == 'fcnaf':
            self.blocks = nn.Sequential(*[FCNAFBlock(channels[level]) for _ in range(layers[level])])
        elif baseblock == 'FFTRes':
            self.blocks = nn.Sequential(*[ResFourier_complex(channels[level]) for _ in range(layers[level])])
        elif baseblock == 'Res':
            self.blocks = nn.Sequential(*[ResBlock1(channels[level]) for _ in range(layers[level])])
        else:
            self.blocks = nn.Sequential(*[NAFBlock(channels[level]) for _ in range(layers[level])])
        # self.Conv_tail = kernel_extra_conv_tail_mean_var(channels[level], self.kernel_size * self.kernel_size)
    def forward(self, *args):
        x, c = args
        x = self.fusion(x, c)
        # x, c = args
        # print(x.shape)

        x = self.blocks(x)
        # kernel_code, kernel_var = self.Conv_tail(x)
        # kernel_code = (kernel_code - torch.mean(kernel_code, dim=[2, 3], keepdim=True)) / torch.std(kernel_code,
        #                                                                                             dim=[2, 3],
        #                                                                                             keepdim=True)
        # # code uncertainty
        # sigma = kernel_var
        # kernel_code_uncertain = kernel_code * torch.sqrt(1 - torch.square(sigma)) + torch.randn_like(
        #     kernel_code) * sigma
        #
        # kernel = generate_k(self.flow, kernel_code_uncertain.reshape(kernel_code.shape[0] * kernel_code.shape[1], -1))
        # kernel = kernel.reshape(kernel_code.shape[0], kernel_code.shape[1], self.kernel_size, self.kernel_size)
        # # kernel_blur = kernel
        #
        # kernel = kernel.permute(0, 2, 3, 1).reshape(B, self.kernel_size * self.kernel_size, H, W)
        # x = self.fusion(x, c)
        return x # , kernel
class Level_Encoder_3level_kernel(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        self.kernel_size = kernel_size
        channels_fusion_down = channels
        channels_fusion_up = [channels[0], channels[1], kernel_size ** 2]
        self.fusion = Fusion_KernelV2(level, channels_fusion_down, channels_fusion_up, first_col)
        if baseblock == 'fnaf':
            self.blocks = nn.Sequential(*[FNAFBlock(channels[level]) for _ in range(layers[level])])
        elif baseblock == 'fcnaf':
            self.blocks = nn.Sequential(*[FCNAFBlock(channels[level]) for _ in range(layers[level])])
        elif baseblock == 'FFTRes':
            self.blocks = nn.Sequential(*[ResFourier_complex(channels[level]) for _ in range(layers[level])])
        else:
            self.blocks = nn.Sequential(*[NAFBlock(channels[level]) for _ in range(layers[level])])
        self.flow = KernelPrior(n_blocks=5, input_size=kernel_size ** 2, hidden_size=25, n_hidden=1, kernel_size=kernel_size)
        self.Conv_tail = kernel_extra_conv_tail_mean_var(channels[level], self.kernel_size * self.kernel_size)
    def forward(self, *args):
        x, c = args
        x = self.fusion(x, c)
        # x, c = args
        # print(x.shape)

        x = self.blocks(x)
        B, C, H, W = x.shape
        kernel_code, kernel_var = self.Conv_tail(x)
        kernel_code = (kernel_code - torch.mean(kernel_code, dim=[2, 3], keepdim=True)) / torch.std(kernel_code,
                                                                                                    dim=[2, 3],
                                                                                                    keepdim=True)
        # code uncertainty
        sigma = kernel_var
        kernel_code_uncertain = kernel_code * torch.sqrt(1 - torch.square(sigma)) + torch.randn_like(
            kernel_code) * sigma
        # print(kernel_code_uncertain.shape, x.shape)
        kernel, _ = self.flow(kernel_code_uncertain.reshape(kernel_code.shape[0] * kernel_code.shape[1], -1))
        # kernel = generate_k(self.flow, kernel_code_uncertain.reshape(kernel_code.shape[0] * kernel_code.shape[1], -1))
        # kernel = generate_k_forward(self.flow, kernel_code_uncertain.reshape(kernel_code.shape[0] * kernel_code.shape[1], -1))
        kernel = kernel.reshape(kernel_code.shape[0], kernel_code.shape[1], self.kernel_size, self.kernel_size).contiguous()
        # kernel_blur = kernel

        kernel = kernel.permute(0, 2, 3, 1).reshape(B, self.kernel_size * self.kernel_size, H, W).contiguous()

        kernel = torch.softmax(kernel, dim=1)
        return kernel # , kernel
class Level_Decoder_3level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        self.kernel_size = kernel_size
        self.fusion = Fusion_Decoder_3level(level, channels, first_col)
        if baseblock == 'fnaf':
            self.blocks = nn.Sequential(*[FNAFBlock(channels[level]) for _ in range(layers[level])])
        elif baseblock == 'fcnaf':
            self.blocks = nn.Sequential(*[FCNAFBlock(channels[level]) for _ in range(layers[level])])
        else:
            self.blocks = nn.Sequential(*[NAFBlock(channels[level]) for _ in range(layers[level])])
        # self.Conv_tail = kernel_extra_conv_tail_mean_var(channels[level], self.kernel_size * self.kernel_size)
    def forward(self, *args):
        x, c = args
        x = self.fusion(x, c)
        # x, c = args
        # print(x.shape)

        x = self.blocks(x)
        return x # , kernel
class Level_Encoder_5(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        self.fusion = Fusion_Encoder5Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        if baseblock == 'naf' or level == 4:
            modules = [NAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
        elif baseblock == 'dcnv3':
            modules = [DCNv3Block(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
        elif baseblock == 'fnaf':
            modules = [FNAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
        elif baseblock == 'fcnaf':
            modules = [FCNAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
        elif baseblock == 'fdcnv3':
            modules = [FDCNv3Block(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
        elif baseblock == 'fftformer':
            modules = [fftformerblock(channels[level], ffn_expansion_factor=2, att=True) for i in
                       range(layers[level])]
        elif baseblock == 'Ffftformer':
            modules = [Ffftformer(channels[level], ffn_expansion_factor=2, att=True) for i in
                       range(layers[level])]
        elif baseblock == 'Fattn_FreqLC':
            modules = [
                AttnBlock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2, cs='channel_mlp')
                for
                i in range(layers[level])]
        elif baseblock == 'loformer_SpaLC':
            modules = [
                loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2,
                              cs='channel_nodct') for
                i in range(layers[level])]
        elif baseblock == 'loformer_SpaLS':
            modules = [
                loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2,
                              cs='spatial_nodct') for
                i in range(layers[level])]
        elif baseblock == 'loformer_FreqLC':
            modules = [
                loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2, cs='channel_mlp')
                for
                i in range(layers[level])]
        elif baseblock == 'loformer_FreqLS':
            modules = [
                loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2, cs='spatial_mlp')
                for
                i in range(layers[level])]
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)

        x = self.blocks(x)
        # print(x.shape)
        return x
class FourierAttention(nn.Module):
    def __init__(self, dim, bias):
        super(FourierAttention, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm2d(dim * 2)

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)
        h, w = q.shape[-2:]
        # q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        # k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q.float())
        k_fft = torch.fft.rfft2(k.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(h, w))
        # out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
        #                 patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output

class FourierAttentionReLU(nn.Module):
    def __init__(self, dim, bias, act_method=nn.ReLU):
        super(FourierAttentionReLU, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.act = act_method()
        self.norm = LayerNorm2d(dim * 2)

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)
        h, w = q.shape[-2:]
        # q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        # k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q.float())
        k_fft = torch.fft.rfft2(k.float())
        k_fft = self.act(k_fft.real)+1.0j*self.act(k_fft.imag) - 0.5 * k_fft
        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(h, w))
        # out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
        #                 patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output
class FourierConvAttention(nn.Module):
    def __init__(self, dim, bias):
        super(FourierConvAttention, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm2d(dim * 2)

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        b, c, h, w, _, _ = k_patch.shape
        k_patch = torch.softmax(k_patch.contiguous().view(b, c, h, w, self.patch_size*self.patch_size), dim=-1).view(b, c, h, w, self.patch_size, self.patch_size)
        k_patch = torch.fft.ifftshift(k_patch, dim=[-2, -1])
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output

class FourierKerAttention(nn.Module):
    def __init__(self, dim, bias):
        super(FourierKerAttention, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm2d(dim * 2)

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        # q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        # k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        b, c, h, w = k.shape
        ker = torch.softmax(k.contiguous().view(b, c, h * w), dim=-1).view(b, c, h, w)
        ker = torch.fft.ifftshift(ker, dim=[-2, -1])
        q_fft = torch.fft.rfft2(q.float())
        k_fft = torch.fft.rfft2(ker.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(h, w))
        # out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
        #                 patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output
class FourierFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, dim_out=None):

        super(FourierFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        if dim_out is None:
            dim_out = dim
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim_out, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x
class FourierformerBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias', att=False):
        super(FourierformerBlock, self).__init__()

        self.att = att
        # if self.att:
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = FourierKerAttention(dim, bias)

        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FourierFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # if self.att:
        x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x
class kernel_adapters(nn.Module):
    def __init__(self, widths):
        super(kernel_adapters, self).__init__()

        self.adapter = nn.ModuleList()
        for i in range(3):
            self.adapter.append(kernel_reblur(widths[i], widths[i]))
        # self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch*2, 1, 1), requires_grad=True)
    def forward(self, inputs):
        eg0, eg1, eg2, ker0, ker1, ker2 = inputs
        eg0 = self.adapter[0](eg0, ker0)
        eg1 = self.adapter[1](eg1, ker1)
        eg2 = self.adapter[2](eg2, ker2)
        return eg0, eg1, eg2
def get_modules(baseblock, level, channels, layers, num_heads, train_size, patch_size=None, kernel_size=19, dp_rate=0., sub_idx=0, d_state=8, inner_rank=32, num_tokens=64, mlp_ratio=1.0):
    spatial_size = train_size # [train_size[0]//(2**level), train_size[1]//(2**level)]
    # print(spatial_size)
    # print(train_size)
    # print(dp_rate)
    # print(layers)
    if baseblock == 'naf':
        modules = [NAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'naf_patch8':
        modules = [NAFBlock_patch(channels[level],patch_size=[8, 8]) for i in range(layers[level])]
    elif baseblock == 'ALGBlock':
        modules = [ALGBlock(channels[level], train_size=train_size, patch_size=patch_size) for i in range(layers[level])]
    elif baseblock == 'naf_nosca':
        modules = [NAFBlock_nosca(channels[level],patch_size=patch_size) for i in range(layers[level])]
    elif baseblock == 'naf_patch':
        modules = [NAFBlock_patch(channels[level],patch_size=patch_size) for i in range(layers[level])]
    elif baseblock == 'naf_patch_grid':
        modules = [NAFBlock_patch_grid(channels[level], patch_size=patch_size) for i in range(layers[level])]
    elif baseblock == 'localnaf':
        modules = [LocalNAFBlock(channels[level], train_size) for i in range(layers[level])]
    elif baseblock == 'locallinearnaf':
        modules = [LocalLinearNAFBlock(channels[level], train_size) for i in range(layers[level])]
    elif baseblock == 'naf_complex':
        modules = [ComplexNAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'localnaf_fouriercomplex':
        modules = [LocalFourierComplexNAFBlock(channels[level], train_size, DW_Fourier=1) for i in range(layers[level])]
    elif baseblock == 'localnaf_fourierconcatchannl':
        modules = [LocalFourierConcatChannelNAFBlock(channels[level], train_size, DW_Fourier=1) for i in range(layers[level])]
    elif baseblock == 'localnaf_fourierconcatchannldwconv':
        modules = [
            LocalFourierConcatChannelDWConvNAFBlock(channels[level], train_size, drop_out_rate=dp_rate[level], DW_Fourier=1)
            for i in range(layers[level])]
    elif baseblock == 'localnaf_FourierBatch':
        modules = [LocalFourierConcatBatchNAFBlock(channels[level], train_size, drop_out_rate=dp_rate[level], DW_Fourier=1) for i in range(layers[level])]
    elif baseblock == 'localnaf_FourierBatch_patch':
        modules = [LocalFourierConcatBatchNAFBlock(channels[level], train_size, drop_out_rate=dp_rate[level], DW_Fourier=1, patch_size=train_size) for i in range(layers[level])]
    elif baseblock == 'naf_FourierBatch':
        modules = [FourierConcatBatchNAFBlock(channels[level], train_size, drop_out_rate=dp_rate[level], DW_Fourier=1) for i in range(layers[level])]
    elif baseblock == 'localnaf_FourierSplit':
        modules = [
            LocalFourierConcatSplitNAFBlock(channels[level], train_size, drop_out_rate=dp_rate[level], DW_Fourier=1) for
            i in range(layers[level])]
    elif baseblock == 'LocalNAFEVSBlock_FourierSplit':
        modules = [LocalNAFEVSBlock_FourierSplit(channels[level], train_size=train_size, DW_Fourier=1, patch_size=patch_size) for i in range(layers[level])]
    elif baseblock == 'localnaf_fouriersplit_patch':
        # print(level, patch_size)
        modules = [
            LocalFourierConcatSplitNAFBlock(channels[level], train_size, drop_out_rate=dp_rate[level], DW_Fourier=1, window_size_fft=patch_size) for
            i in range(layers[level])]
    elif baseblock == 'naf_FourierSplit':
        # print(level, patch_size)
        modules = [
            FourierConcatSplitNAFBlock(channels[level], train_size, drop_out_rate=dp_rate[level], DW_Fourier=1) for
            i in range(layers[level])]
    elif baseblock == 'localnaf_fourierconcatsplitdwconv':
        modules = [
            LocalFourierConcatSplitDwConvNAFBlock(channels[level], train_size, drop_out_rate=dp_rate[level], DW_Fourier=1) for
            i in range(layers[level])]
    elif baseblock == 'localnaf_fourierconcatsplitdw2':
        modules = [
            LocalFourierConcatSplitNAFBlock(channels[level], train_size, drop_out_rate=dp_rate[level], DW_Fourier=2) for
            i in range(layers[level])]
    elif baseblock == 'nafwosg':
        modules = [NAFwoSGBlock(channels[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'nafwoffn':
        modules = [NAFwoFFNBlock(channels[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'nafwoscaffn':
        modules = [NAFwoSCAFFNBlock(channels[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'nafgeluwoffn':
        modules = [NAFGELUwoFFNBlock(channels[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'nafwosgsca':
        modules = [NAFwoSGSCABlock(channels[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'nafwosgscaffn':
        modules = [NAFwoSGSCAFFNBlock(channels[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'nafwosgscawmambaffn':
        modules = [NAFwoSGSCAFFNwMambaBlock(channels[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'kerv1BN':
        modules = [KerBNBlock(channels[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'kerv1LN':
        modules = [KerLNBlock(channels[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'kerv2LN':
        modules = [Kerv2LNBlock(channels[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'AVSBlock':
        modules = [AVSBlock(channels[level], d_state=d_state, input_resolution=train_size, inner_rank=inner_rank, num_tokens=num_tokens, mlp_ratio=mlp_ratio) for i in range(layers[level])]
    elif baseblock == 'ASSMBlock':
        modules = [ASSMBlock(channels[level], ffn_expansion_factor=3, d_state=8, train_size=train_size, att=True) for i in range(layers[level])]
    elif baseblock == 'NAFEVSFourierBlock':
        modules = [NAFEVSFourierBlock(channels[level], train_size=train_size) for i in range(layers[level])]
    elif baseblock == 'EVS_nogrid':
        if level > 0:
            for i in range(level):
                sub_idx = sub_idx + layers[i]
        modules = [EVS_nogrid(channels[level], ffn_expansion_factor=3, idx=i+sub_idx, att=True) for i in range(layers[level])]
    elif baseblock == 'EVS_noflip':
        modules = [EVS_nogrid(channels[level], ffn_expansion_factor=3, idx=-1, att=True, patch_size=patch_size) for i in range(layers[level])]
    elif baseblock == 'EVS':
        modules = [EVS_nogrid(channels[level], ffn_expansion_factor=3, idx=i, att=True, patch_size=patch_size) for i in range(layers[level])]
    elif baseblock == 'EVS_noflip_cfft':
        modules = [EVS_nogrid(channels[level], ffn_expansion_factor=3, idx=-1, att=True, patch_size=patch_size, global_fft=True) for i in range(layers[level])]
    elif baseblock == 'EVS_cfft':
        modules = [EVS_nogrid(channels[level], ffn_expansion_factor=3, idx=i, att=True, patch_size=patch_size, global_fft=True) for i in range(layers[level])]
    elif baseblock == 'NAFEVSBlock':
        modules = [NAFEVSBlock(channels[level], train_size=train_size, idx=i) for i in range(layers[level])]
    elif baseblock == 'NAFEVSBlock_patch':
        modules = [NAFEVSBlock(channels[level], train_size=train_size, patch_size=patch_size, idx=i) for i in range(layers[level])]
    elif baseblock == 'NAFEVSBlock_noflip':
        modules = [NAFEVSBlock(channels[level], train_size=train_size, idx=-1) for i in range(layers[level])]
    elif baseblock == 'LocalNAFEVSBlock':
        modules = [LocalNAFEVSBlock(channels[level], train_size=train_size, idx=i) for i in range(layers[level])]
    elif baseblock == 'LocalNAFEVSBlock_patch':
        modules = [LocalNAFEVSBlock(channels[level], train_size=train_size, patch_size=patch_size, idx=i) for i in range(layers[level])]
    elif baseblock == 'LocalNAFEVSBlock_noflip':
        modules = [LocalNAFEVSBlock(channels[level], train_size=train_size, idx=-1) for i in range(layers[level])]
    # elif baseblock == 'NAFEVSBlock':
    #     modules = [LocalNAFEVSBlock(channels[level], train_size=train_size, idx=i) for i in range(layers[level])]
    elif baseblock == 'LocalNAFEVSBlock_patch_noflip':
        modules = [LocalNAFEVSBlock(channels[level], train_size=train_size, patch_size=patch_size, idx=-1) for i in range(layers[level])]
    elif baseblock == 'EVSSBlock':
        modules = [EVSSBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'EVSBlock':
        modules = [EVSBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'EVSSNAFBlock':
        modules = [EVSSNAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'EVSS':
        modules = [EVSS(channels[level]) for i in range(layers[level])]
    elif baseblock == 'LocalNAFEVSBlock':
        modules = [LocaNAFEVSBlock(channels[level], train_size=train_size, idx=i) for i in range(layers[level])]
    elif baseblock == 'LocalNAFEVSBlock_patch':
        modules = [LocaNAFEVSBlock(channels[level], train_size=train_size, patch_size=patch_size, idx=i) for i in range(layers[level])]
    # elif baseblock == 'LocalNAFEVSBlock_patch':
    #     modules = [LocalNAFEVSBlock(channels[level], train_size=train_size, patch_size=patch_size) for i in range(layers[level])]
    elif baseblock == 'LocalNAFEVSDFFNBlock':
        modules = [LocalNAFEVSDFFNBlock(channels[level], train_size=train_size) for i in range(layers[level])]
    elif baseblock == 'LocalNAFSpaGSBlock':
        # print(num_heads, level)
        modules = [LocalNAFSpaGSBlock(channels[level], train_size=train_size, num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'nafRes':
        modules = [NAFResBlock(channels[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'LNResBlock':
        modules = [LNResBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'LNResBlock1x1':
        modules = [LNResBlock1x1(channels[level]) for i in range(layers[level])]
    elif baseblock == 'ResBlock':
        modules = [ResBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'fouriernaf':
        modules = [FourierNAFBlock(channels[level], num_heads=num_heads[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'fourierknaf':
        modules = [FourierKerNAFBlock(channels[level], num_heads=num_heads[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'logfouriernaf':
        modules = [LogFourierNAFBlock(channels[level], num_heads=num_heads[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'fourierdw2naf':
        modules = [FourierNAFBlock(channels[level], num_heads=num_heads[level], drop_out_rate=dp_rate[level], DW_Fourier=2) for i in range(layers[level])]
    elif baseblock == 'fourierResnaf':
        modules = [FourierNAFResBlock(channels[level], num_heads=num_heads[level], drop_out_rate=dp_rate[level]) for i in
                   range(layers[level])]
    elif baseblock == 'fourierSCAnaf':
        modules = [FourierSCANAFBlock(channels[level], num_heads=num_heads[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'fouriernaf2act':
        modules = [Fourier2ActNAFBlock(channels[level], num_heads=num_heads[level], drop_out_rate=dp_rate[level]) for i in
                   range(layers[level])]
    elif baseblock == 'fouriernaf2':
        modules = [FourierNAFBlock2(channels[level], num_heads=num_heads[level], drop_out_rate=dp_rate[level]) for i in range(layers[level])]
    elif baseblock == 'naf1x1':
        modules = [NAF1x1Block(channels[level]) for i in range(layers[level])]
    elif baseblock == 'mamba_naf':
        modules = [MambaNAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'mambaforward_naf':
        modules = [MambaForwardNAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'fmambaforward_naf':
        modules = [FMambaForwardNAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'fdmambaforward_naf':
        modules = [FdownMambaForwardNAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'fsmambaforward_naf':
        modules = [FshuffleMambaForwardNAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'flocalmambaforward_naf':
        modules = [FLocalMambaForwardNAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'mambav2_naf':
        modules = [MambaV2NAFBlock(channels[level], num_heads=num_heads) for i in range(layers[level])]
    elif baseblock == 'atdmamba_naf':
        modules = [ATDMambaNAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'mamba_f1naf':
        modules = [MambaF1NAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'vmamba_f1naf':
        modules = [VMambaF1NAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'naf_local':
        modules = [NAFBlock_local(channels[level], train_size=spatial_size, patch_size=patch_size) for i in range(layers[level])]
    elif baseblock == 'dcnv3':
        modules = [DCNv3Block(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'FFTRes':
        modules = nn.Sequential(*[ResFourier_complex(channels[level]) for _ in range(layers[level])])
    elif baseblock == 'naf_conv':
        modules = nn.Sequential(
            *[NAFBlock_conv(channels[level], feature_ch=channels[level]) for _ in range(layers[level])])
    elif baseblock == 'naf_deblur':
        modules = nn.Sequential(
            *[NAFBlock_deblur(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'Resblock':
        modules = nn.Sequential(*[ResBlock1(channels[level]) for _ in range(layers[level])])
    elif baseblock == 'fnaf':
        modules = [FNAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'EVSSblock':
        modules = [EVSSBlock(channels[level], ffn_expansion_factor=3, att=True) for i in
                   range(layers[level])]
    elif baseblock == 'f1x1conv2dnaf':
        modules = [F1x1conv2dNAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'fourierx2_naf':
        modules = [Fourierx2NAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'f1naf':
        modules = [F1NAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'patch_f1naf':
        modules = [F1NAFBlock(channels[level], num_heads=num_heads[level], window_size_fft=patch_size) for i in range(layers[level])]
    elif baseblock == 'fc1naf':
        modules = [FC1NAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'fc1naf_dffn':
        modules = [FC1NAFBlockDFFN(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'fc1naf_ffn':
        modules = [FC1NAFBlockFFN(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'fc1xnaf':
        modules = [FC1XNAFBlock(channels[level], num_heads=num_heads[level], train_size=train_size) for i in range(layers[level])]
    elif baseblock == 'fc1senaf':
        modules = [FC1SENAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'fc2naf':
        modules = [FC2NAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'fcnaf':
        modules = [FCNAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'focnaf':
        modules = [FOCNAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'flc1naf':
        modules = [FLC1NAFBlock(channels[level], num_heads=num_heads[level], kernel_size=kernel_size) for i in range(layers[level])]
    elif baseblock == 'fcanaf':
        modules = [FCANAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'fourconv':
        modules = [fft_bench_complex_conv(channels[level], dw=2, bias=True, act_method=nn.GELU) for i in
                   range(layers[level])]
    elif baseblock == 'four_defconv':
        modules = [fft_bench_deform_conv(channels[level], dw=2, bias=True, act_method=nn.GELU, num_heads=num_heads[level]) for i in
                   range(layers[level])]
    elif baseblock == 'four_defconv2':
        modules = [fft_bench_deform_conv2(channels[level], dw=2, bias=True, act_method=nn.GELU, num_heads=num_heads[level]) for i in
                   range(layers[level])]
    elif baseblock == 'fourmlp':
        modules = [fft_bench_mlp(channels[level], dw=2, bias=True, act_method=nn.GELU, spatial_size=spatial_size) for i in
                   range(layers[level])]
    elif baseblock == 'fourconv_scaleup':
        modules = [fft_bench_complex_conv_with_scale_up(channels[level], dim=channels[level]//num_heads[level],
                                                        scale_up=num_heads[level],
                                                        dw=2, bias=True, act_method=nn.GELU) for i in
                   range(layers[level])]
    elif baseblock == 'funaf':
        modules = [FUNAFBlock(channels[level], scale_up=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'fdcnv3':
        modules = [FDCNv3Block(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
    elif baseblock == 'fftformer':
        modules = [fftformerblock(channels[level], ffn_expansion_factor=3, att=True) for i in
                   range(layers[level])]
    elif baseblock == 'fftformer_noattn':
        modules = [fftformerblock(channels[level], ffn_expansion_factor=3, att=False) for i in
                   range(layers[level])]
    elif baseblock == 'fourierformer':
        modules = [FourierformerBlock(channels[level], ffn_expansion_factor=2, att=True) for i in
                   range(layers[level])]
    elif baseblock == 'Ffftformer':
        modules = [Ffftformer(channels[level], ffn_expansion_factor=2, att=True) for i in
                   range(layers[level])]
    elif baseblock == 'Fattn_FreqLC':
        modules = [
            AttnBlock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2, cs='channel_mlp')
            for
            i in range(layers[level])]
    elif baseblock == 'loformer_SpaGS':
        modules = [
            loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2,
                          cs='global_spatial_nodct') for
            i in range(layers[level])]
    elif baseblock == 'loformer_FreqGS':
        modules = [
            loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2,
                          cs='global_spatial') for
            i in range(layers[level])]
    elif baseblock == 'loformer_SpaCS':
        modules = [
            loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2,
                          cs='global_channel_nodct') for
            i in range(layers[level])]
    elif baseblock == 'loformer_SpaGS_DFFN':
        modules = [
            loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2, train_size=spatial_size,
                          cs='global_spatial_nodct', ffn='DFFN') for
            i in range(layers[level])]
    elif baseblock == 'loformer_SpaGS_GDFFN':
        modules = [
            loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2, train_size=spatial_size,
                          cs='global_spatial_nodct', ffn='GDFFN') for
            i in range(layers[level])]
    elif baseblock == 'loformer_SpaLC':
        modules = [
            loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2,
                          cs='channel_nodct') for
            i in range(layers[level])]
    elif baseblock == 'loformer_SpaLS':
        modules = [
            loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2,
                          cs='spatial_nodct') for
            i in range(layers[level])]
    elif baseblock == 'loformer_FreqLC':
        modules = [
            loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2, cs='channel_mlp')
            for
            i in range(layers[level])]
    elif baseblock == 'loformer_FreqLS':
        modules = [
            loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2, cs='spatial_mlp')
            for
            i in range(layers[level])]
    else:
        modules = [nn.Identity()]
    return modules


def get_ker_modules(baseblock, level, channels, layers, num_heads, train_size, kernel_size, patch_size=None):
    spatial_size = train_size # [train_size[0]//(2**level), train_size[1]//(2**level)]
    # print(spatial_size)
    # print(baseblock)
    if baseblock == 'fnaf':
        modules = nn.Sequential(
            *[FNAFBlock_kernel_feature(channels[level], feature_ch=channels[level]) for _ in range(layers[level])])
    elif baseblock == 'maskedmamba_naf':
        modules = [MaskMambaNAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'masked_f1naf':
        modules = [MaskF1NAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'maskedsca_f1naf':
        modules = [MaskSCAF1NAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'maskedmamba_f1naf':
        modules = [MaskMambaF1NAFBlock(channels[level]) for i in range(layers[level])]
    elif baseblock == 'maskedmamba_patch_f1naf':
        modules = [MaskMambaPatchF1NAFBlock(channels[level], patch_size=patch_size) for i in range(layers[level])]
    elif baseblock == 'patch_maskedmamba_f1naf':
        modules = [PatchMaskMambaF1NAFBlock(channels[level], patch_size=patch_size) for i in range(layers[level])]
    elif baseblock == 'fcnaf':
        modules = nn.Sequential(
            *[FCNAFBlock_kernel_feature(channels[level], feature_ch=channels[level]) for _ in range(layers[level])])
    elif baseblock == 'naf':
        modules = nn.Sequential(
            *[NAFBlock_kernel_feature(channels[level], feature_ch=channels[level]) for _ in range(layers[level])])
    elif baseblock == 'naf_reblur':
        modules = nn.Sequential(
            *[NAFBlock_reblur(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'naf_patch_reblur':
        modules = nn.Sequential(
            *[NAFBlock_patch_reblur(channels[level], kernel_size=kernel_size, patch_size=patch_size) for _ in range(layers[level])])
    elif baseblock == 'naf_logdeblur':
        modules = nn.Sequential(
            *[NAFBlock_logdeblur(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'f1naf_reblur':
        modules = nn.Sequential(
            *[F1NAFBlock_reblur(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'naf_reblur_attn':
        modules = nn.Sequential(
            *[NAFBlock_reblur_attn(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'naf_reblur_attn_feature':
        modules = nn.Sequential(
            *[NAFBlock_reblur_attn_feature(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'naf_reblur_feature':
        modules = nn.Sequential(
            *[NAFBlock_reblur_feature(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'naf_reblur_softmax':
        modules = nn.Sequential(
            *[NAFBlock_reblur_softmax(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'naf_deblur':
        modules = nn.Sequential(
            *[NAFBlock_deblur(channels[level], kernel_size=kernel_size, train_size=spatial_size) for _ in range(layers[level])])
    elif baseblock == 'naf_mambadeblur':
        modules = nn.Sequential(
            *[NAFBlock_mamba_deblur(channels[level], kernel_size=kernel_size, train_size=spatial_size) for _ in range(layers[level])])
    elif baseblock == 'naf_deformdeblur':
        modules = nn.Sequential(
            *[NAFBlock_deformdeblur(channels[level], kernel_size=kernel_size, train_size=spatial_size) for _ in range(layers[level])])
    elif baseblock == 'naf_patch_deblur':
        modules = nn.Sequential(
            *[NAFBlock_patch_deblur(channels[level], kernel_size=kernel_size, train_size=spatial_size) for _ in range(layers[level])])
    elif baseblock == 'naf_patch_grid_deblur':
        modules = nn.Sequential(
            *[NAFBlock_patch_grid_deblur(channels[level], kernel_size=kernel_size, train_size=spatial_size, patch_size=patch_size) for _ in range(layers[level])])
    elif baseblock == 'naf_local_patch_grid_deblur':
        modules = nn.Sequential(
            *[NAFBlock_local_patch_grid_deblur(channels[level], kernel_size=kernel_size, train_size=spatial_size,
                                         patch_size=patch_size) for _ in range(layers[level])])

    elif baseblock == 'naf_noker_deblur':
        modules = nn.Sequential(
            *[NAFBlock_noker_deblur(channels[level], kernel_size=kernel_size, train_size=spatial_size) for _ in range(layers[level])])
    elif baseblock == 'fcnaf_deblur':
        modules = nn.Sequential(
            *[FCNAFBlock_deblur(channels[level], kernel_size=kernel_size, train_size=spatial_size) for _ in range(layers[level])])
    elif baseblock == 'fc1naf_deblur':
        modules = nn.Sequential(
            *[FC1NAFBlock_deblur(channels[level], kernel_size=kernel_size, train_size=spatial_size) for _ in range(layers[level])])
    elif baseblock == 'naf_deblur_fourierconv':
        modules = nn.Sequential(
            *[NAFBlock_deblur_fourierconv(channels[level], kernel_size=kernel_size, train_size=spatial_size) for _ in range(layers[level])])
    elif baseblock == 'naf_deblur_feature':
        modules = nn.Sequential(
            *[NAFBlock_deblur_feature(channels[level], kernel_size=kernel_size, train_size=spatial_size) for _ in range(layers[level])])
    elif baseblock == 'naf_deblur_attn':
        modules = nn.Sequential(
            *[NAFBlock_deblur_attn(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'kernelfnaf':
        modules = nn.Sequential(
            *[FNAFBlock_kernel(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'kernelfcnaf':
        modules = nn.Sequential(
            *[FCNAFBlock_kernel(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'kernelfc1naf':
        modules = nn.Sequential(
            *[FC1NAFBlock_kernel(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'kernelnaf':
        modules = nn.Sequential(
            *[NAFBlock_kernel(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'kernelnaf_deblur':
        modules = nn.Sequential(
            *[NAFBlock_kernel_deblur(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'kernelnaf_deblurX':
        modules = nn.Sequential(
            *[NAFBlock_kernel_deblurX(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'kernelnaf_deblurXY':
        modules = nn.Sequential(
            *[NAFBlock_kernel_deblurXY(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'kernelnaf_xdeblur':
        modules = nn.Sequential(
            *[NAFBlock_kernel_xdeblur(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    elif baseblock == 'kernelnaf_xattn_deblur':
        modules = nn.Sequential(
            *[NAFBlock_kernel_xattn_deblur(channels[level], kernel_size=kernel_size, num_heads=num_heads) for _ in range(layers[level])])
    elif baseblock == 'kernelnaf_xkattn_deblur':
        modules = nn.Sequential(
            *[NAFBlock_kernel_xkattn_deblur(channels[level], kernel_size=kernel_size, num_heads=num_heads) for _ in
              range(layers[level])])
    elif baseblock == 'kernelnaf_kfattn_deblur':
        modules = nn.Sequential(
            *[NAFBlock_kernel_kfattn_deblur(channels[level], kernel_size=kernel_size, num_heads=num_heads) for _ in
              range(layers[level])])
    elif baseblock == 'kernelnaf_phasedeblur':
        modules = nn.Sequential(
            *[NAFBlock_kernel_phasedeblur(channels[level], kernel_size=kernel_size) for _ in range(layers[level])])
    else:
        modules = [nn.Identity()]
    return modules
class Level_SimpleEncoder_5(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[96, 96], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        train_size = [train_size[0]//(2**level), train_size[1]//(2**level)]
        patch_size = [patch_size[0]//(2**level), patch_size[1]//(2**level)]
        self.fusion = SimpleFusion_Encoder5Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        # if level == 4:
        #     modules = [NAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
        # else:
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, kernel_size=kernel_size, patch_size=patch_size)

        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)

        x = self.blocks(x)
        # print(x.shape)
        return x

class Level_SimpleFusionEncoder_5(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[96, 96], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        train_size = [train_size[0]//(2**level), train_size[1]//(2**level)]
        patch_size = [patch_size[0]//(2**level), patch_size[1]//(2**level)]
        self.fusion = Fusion_Encoder5Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        # if level == 4:
        #     modules = [NAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
        # else:
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, kernel_size=kernel_size, patch_size=patch_size)

        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)

        x = self.blocks(x)
        # print(x.shape)
        return x
class Level_SimpleDecoder_5(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[96, 96], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        # print(channels, num_heads, layers)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        train_size = [train_size[0] // (2 ** (4-level)), train_size[1] // (2 ** (4-level))]
        # print(patch_size)
        patch_size = [patch_size[0] // (2 ** (4-level)), patch_size[1] // (2 ** (4-level))]
        self.fusion = SimpleFusion_Decoder5Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        # if level == 0:
        #     modules = [nn.Identity()]
        # else:
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, kernel_size=kernel_size, patch_size=patch_size)

        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args

        x = self.blocks(x)

        x = self.fusion(x, c)
        # print(x.shape)
        return x

class Level_SimpleDyDecoder_5(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[96, 96], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        # print(channels, num_heads, layers)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        train_size = [train_size[0] // (2 ** (4-level)), train_size[1] // (2 ** (4-level))]
        # print(patch_size)
        patch_size = [patch_size[0] // (2 ** (4-level)), patch_size[1] // (2 ** (4-level))]
        self.fusion = SimpleFusion_Decoder5Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        # if level == 0:
        #     modules = [nn.Identity()]
        # else:
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, kernel_size=kernel_size, patch_size=patch_size)
        self.exit_threshold = 0.5
        self.blocks = nn.Sequential(*modules)
        # self.simple_blocks = NAFBlock(channels[level])
        self.dropout = nn.Dropout(p=1.)
    def forward(self, *args):
        x, c, ker_sel = args

        x = self.blocks(x)
        x = self.fusion(x, c)
        # print(x.shape)
        return x
    def forward_x(self, *args):
        x, c, ker_sel = args


        # print(x.shape, c.shape)
        # eid = torch.where(torch.arange(0, x.shape[0], device=x.device) != pid)[0]
        # print(eid)
        pid = torch.where(ker_sel < self.exit_threshold)[0]
        eid = torch.where(ker_sel >= self.exit_threshold)[0]
        # eid = torch.where(exit_index >= 0.0)[0]
        # pid = torch.where(exit_index < 0.0)[0]
        if self.training:
            y = x.clone()
            b, c, h, w = x.shape

            scale_eid = False
            scale_pid = False
            if len(eid) == 0:
                _, eid = torch.topk(ker_sel, 1, dim=0, sorted=False)
                scale_eid = True
                # _, pid = torch.topk(-ker_sel, b-1, dim=0, sorted=False)
                # y = torch.scatter_add(y, 0, eid, y2)
            if len(pid) == 0:
                scale_pid = True
                # _, eid = torch.topk(ker_sel, b-1, dim=0, sorted=False)
                _, pid = torch.topk(-ker_sel, 1, dim=0, sorted=False)
            eid = eid.flatten()
            pid = pid.flatten()
            # print(len(eid), pid)
            # if len(eid) > 0:
                # eid = eid.view(-1, 1, 1, 1)
            eid = repeat(eid, 'b -> b c h w', c=c, h=h, w=w)
            y2 = torch.gather(x, 0, eid)
            y2 = self.simple_blocks(y2)
            # y = y.scatter_(0, eid, y2)
            pid = repeat(pid, 'b -> b c h w', c=c, h=h, w=w)
            y1 = torch.gather(x, 0, pid)
            # print(y1.shape)
            y1 = self.blocks(y1)
            # eid = torch.arange(0, x.shape[0], device=x.device) != pid
            # y = y.scatter_(0, pid, y1)
            if scale_pid:
                y = y.scatter_(0, eid, y2)
                y = torch.scatter_add(y, 0, pid, self.dropout(y1))
            if scale_eid:
                y = y.scatter_(0, pid, y1)
                y = torch.scatter_add(y, 0, eid, self.dropout(y2))
            else:
                y = y.scatter_(0, torch.cat([pid, eid], 0), torch.cat([y1, y2], 0))
            # x = x.detach()
            # x.requires_grad = True
            # if len(pid) > 0:
                # pid = pid.view(-1, 1, 1, 1)

            # y = y.scatter_(0, torch.cat([pid, eid], 0), torch.cat([y1, y2], 0))
            x = self.fusion(y, c)
        else:
            # print(pid, eid)
            if len(pid) > 0:
                x[pid, ...] = self.blocks(x[pid, ...])
            if len(eid) > 0:
                x[eid, ...] = self.simple_blocks(x[eid, ...])
            x = self.fusion(x, c)
        # print(x.shape)
        return x
class Level_DyDecoder_KernelPrior(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[96, 96], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        # print(channels, num_heads, layers)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        train_size = [train_size[0] // (2 ** (4-level)), train_size[1] // (2 ** (4-level))]
        patch_size = [patch_size[0] // (2 ** (4 - level)), patch_size[1] // (2 ** (4 - level))]
        self.fusion = SimpleFusion_Decoder5Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        # if level == 0:
        #     modules = [nn.Identity()]
        # else:
        modules = get_ker_modules(baseblock[level], level, channels, layers, num_heads[level], train_size, kernel_size=kernel_size, patch_size=patch_size)
        self.exit_threshold = 0.6
        self.blocks = nn.Sequential(*modules)
        self.simple_blocks = NAFBlock(channels[level])
        self.kernel_size = kernel_size
        self.dropout = nn.Dropout(1.)
    def forward(self, *args):
        x, c, kernel, ker_sel = args
        # print(x.shape, c.shape)
        # pid = torch.where(ker_sel < self.exit_threshold)[0]
        # eid = torch.where(exit_index >= 0.0)[0]
        # pid = torch.where(exit_index < 0.0)[0]

        # pid = torch.where(ker_sel < self.exit_threshold)[0]
        # eid = torch.where(ker_sel >= self.exit_threshold)[0]

        if self.training:
            ker_sel_ = F.gumbel_softmax(ker_sel, tau=1, hard=True, dim=-1)
            pid = torch.where(ker_sel_[:, 0] == 1)[0]
            eid = torch.where(ker_sel_[:, 1] == 1)[0]
            y = x.clone()
            b, c, h, w = x.shape
            # print(pid, eid)
            scale_eid = False
            scale_pid = False
            if len(eid) == 0:
                _, eid = torch.topk(ker_sel[:, 1], 1, dim=0, sorted=False)
                scale_eid = True
                # _, pid = torch.topk(-ker_sel, b-1, dim=0, sorted=False)
                # y = torch.scatter_add(y, 0, eid, y2)
            if len(pid) == 0:
                scale_pid = True
                # _, eid = torch.topk(ker_sel, b-1, dim=0, sorted=False)
                _, pid = torch.topk(ker_sel[:, 0], 1, dim=0, sorted=False)
            eid = eid.flatten()
            pid = pid.flatten()

            eid = repeat(eid, 'b -> b c h w', c=c, h=h, w=w)
            y2 = torch.gather(x, 0, eid)
            y2 = self.simple_blocks(y2)
            pidx = repeat(pid, 'b -> b c h w', c=c, h=h, w=w)
            pidk = repeat(pid, 'b -> b c h w', c=c, h=self.kernel_size, w=self.kernel_size)
            y1 = torch.gather(x, 0, pidx)
            k1 = torch.gather(kernel, 0, pidk)
            y1 = self.blocks([y1, k1])

            if scale_pid:
                y = y.scatter_(0, eid, y2)
                y = torch.scatter_add(y, 0, pidx, self.dropout(y1))
            if scale_eid:
                y = y.scatter_(0, pidx, y1)
                y = torch.scatter_add(y, 0, eid, self.dropout(y2))
            else:
                y = y.scatter_(0, torch.cat([pidx, eid], 0), torch.cat([y1, y2], 0))
            x = self.fusion(y, c)
        else:
            # ker_sel_ = torch.softmax(ker_sel, dim=-1)
            ker_sel_ = torch.argmax(ker_sel, dim=-1)
            # print(ker_sel_)
            pid = torch.where(ker_sel_ == 0)[0]
            eid = torch.where(ker_sel_ == 1)[0]
            if len(pid) > 0:
                x[pid, ...] = self.blocks([x[pid, ...], kernel[pid, ...]])
            if len(eid) > 0:
                x[eid, ...] = self.simple_blocks(x[eid, ...])

            x = self.fusion(x, c)

        # # x = self.blocks([x, kernel])
        # x = self.fusion(x, c)
        # print(x.shape)
        return x
class Level_SimpleDecoder_4(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66, sub_idx=0, upsample_mode='conv') -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        # print(channels, num_heads, layers)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        train_size = [train_size[0] // (2 ** (3 - level)), train_size[1] // (2 ** (3 - level))]
        patch_size = [patch_size[0] // (2 ** (3 - level)), patch_size[1] // (2 ** (3 - level))]
        self.fusion = SimpleFusion_Decoder4Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor, upsample_mode=upsample_mode)
        # if level == 0:
        #     modules = [nn.Identity()]
        # else:
        # self.up_u = nn.Conv2d(channels[level] * 2, channels[level], 1)
        # if level == 0:
        #     modules = [nn.Identity()]
        # else:
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size,
                              dp_rate, sub_idx=sub_idx)

        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        # if type(c) != int:
        # x = self.up_u(torch.cat([x, c], dim=1))
        x = self.blocks(x)
        x = self.fusion(x, c)
        # print(x.shape)
        return x
class Level_SimpleDecoder_4_KerEst(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66, sub_idx=0) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        # print(channels, num_heads, layers)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        train_size = [train_size[0] // (2 ** (3 - level)), train_size[1] // (2 ** (3 - level))]
        patch_size = [patch_size[0] // (2 ** (3 - level)), patch_size[1] // (2 ** (3 - level))]
        self.fusion = SimpleFusion_Decoder4Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        self.mask_feature = nn.Sequential(
            nn.Conv2d(in_channels=channels[level], out_channels=channels[level], kernel_size=3, padding=1, stride=1, groups=1,
                      bias=True),
            nn.GELU()
        )
        self.intro_feature = nn.Sequential(
            nn.Conv2d(in_channels=channels[level], out_channels=channels[level], kernel_size=3, padding=1, stride=1,
                      groups=1,
                      bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=channels[level], out_channels=channels[level], kernel_size=3, padding=1, stride=1,
                      groups=1,
                      bias=True)
        )
        self.intro_img = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=channels[level], kernel_size=3, padding=1, stride=1,
                      groups=1,
                      bias=True))

        self.latent_sharp = nn.Conv2d(in_channels=channels[level], out_channels=3, kernel_size=3, padding=1, stride=1,
                      groups=1,
                      bias=True)
        self.ending = nn.Sequential(
            nn.Conv2d(in_channels=channels[level] * 4, out_channels=channels[level], kernel_size=1, padding=0,
                      stride=1,
                      groups=1,
                      bias=True)
            )
        self.ending_ker = nn.Sequential(nn.Conv2d(in_channels=channels[level], out_channels=channels[level] * 2, kernel_size=3, padding=1,
                                              stride=1,
                                              groups=1,
                                              bias=True)
                                    )
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size,
                              dp_rate, sub_idx=sub_idx)
        self.deblur_feature = nn.Sequential(
            nn.Conv2d(in_channels=channels[level] * 2, out_channels=channels[level], kernel_size=1, padding=0,
                      stride=1,
                      groups=1,
                      bias=True)
            )
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, inp_image = args
        inp = x
        latent_sharp = self.latent_sharp(x)
        x_mask = self.mask_feature(x)
        x = self.intro_feature(x)
        x_ = self.intro_img(torch.cat([inp_image, latent_sharp + inp_image], dim=1))
        inp_freq = fft_ri_(x, clamp_min=math.exp(-10))
        img_freq = fft_ri_(x_, clamp_min=math.exp(-10))
        x = self.ending(torch.cat([inp_freq.real, img_freq.real, inp_freq.imag, img_freq.imag], dim=1))
        # print(x.shape, c.shape)
        # if type(c) != int:
        # x = self.up_u(torch.cat([x, c], dim=1))
        x = self.blocks(x)
        x = self.ending_ker(x)
        ker_real, ker_imag = torch.chunk(x, 2, dim=1)
        otf_deblur = inverse_ifft_ri_otf(ker_real, ker_imag)
        x = torch.fft.ifft2(inp_freq * torch.conj(otf_deblur)).real.contiguous()
        x = x * x_mask
        x = self.deblur_feature(torch.cat([x, inp], dim=1))
        x = self.fusion(x, None)
        # print(x.shape)
        return torch.cat([x, latent_sharp], dim=1)
class Level_SimpleMDecoder_4(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        # print(channels, num_heads, layers)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        train_size = [train_size[0] // (2 ** (3-level)), train_size[1] // (2 ** (3-level))]
        patch_size = [patch_size[0] // (2 ** (3 - level)), patch_size[1] // (2 ** (3 - level))]
        self.fusion = SimpleFusion_Decoder4Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        # if level == 0:
        #     modules = [nn.Identity()]
        # else:
        self.up_u = nn.Conv2d(channels[level] * 2, channels[level], 1)
        # if level == 0:
        #     modules = [nn.Identity()]
        # else:
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size,
                              dp_rate)

        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        # if type(c) != int:
        x = self.up_u(torch.cat([x, c], dim=1))
        x = self.blocks(x)
        x = self.fusion(x, c)
        # print(x.shape)
        return x

class Level_SimpleDecoder_3(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66, upsample_mode='conv', deblur_kernel=False) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        self.deblur_kernel = deblur_kernel
        # print(channels, num_heads, layers)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        train_size = [train_size[0] // (2 ** (2-level)), train_size[1] // (2 ** (2-level))]
        patch_size = [patch_size[0] // (2 ** (2 - level)), patch_size[1] // (2 ** (2 - level))]
        # print(level, channels, first_col, train_size, num_heads)
        self.fusion = SimpleFusion_Decoder3Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor, upsample_mode=upsample_mode)
        # if level == 0:
        #     modules = [nn.Identity()]
        # else:
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size, dp_rate)

        self.blocks = nn.Sequential(*modules)
        if self.deblur_kernel:
            self.ker_est = KernelEstimator(channels[level], baseblock_enc_ker=[baseblock[-1]], enc_ker_layers=[layers[-1]], num_heads_enc=[1], train_size=train_size, patch_size=patch_size)
    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.blocks(x)
        if self.deblur_kernel:
            x = self.ker_est(x)
        x = self.fusion(x, c)

        # print(x.shape)
        return x

class Level_FourierDecoder_3(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        # print(channels, num_heads, layers)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        train_size = [train_size[0] // (2 ** (2-level)), train_size[1] // (2 ** (2-level))]
        patch_size = [patch_size[0] // (2 ** (2 - level)), patch_size[1] // (2 ** (2 - level))]
        # print(level, channels, first_col, train_size, num_heads)
        self.fusion = SimpleFusion_Decoder3Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        # if level == 0:
        #     modules = [nn.Identity()]
        # else:
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size, dp_rate)

        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = torch.fft.rfft2(x)
        x = torch.cat([x.real, x.imag], dim=0)
        x = self.blocks(x)
        x_real, x_imag = x.chunk(2, dim=0)
        x = torch.complex(x_real, x_imag)
        x = torch.fft.irfft2(x)
        x = self.fusion(x, c)
        # print(x.shape)
        return x
class Level_SimpleKernelUpDecoder_3(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        # print(channels, num_heads, layers)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        train_size = [train_size[0] // (2 ** (4-level)), train_size[1] // (2 ** (4-level))]
        patch_size = [patch_size[0] // (2 ** (4 - level)), patch_size[1] // (2 ** (4 - level))]
        # print(level, channels, first_col, train_size, num_heads)
        self.fusion = SimpleFusion_KernelUpDecoder3Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        # if level == 0:
        #     modules = [nn.Identity()]
        # else:
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size, dp_rate)

        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.blocks(x)
        x = self.fusion(x, c)
        # print(x.shape)
        return x
class Level_SimpleMDecoder_3(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66, upsample_mode='conv') -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        # print(channels, num_heads, layers)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        train_size = [train_size[0] // (2 ** (2-level)), train_size[1] // (2 ** (2-level))]
        patch_size = [patch_size[0] // (2 ** (2 -level)), patch_size[1] // (2 ** (2-level))]
        self.fusion = SimpleFusion_Decoder3Level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor, upsample_mode=upsample_mode)
        # if level in [0, 1]:
        #     self.up_u = nn.Sequential(nn.Conv2d(channels[level+1], channels[level], 1, bias=False))
        # elif level in [2]:
        #     self.up_u = nn.Sequential(nn.Conv2d(channels[level], channels[level] * 4, 1, bias=False),
        #                               nn.PixelShuffle(2),
        #                               # LayerNorm(channels[level])
        #                               )
        self.up_u = nn.Conv2d(channels[level]*2, channels[level], 1)
        # if level == 0:
        #     modules = [nn.Identity()]
        # else:
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size, dp_rate)

        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        # if type(c) != int:
        x = self.up_u(torch.cat([x, c], dim=1))

        # print(c.shape, x.shape)
        x = self.blocks(x)

        x = self.fusion(x, c)
        # print(x.shape)
        return x
class Level_Encoder(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        # print(baseblock)
        self.fusion = Fusion_Encoder(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        if baseblock == 'naf':
            modules = [NAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
        elif baseblock == 'dcnv3':
            modules = [DCNv3Block(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
        elif baseblock == 'Res':
            modules = [ResBlock1(channels[level]) for _ in range(layers[level])]
        elif baseblock == 'FFTRes':
            modules = [ResFourier_complex(channels[level]) for _ in range(layers[level])]
        elif baseblock == 'fnaf':
            modules = [FNAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
        elif baseblock == 'fcnaf':
            modules = [FCNAFBlock(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
        elif baseblock == 'fdcnv3':
            modules = [FDCNv3Block(channels[level], num_heads=num_heads[level]) for i in range(layers[level])]
        elif baseblock == 'fftformer':
            modules = [fftformerblock(channels[level], ffn_expansion_factor=2, att=True) for i in
                       range(layers[level])]
        elif baseblock == 'Ffftformer':
            modules = [Ffftformer(channels[level], ffn_expansion_factor=2, att=True) for i in
                       range(layers[level])]
        elif baseblock == 'Fattn_FreqLC':
            modules = [
                AttnBlock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2, cs='channel_mlp')
                for
                i in range(layers[level])]
        elif baseblock == 'loformer_SpaLC':
            modules = [
                loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2,
                              cs='channel_nodct') for
                i in range(layers[level])]
        elif baseblock == 'loformer_SpaLS':
            modules = [
                loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2,
                              cs='spatial_nodct') for
                i in range(layers[level])]
        elif baseblock == 'loformer_FreqLC':
            modules = [
                loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2, cs='channel_mlp')
                for
                i in range(layers[level])]
        elif baseblock == 'loformer_FreqLS':
            modules = [
                loformerblock(channels[level], num_heads=num_heads[level], ffn_expansion_factor=2, cs='spatial_mlp')
                for
                i in range(layers[level])]
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)

        x = self.blocks(x)
        # print(x.shape)
        return x
class Level_Encoder_nofuse(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66, downsample_mode='conv') -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        self.fusion = Fusion_Encoder_nofuse(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor, downsample_mode=downsample_mode)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size, dp_rate=dp_rate)
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)

        x = self.blocks(x)
        # print(x.shape)
        return x

class Level_Encoder_nodownup(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66, flip=False, transpose=False) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        self.flip = flip
        self.transpose = transpose
        self.fusion = Fusion_Encoder_nodownup(level, channels, first_col, train_size, num_heads[level],
                                              ffn_expansion_factor=ffn_expansion_factor,)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock) baseblock, level, channels, layers, num_heads, train_size
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, dp_rate=dp_rate)
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)
        # if self.flip:
        #     x = torch.flip(x, dims=(-2, -1)).contiguous()
        # if self.transpose:
        #     x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
        x = self.blocks(x)
        # if self.flip:
        #     x = torch.flip(x, dims=(-2, -1)).contiguous()
        # if self.transpose:
        #     x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()
        # print(x.shape)
        return x

class Level_Encoder_nofuse_3level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66, sub_idx=0, downsample_mode='conv') -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        train_size = [train_size[0] // (2 ** level), train_size[1] // (2 ** level)]
        patch_size = [patch_size[0] // (2 ** level), patch_size[1] // (2 ** level)]
        self.fusion = Fusion_Encoder_nofuse_3level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor, downsample_mode=downsample_mode)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        # print('dp_rate', dp_rate)
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size, dp_rate=dp_rate, sub_idx=sub_idx)
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)

        x = self.blocks(x)
        # print(x.shape)
        return x

class Level_FourierEncoder_nofuse_3level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        train_size = [train_size[0] // (2 ** level), train_size[1] // (2 ** level)]
        patch_size = [patch_size[0] // (2 ** level), patch_size[1] // (2 ** level)]
        self.fusion = Fusion_Encoder_nofuse_3level(level, channels, first_col, train_size, num_heads[level],
                                                   ffn_expansion_factor=ffn_expansion_factor)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        # print('dp_rate', dp_rate)
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size,
                              kernel_size, dp_rate=dp_rate)
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)

        x = torch.fft.rfft2(x)
        x = torch.cat([x.real, x.imag], dim=0)
        x = self.blocks(x)
        x_real, x_imag = x.chunk(2, dim=0)
        x = torch.complex(x_real, x_imag)
        x = torch.fft.irfft2(x)
        # print(x.shape)
        return x


class Level_Encoder_fuse_3level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        train_size = [train_size[0] // (2 ** level), train_size[1] // (2 ** level)]
        patch_size = [patch_size[0] // (2 ** level), patch_size[1] // (2 ** level)]
        self.fusion = Fusion_Encoder_3level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size)
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)

        x = self.blocks(x)
        # print(x.shape)
        return x

class Level_Encoder_mfuse_3level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66, downsample='downsample', downsample_mode='conv', data_mode='BCHW') -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        self.data_mode = data_mode
        train_size = [train_size[0] // (2 ** level), train_size[1] // (2 ** level)]
        patch_size = [patch_size[0] // (2 ** level), patch_size[1] // (2 ** level)]
        self.fusion = MFusion_Encoder_3level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor, downsample=downsample, downsample_mode=downsample_mode)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size, dp_rate)
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)

        # if self.data_mode == 'BNC':
        #     B, C, H, W = x.shape
        #     x, B, C, H, W =
        # else:
        x = self.blocks(x)
        # print(x.shape)
        return x
class Level_Encoder_mfuse_4level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66, downsample='downsample', downsample_mode='conv') -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        train_size = [train_size[0] // (2 ** level), train_size[1] // (2 ** level)]
        patch_size = [patch_size[0] // (2 ** level), patch_size[1] // (2 ** level)]
        self.fusion = MFusion_Encoder_4level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor, downsample=downsample, downsample_mode=downsample_mode)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size, dp_rate)
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)

        x = self.blocks(x)
        # print(x.shape)
        return x
class Level_FourierEncoder_mfuse_3level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66, downsample=True) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        train_size = [train_size[0] // (2 ** level), train_size[1] // (2 ** level)]
        patch_size = [patch_size[0] // (2 ** level), patch_size[1] // (2 ** level)]
        self.fusion = MFusion_Encoder_3level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor, downsample=downsample)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size, dp_rate)
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.fusion(x, c)

        x = torch.fft.rfft2(x)
        x = torch.cat([x.real, x.imag], dim=0)
        x = self.blocks(x)
        x_real, x_imag = x.chunk(2, dim=0)
        x = torch.complex(x_real, x_imag)
        x = torch.fft.irfft2(x)
        # print(x.shape)
        return x
class Level_Encoder_mkerfuse_3level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        train_size = [train_size[0] // (2 ** level), train_size[1] // (2 ** level)]
        patch_size = [patch_size[0] // (2 ** level), patch_size[1] // (2 ** level)]
        # self.fusion = MFusion_Encoder_3level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        self.fusion = Fusion_Encoder_nofuse_3level(level, channels, first_col, train_size, num_heads[level],
                                             ffn_expansion_factor=ffn_expansion_factor)
        self.kernel_atttion = kernel_attention(kernel_size, in_ch=channels[level], out_ch=channels[level])
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size, dp_rate)
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x, c = args
        x = self.fusion(x, c)
        # print(x.shape, c.shape)
        x = self.kernel_atttion(x, c)

        x = self.blocks(x)
        # print(x.shape)
        return x
class Level_Encoder_maxmid_fuse_3level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        train_size = [train_size[0] // (2 ** level), train_size[1] // (2 ** level)]
        patch_size = [patch_size[0] // (2 ** level), patch_size[1] // (2 ** level)]
        self.fusion = MaxMidFusion_Encoder_3level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size, dp_rate)
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x_max, x_mid = args
        # print(x.shape, c.shape)
        x = self.fusion(x_max, x_mid)

        x = self.blocks(x)
        # print(x.shape)
        return x
class Level_Encoder_midmax_fuse_3level(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], patch_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        train_size = [train_size[0] // (2 ** level), train_size[1] // (2 ** level)]
        patch_size = [patch_size[0] // (2 ** level), patch_size[1] // (2 ** level)]
        self.fusion = MidMaxFusion_Encoder_3level(level, channels, first_col, train_size, num_heads[level], ffn_expansion_factor=ffn_expansion_factor)
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        modules = get_modules(baseblock[level], level, channels, layers, num_heads, train_size, patch_size, kernel_size, dp_rate)
        self.blocks = nn.Sequential(*modules)

    def forward(self, *args):
        x_mid, x_max = args
        # print(x.shape, c.shape)
        x = self.fusion(x_mid, x_max)

        x = self.blocks(x)
        # print(x.shape)
        return x
class Level_Pred_kernel(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        # expansion = 4
        self.down = nn.Sequential(
                nn.AdaptiveAvgPool2d((kernel_size, kernel_size))
            )
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        modules = get_modules(baseblock[level], level-1, channels, layers, num_heads, train_size)

        self.blocks = nn.Sequential(*modules)
        hid_dim = 1024
        self.mlp = nn.Sequential(
                        nn.Linear(kernel_size**2, hid_dim),
                        nn.BatchNorm1d(channels[level-1]),
                        nn.GELU(),
                        nn.Linear(hid_dim, kernel_size**2),
                        # nn.Sigmoid()
                        )
        # self.flow = KernelPrior(n_blocks=5, input_size=kernel_size ** 2, hidden_size=25, n_hidden=1, kernel_size=kernel_size)
        # self.kernel_hid = nn.Conv2d(channels[level-1], channels[level], kernel_size=3, stride=1, padding=1, bias=True)
        # self.kernel_extra = num_kernel_extra_conv_tail_mean_var(in_ch=channels[level+1], kernel_size=kernel_size)
        self.kernel_size = kernel_size
    # def post_process(self, x):
    #     x = torch.sigmoid(x)
    #     return x
    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        x = self.down(x)
        x = self.blocks(x)
        x = rearrange(x, 'b c k1 k2 -> b c (k1 k2)')
        x = self.mlp(x)
        kernel = rearrange(x, 'b c (k1 k2) -> b c k1 k2', k1=self.kernel_size, k2=self.kernel_size)
        # kernel = self.kernel_hid(kernel)
        # kernel_code, kernel_var = self.kernel_extra(x)
        # kernel_code = (kernel_code - torch.mean(kernel_code, dim=[2, 3], keepdim=True)) / torch.std(kernel_code,
        #                                                                                             dim=[2, 3],
        #                                                                                             keepdim=True)
        # # code uncertainty
        # sigma = kernel_var
        # kernel_code_uncertain = kernel_code * torch.sqrt(1 - torch.square(sigma)) + torch.randn_like(
        #     kernel_code) * sigma
        #
        # kernel = generate_k_forward(self.flow, kernel_code_uncertain.reshape(kernel_code.shape[0] * kernel_code.shape[1], -1))
        # # kernel = self.post_process(kernel)
        # kernel = kernel.reshape(kernel_code.shape[0], kernel_code.shape[1], self.kernel_size, self.kernel_size)
        # print(x.shape)
        return kernel

class Level_Pred_kernel_flow(nn.Module):
    def __init__(self, level, channels, layers, kernel_size, first_col, dp_rate=0.0, num_heads=[8, 4, 2, 1],
                 baseblock='naf', train_size=[256, 256], ffn_expansion_factor=2.66) -> None:
        super().__init__()
        # countlayer = sum(layers[:level])
        expansion = 4
        h_dim = int(np.sqrt(channels[level])) * expansion
        print(channels[level], h_dim)
        self.h_dim = h_dim
        k_dim = h_dim ** 2
        # self.down = nn.Sequential(
        #         nn.AdaptiveAvgPool2d((h_dim, h_dim))
        #     )
        if not isinstance(baseblock, list):
            baseblock = [baseblock, baseblock, baseblock, baseblock, baseblock]
        elif len(baseblock) == 1:
            baseblock = [baseblock[0], baseblock[0], baseblock[0], baseblock[0], baseblock[0]]
        # print(baseblock)
        modules = get_modules(baseblock[level], level-1, channels, layers, num_heads, train_size)
        self.blocks = nn.Sequential(*modules)
        hid_dim = 1024
        # self.mlp = nn.Sequential(
        #                 nn.Conv2d(channels[level-1], hid_dim, kernel_size=1, padding=0),
        #                 nn.BatchNorm2d(hid_dim),
        #                 nn.GELU(),
        #                 nn.Conv2d(hid_dim, kernel_size**2, kernel_size=3, padding=1)
        #                 )
        self.flow = KernelPrior(n_blocks=5, input_size=kernel_size ** 2, hidden_size=25, n_hidden=1, kernel_size=kernel_size)
        # self.kernel_hid = nn.Conv2d(k_dim, channels[level], kernel_size=3, stride=1, padding=1, bias=True)
        self.kernel_extra = kernel_extra_conv_tail_mean_var(in_ch=channels[level-1], out_ch=kernel_size**2)
        # self.kernel_extra = code_extra_mean_varX(channels[level-1], channels[level-1]+64, kernel_size)
        self.kernel_size = kernel_size

        # state_dict_pth = '/home/ubuntu/90t/personal_data/mxt/MXT/RevIR/code_reference/FKP/data/log_FKP/FKP_x2/model_checkpoint.pt'
        # state = torch.load(state_dict_pth)
        # state_dict = checkpoint["params"]
        # self.flow.load_state_dict(state['model_state'])
        # self.flow.eval()
        # for p in self.flow.parameters(): p.requires_grad = False
        flow_state_dict_pth = '/home/ubuntu/90t/personal_data/mxt/MXT/ckpts/flow/motion_blur/motion_blur_flow.pth'
        flow_state = torch.load(flow_state_dict_pth)
        # print(flow_state.keys())
        self.flow.load_state_dict(flow_state)
        self.flow.eval()
        for p in self.flow.parameters(): p.requires_grad = False
        # for k, v in self.flow.named_parameters():
        #     v.requires_grad = False
    # def post_process(self, x):
    #     x = torch.sigmoid(x)
    #     return x
    def forward(self, *args):
        x, c = args
        # print(x.shape, c.shape)
        # x = self.down(x)
        x = self.blocks(x)
        # x = self.mlp(x)
        # x = rearrange(x, 'b c k1 k2 -> b c (k1 k2)')
        # x = self.mlp(x)
        # x = rearrange(x, 'b c (k1 k2) -> b  c k1 k2', k1=self.kernel_size, k2=self.kernel_size)

        kernel_code, kernel_var = self.kernel_extra(x)
        kernel_code = (kernel_code - torch.mean(kernel_code, dim=[2, 3], keepdim=True)) / torch.std(kernel_code,
                                                                                                    dim=[2, 3],
                                                                                                    keepdim=True)
        # code uncertainty
        sigma = kernel_var
        kernel_code_uncertain = kernel_code * torch.sqrt(1 - torch.square(sigma)) + torch.randn_like(
            kernel_code) * sigma

        kernel = generate_k(self.flow, kernel_code_uncertain.reshape(kernel_code.shape[0] * kernel_code.shape[1], -1))
        # kernel = self.post_process(kernel)
        kernel = kernel.reshape(kernel_code.shape[0], kernel_code.shape[1], self.kernel_size, self.kernel_size)
        # kernel = kernel.permute(0, 2, 3, 1).reshape(kernel_code.shape[0], self.kernel_size * self.kernel_size, self.h_dim,
        #                                                     self.h_dim)
        kernel = kernel.permute(0, 2, 3, 1).reshape(x.shape[0], self.kernel_size * self.kernel_size, x.shape[2], x.shape[3])
        # kernel = torch.softmax(kernel, dim=1)
        # print(x.shape)
        # kernel = self.kernel_hid(kernel)
        return kernel
class Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = in_channels//2
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_channels, eps=1e-6),  # final norm layer
            # nn.Softmax(-1),
            # nn.Linear(in_channels, num_classes),
            nn.Linear(in_channels, hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_classes)
            # nn.Softmax(-1)
            # nn.Tanh()
        )

        self.alpha = nn.Parameter(torch.ones((1, num_classes)) / math.sqrt(num_classes),
                                  requires_grad=True)

    def forward(self, x):
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)

        x = x.view(x.size(0), -1)
        x = self.classifier(x) * self.alpha  # * self.alpha0
        # print('ics: ', x)
        return x  # torch.softmax(x, -1)
class DegClassifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        hidden_channels = 1024
        self.classifier = nn.Sequential(

            # nn.LayerNorm(in_channels, eps=1e-6),  # final norm layer
            # nn.Softmax(-1),
            # nn.Linear(in_channels, num_classes),
            nn.Linear(in_channels, hidden_channels),
            nn.BatchNorm1d(hidden_channels),
            nn.GELU(),
            nn.Linear(hidden_channels, num_classes),
            # nn.Softmax(-1)
            # nn.Tanh()
        )

        # self.alpha = nn.Parameter(torch.ones((1, num_classes)) / math.sqrt(num_classes),
        #                           requires_grad=True)

    def forward(self, x):
        # print(x.shape)
        x = self.avgpool(x)
        # print(x.shape)

        x = x.view(x.size(0), -1)
        x = self.classifier(x)  # * self.alpha  # * self.alpha0
        # print('ics: ', x)
        return x  # torch.softmax(x, -1)
class UniSimpleDecoder5Level(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[96, 96], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha4 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_SimpleDecoder_5(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_SimpleDecoder_5(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_SimpleDecoder_5(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level3 = Level_SimpleDecoder_5(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level4 = Level_SimpleDecoder_5(4, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

    # @torch.no_grad()
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, c4 = args

        # print('xxx')
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, c4)
        # print('c4: ', c4)
        # print('c3: ', c3.shape)
        c4 = (self.alpha4) * c4 + self.level4(c3, None)
        return c0, c1, c2, c3, c4

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4 = ReverseFunctionDecoder5Level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4
    def _forward_reverse_adapter(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4 = ReverseFunctionDecoder5Level_Adapter.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4
    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        self._clamp_abs(self.alpha4.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleDecoderF5Level(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[96, 96], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha4 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_SimpleDecoder_5(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_SimpleDecoder_5(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_SimpleDecoder_5(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level3 = Level_SimpleDecoder_5(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level4 = Level_SimpleDecoder_5(4, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, c4 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, c4)
        # print('c4: ', c4)
        # print('c3: ', c3.shape)
        c4 = (self.alpha4) * c4 + self.level4(c3, None)
        return c0, c1, c2, c3, c4

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4 = ReverseFunctionDecoderF5Level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        self._clamp_abs(self.alpha4.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleDecoder3Level(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1, upsample_mode='conv', deblur_kernel=False) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        # self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.register_parameter('alpha0', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha1', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha2', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                                       requires_grad=True))
        # print(dp_rates)
        self.level0 = Level_SimpleDecoder_3(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, upsample_mode=upsample_mode, deblur_kernel=deblur_kernel)

        self.level1 = Level_SimpleDecoder_3(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, upsample_mode=upsample_mode, deblur_kernel=deblur_kernel)

        self.level2 = Level_SimpleDecoder_3(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, upsample_mode=upsample_mode, deblur_kernel=deblur_kernel)
        self.just_backward = False
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, None)

        return c0, c1, c2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2]
        alpha = [self.alpha0, self.alpha1, self.alpha2]
        _, c0, c1, c2 = ReverseFunctionSimpleDecoder3level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2
    def _forward_backward(self, *args):

        # local_funs = [self.level0, self.level1, self.level2]
        # alpha = [self.alpha0, self.alpha1, self.alpha2]
        # _, c0, c1, c2 = ReverseFunctionSimpleDecoder3level_Adapter.apply(
        #     local_funs, alpha, *args)
        with torch.no_grad():
            return self._forward_nonreverse(*args)
    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        if self.just_backward:
            return self._forward_backward(*args)
        elif self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniFourierDecoder3Level(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        # print(dp_rates)
        self.level0 = Level_FourierDecoder_3(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_FourierDecoder_3(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_FourierDecoder_3(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)
        self.just_backward = False
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, None)

        return c0, c1, c2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2]
        alpha = [self.alpha0, self.alpha1, self.alpha2]
        _, c0, c1, c2 = ReverseFunctionSimpleDecoder3level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2
    def _forward_backward(self, *args):

        # local_funs = [self.level0, self.level1, self.level2]
        # alpha = [self.alpha0, self.alpha1, self.alpha2]
        # _, c0, c1, c2 = ReverseFunctionSimpleDecoder3level_Adapter.apply(
        #     local_funs, alpha, *args)
        with torch.no_grad():
            return self._forward_nonreverse(*args)
    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        if self.just_backward:
            return self._forward_backward(*args)
        elif self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleDecoder2Level(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # print(dp_rates)
        self.level0 = Level_SimpleDecoder_3(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_SimpleDecoder_3(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        # self.level2 = Level_SimpleDecoder_3(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
        #                             train_size, patch_size, ffn_expansion_factor)
        self.just_backward = False
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, None)
        # c2 = (self.alpha2) * c2 + self.level2(c1, None)

        return c0, c1

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1]
        alpha = [self.alpha0, self.alpha1]
        _, c0, c1, c2 = ReverseFunctionSimpleDecoder2level.apply(
            local_funs, alpha, *args)

        return c0, c1
    def _forward_backward(self, *args):

        # local_funs = [self.level0, self.level1, self.level2]
        # alpha = [self.alpha0, self.alpha1, self.alpha2]
        # _, c0, c1, c2 = ReverseFunctionSimpleDecoder3level_Adapter.apply(
        #     local_funs, alpha, *args)
        with torch.no_grad():
            return self._forward_nonreverse(*args)
    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        if self.just_backward:
            return self._forward_backward(*args)
        elif self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleKernelUpDecoder3Level(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        # print(dp_rates)
        self.level0 = Level_SimpleKernelUpDecoder_3(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_SimpleKernelUpDecoder_3(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_SimpleKernelUpDecoder_3(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)
        self.just_backward = False
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, None)

        return c0, c1, c2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2]
        alpha = [self.alpha0, self.alpha1, self.alpha2]
        _, c0, c1, c2 = ReverseFunctionSimpleDecoder3level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2
    def _forward_backward(self, *args):

        # local_funs = [self.level0, self.level1, self.level2]
        # alpha = [self.alpha0, self.alpha1, self.alpha2]
        # _, c0, c1, c2 = ReverseFunctionSimpleDecoder3level_Adapter.apply(
        #     local_funs, alpha, *args)
        with torch.no_grad():
            return self._forward_nonreverse(*args)
    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        if self.just_backward:
            return self._forward_backward(*args)
        elif self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleMDecoder3Level(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1, upsample_mode='conv') -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        # self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.register_parameter('alpha0', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha1', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha2', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                                       requires_grad=True))
        self.level0 = Level_SimpleMDecoder_3(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, upsample_mode=upsample_mode)

        self.level1 = Level_SimpleMDecoder_3(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, upsample_mode=upsample_mode)

        self.level2 = Level_SimpleMDecoder_3(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, upsample_mode=upsample_mode)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, d0, d1, d2 = args

        c0 = (self.alpha0) * c0 + self.level0(x, d0)
        c1 = (self.alpha1) * c1 + self.level1(c0, d1)
        c2 = (self.alpha2) * c2 + self.level2(c1, d2)

        return c0, c1, c2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2]
        alpha = [self.alpha0, self.alpha1, self.alpha2]
        _, c0, c1, c2, _, _, _ = ReverseFunctionSimpleMDecoder3level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign

class UniSimpleMDecoder4Level(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_SimpleMDecoder_4(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_SimpleMDecoder_4(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_SimpleMDecoder_4(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level3 = Level_SimpleMDecoder_4(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, d0, d1, d2, d3 = args

        c0 = (self.alpha0) * c0 + self.level0(x, d0)
        c1 = (self.alpha1) * c1 + self.level1(c0, d1)
        c2 = (self.alpha2) * c2 + self.level2(c1, d2)
        c3 = (self.alpha3) * c3 + self.level3(c2, d3)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3, _, _, _, _ = ReverseFunctionSimpleMDecoder4level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleDecoder4Level(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_SimpleDecoder_4(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx)

        self.level1 = Level_SimpleDecoder_4(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_SimpleDecoder_4(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx)

        self.level3 = Level_SimpleDecoder_4(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunctionSimpleDecoder4level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign

class UniSimpleDecoder4Level_(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1, upsample_mode='conv') -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory
        self.first_col = first_col
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_SimpleDecoder_4(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx, upsample_mode=upsample_mode)

        self.level1 = Level_SimpleDecoder_4(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx, upsample_mode=upsample_mode)

        self.level2 = Level_SimpleDecoder_4(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx, upsample_mode=upsample_mode)

        self.level3 = Level_SimpleDecoder_4(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx, upsample_mode=upsample_mode)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunctionSimpleDecoder4level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3
    def _forward_reverse_first(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunctionSimpleDecoder4level_first.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3
    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        if self.save_memory:
            if self.first_col:
                return self._forward_reverse_first(*args)
            else:
                return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleDecoder4Level_KerEst(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1, sub_image=False, return_local=False) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        self.overlap_size = [32, 32]
        self.ker_size = patch_size
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory
        self.sub_image = sub_image
        self.return_local = return_local
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha4 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4]+3, 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.dim = channels[4]
        self.level0 = Level_SimpleDecoder_4(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx)

        self.level1 = Level_SimpleDecoder_4(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_SimpleDecoder_4(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx)

        self.level3 = Level_SimpleDecoder_4(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx)
        self.level4 = Level_SimpleDecoder_4_KerEst(4, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                            train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx)
    def grids(self, x, level=0):
        b, c, h, w = x.shape
        n_level = 2 ** level

        assert b == 1
        k1, k2 = self.ker_size[0] // n_level, self.ker_size[1] // n_level
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = [self.overlap_size[0] // n_level, self.overlap_size[1] // n_level]  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])

        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1

        # import math
        step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                # if not self.ek1:
                #     # print(step_i, i, k1, h)
                #     self.ek1 = i + k1 - h # - self.overlap_size[0]
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    # if not self.ek2:
                    #     self.ek2 = j + k2 - w # + self.overlap_size[1]
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i * self.up_scale, 'j': j * self.up_scale})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        if level == 0:
            self.original_size = (b, self.out_channels, h * self.up_scale, w * self.up_scale)
            self.stride = (stride[0] * self.up_scale, stride[1] * self.up_scale)
            self.nr = num_row
            self.nc = num_col
            self.idxes = idxes
        return parts
    def grids_pad(self, x, level=0):
        b, c, H, W = x.shape
        s = [min(H, self.ker_size[0])-self.overlap_size[0], min(W, self.ker_size[1])-self.overlap_size[1]]
        num_row = (H - self.overlap_size[0] - 1) // s[0] + 1
        num_col = (W - self.overlap_size[1] - 1) // s[1] + 1
        # k2 = min(w, k2)
        x = F.pad(x, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='replicate')
        b, c, h, w = x.shape
        n_level = 2 ** level

        assert b == 1
        k1, k2 = (self.ker_size[0]+2*self.pad_size) // n_level, (self.ker_size[1]+2*self.pad_size) // n_level
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = [(self.overlap_size[0]+self.pad_size) // n_level, (self.overlap_size[1]+self.pad_size) // n_level]  # (64, 64)

        stride = (k1 - overlap_size[0]-self.pad_size//n_level, k2 - overlap_size[1]-self.pad_size//n_level)

        # num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        # num_col = (w - overlap_size[1] - 1) // stride[1] + 1

        # import math
        step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                # if not self.ek1:
                #     # print(step_i, i, k1, h)
                #     self.ek1 = i + k1 - h # - self.overlap_size[0]
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    # if not self.ek2:
                    #     self.ek2 = j + k2 - w # + self.overlap_size[1]
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i * self.up_scale, 'j': j * self.up_scale})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        if level == 0:
            self.original_size = (b, self.out_channels, H * self.up_scale, W * self.up_scale)
            self.stride = (s[0] * self.up_scale, s[1] * self.up_scale)
            self.nr = num_row
            self.nc = num_col
            self.idxes = idxes
        return parts

    def get_overlap_matrix(self, h, w):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        self.h = h
        self.w = w
        # print(self.stride, self.overlap_size_up)
        self.ek1 = self.nr * self.stride[0] + self.overlap_size_up[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.overlap_size_up[1] * 2 - w
        # self.ek1, self.ek2 = 48, 224
        # print(self.ek1, self.ek2, self.nr)
        # print(self.overlap_size_up,self.ek1, self.ek2, self.nr, self.nc)
        # self.overlap_size = [8, 8]
        # self.overlap_size = [self.overlap_size[0] * 2, self.overlap_size[1] * 2]
        self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
        self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
        self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
        self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1)
        self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1)

    def grids_inverse(self, outs, out_device=None, pix_max=1.):
        if out_device is None:
            out_device = outs.device
        # print(out_device)

        type_out = torch.uint8 if pix_max == 255. else torch.float32
        # type_out = torch.int16
        preds = torch.zeros(self.original_size, device=out_device, dtype=type_out)  # .to(out_device)
        b, c, h, w = self.original_size
        # print(self.original_size)
        # outs = torch.clamp(outs, 0, 1) # * pix_max
        # outs = outs.type(type_out)
        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size_up
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)
        # rounding_mode = 'floor'
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']

            if i != 0 and i + k1 != h:
                outs[cnt, :, :self.overlap_size_up[0], :] = torch.mul(outs[cnt, :, :self.overlap_size_up[0], :],
                                                                      self.fuse_matrix_h2.to(outs.device))
                # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
            if i + k1 * 2 - self.ek1 < h:
                outs[cnt, :, -self.overlap_size_up[0]:, :] = torch.mul(outs[cnt, :, -self.overlap_size_up[0]:, :],
                                                                       self.fuse_matrix_h1.to(outs.device))
            # print(self.fuse_matrix_eh1.dtype)
            if i + k1 == h:
                outs[cnt, :, :self.ek1, :] = torch.mul(outs[cnt, :, :self.ek1, :], self.fuse_matrix_eh2.to(outs.device))
            if i + k1 * 2 - self.ek1 == h:
                outs[cnt, :, -self.ek1:, :] = torch.mul(outs[cnt, :, -self.ek1:, :],
                                                        self.fuse_matrix_eh1.to(outs.device))

            if j != 0 and j + k2 != w:
                outs[cnt, :, :, :self.overlap_size_up[1]] = torch.mul(outs[cnt, :, :, :self.overlap_size_up[1]],
                                                                      self.fuse_matrix_w2.to(outs.device))
            if j + k2 * 2 - self.ek2 < w:
                # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, -self.overlap_size_up[1]:] = torch.mul(outs[cnt, :, :, -self.overlap_size_up[1]:],
                                                                       self.fuse_matrix_w1.to(outs.device))
            if j + k2 == w:
                # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                outs[cnt, :, :, :self.ek2] = torch.mul(outs[cnt, :, :, :self.ek2], self.fuse_matrix_ew2.to(outs.device))
            if j + k2 * 2 - self.ek2 == w:
                # print('j + k2*2 - self.ek2 == w: ')
                outs[cnt, :, :, -self.ek2:] = torch.mul(outs[cnt, :, :, -self.ek2:],
                                                        self.fuse_matrix_ew1.to(outs.device))
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            # pred_win = outs[cnt, :, :, :].clone() * pix_max
            # pred_win = pred_win.type(type_out)
            # print(pred_win.dtype, preds.dtype, type_out, outs.dtype)
            # print(outs[cnt, :, :, :].max())
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].type(type_out).to(out_device)
            # preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].to(out_device) * pix_max
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds  # / count_mt
    def _forward_nonreverse(self, *args):
        inp_image, x, c0, c1, c2, c3, c4 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, c4)
        if self.sub_image:
            ck = self.grid(c3)
        else:
            ck = c3
        c4 = (self.alpha4) * c4 + self.level4(ck, inp_image)
        if self.sub_image and not self.return_local:
            c4 = self.grids_inverse(c4)
        return c0, c1, c2, c3, c4

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4 = ReverseFunction5Level_KernelEst.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        self._clamp_abs(self.alpha4.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)
    def _clamp_abs_alpha4(self, data, value):
        data, d = torch.split(data, [self.dim, 3], 1)
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
            d = d * 0.
            self.alpha4.data = torch.cat([data, d], dim=1)
    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniDecoder(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates, save_memory, num_heads=[8, 4, 2, 1], baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [8, 4, 2, 1]
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level_Decoder(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_Decoder(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_Decoder(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level3 = Level_Decoder(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

    def load_pretain_model(self, state_dict_pth):
        checkpoint = torch.load(state_dict_pth)

        state_dict = checkpoint["params"]
        encoder1_state_dict = OrderedDict()
        encoder2_state_dict = OrderedDict()
        encoder3_state_dict = OrderedDict()
        encoder4_state_dict = OrderedDict()
        up1_state_dict = OrderedDict()
        up2_state_dict = OrderedDict()
        up3_state_dict = OrderedDict()
        up4_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)

            # v = repeat(v, )
            if k[:8] == 'decoders':
                name_a = k[9:]  # remove `module.`
                name_1 = int(name_a[0])
                name = k[11:]
                # print(v.shape)
                if name_1 == 0:
                    encoder1_state_dict[name] = v
                elif name_1 == 1:
                    encoder2_state_dict[name] = v
                elif name_1 == 2:
                    encoder3_state_dict[name] = v
                elif name_1 == 3:
                    encoder4_state_dict[name] = v
                    # print(name)
            elif k[:3] == 'ups':
                name_a = k[4:]
                idx = name_a.split('.')
                name = '0' + k[7:]
                name_1 = int(idx[0])
                if name_1 == 0:
                    up1_state_dict[name] = v
                elif name_1 == 1:
                    up2_state_dict[name] = v
                elif name_1 == 2:
                    up3_state_dict[name] = v
                elif name_1 == 3:
                    up4_state_dict[name] = v

        self.level0.blocks.load_state_dict(encoder1_state_dict, strict=True)
        self.level1.blocks.load_state_dict(encoder2_state_dict, strict=True)
        self.level2.blocks.load_state_dict(encoder3_state_dict, strict=True)
        self.level3.blocks.load_state_dict(encoder4_state_dict, strict=True)
        self.level0.fusion.up.load_state_dict(up1_state_dict, strict=True)
        self.level1.fusion.up.load_state_dict(up2_state_dict, strict=True)
        self.level2.fusion.up.load_state_dict(up3_state_dict, strict=True)
        self.level3.fusion.up.load_state_dict(up4_state_dict, strict=True)

        print('-----------load pretrained decoder from ' + state_dict_pth + '----------------')

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign

class UniEncoder(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        num_heads = [1, 2, 4, 8]
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level_Encoder(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level1 = Level_Encoder(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level2 = Level_Encoder(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level3 = Level_Encoder(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign

    def load_pretain_model(self, state_dict_pth):
        checkpoint = torch.load(state_dict_pth)

        state_dict = checkpoint["params"]
        encoder1_state_dict = OrderedDict()
        encoder2_state_dict = OrderedDict()
        encoder3_state_dict = OrderedDict()
        encoder4_state_dict = OrderedDict()
        up1_state_dict = OrderedDict()
        up2_state_dict = OrderedDict()
        up3_state_dict = OrderedDict()
        up4_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)

            # v = repeat(v, )
            if k[:8] == 'encoders':
                name_a = k[9:]  # remove `module.`
                name_1 = int(name_a[0])
                name = k[11:]
                # print(v.shape)
                if name_1 == 0:
                    encoder1_state_dict[name] = v
                elif name_1 == 1:
                    encoder2_state_dict[name] = v
                elif name_1 == 2:
                    encoder3_state_dict[name] = v
                elif name_1 == 3:
                    encoder4_state_dict[name] = v
                    # print(name)
            elif k[:5] == 'downs':
                name_a = k[6:]
                idx = name_a.split('.')
                name = '0' + k[7:]
                name_1 = int(idx[0])
                if name_1 == 0:
                    up1_state_dict[name] = v
                elif name_1 == 1:
                    up2_state_dict[name] = v
                elif name_1 == 2:
                    up3_state_dict[name] = v
                # elif name_1 == 3:
                #     up4_state_dict[name] = v

        self.level0.blocks.load_state_dict(encoder1_state_dict, strict=True)
        self.level1.blocks.load_state_dict(encoder2_state_dict, strict=True)
        self.level2.blocks.load_state_dict(encoder3_state_dict, strict=True)
        self.level3.blocks.load_state_dict(encoder4_state_dict, strict=True)
        # self.level0.fusion.down.load_state_dict(up1_state_dict, strict=True)
        self.level1.fusion.down.load_state_dict(up1_state_dict, strict=True)
        self.level2.fusion.down.load_state_dict(up2_state_dict, strict=True)
        self.level3.fusion.down.load_state_dict(up3_state_dict, strict=True)

        print('-----------load pretrained subencoder from ' + state_dict_pth + '----------------')
class UniEncoderKernelPrior(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        num_heads = [8, 4, 2, 1]
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level_Encoder_KernelPrior(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level1 = Level_Encoder_KernelPrior(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level2 = Level_Encoder_KernelPrior(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level3 = Level_Encoder(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, k0, k1, k2 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1, k0)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2, k1)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3, k2)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3, _, _, _ = ReverseFunctionKernelPrior.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
    def load_pretain_model(self, state_dict_pth):
        checkpoint = torch.load(state_dict_pth)

        state_dict = checkpoint["params"]
        encoder1_state_dict = OrderedDict()
        encoder2_state_dict = OrderedDict()
        encoder3_state_dict = OrderedDict()
        encoder4_state_dict = OrderedDict()
        up1_state_dict = OrderedDict()
        up2_state_dict = OrderedDict()
        up3_state_dict = OrderedDict()
        up4_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)

            # v = repeat(v, )
            if k[:8] == 'encoders':
                name_a = k[9:]  # remove `module.`
                name_1 = int(name_a[0])
                name = k[11:]
                # print(v.shape)
                if name_1 == 0:
                    encoder1_state_dict[name] = v
                elif name_1 == 1:
                    encoder2_state_dict[name] = v
                elif name_1 == 2:
                    encoder3_state_dict[name] = v
                elif name_1 == 3:
                    encoder4_state_dict[name] = v
                    # print(name)
            elif k[:5] == 'downs':
                name_a = k[6:]
                idx = name_a.split('.')
                name = '0' + k[7:]
                name_1 = int(idx[0])
                if name_1 == 0:
                    up1_state_dict[name] = v
                elif name_1 == 1:
                    up2_state_dict[name] = v
                elif name_1 == 2:
                    up3_state_dict[name] = v
                # elif name_1 == 3:
                #     up4_state_dict[name] = v

        self.level0.blocks.load_state_dict(encoder1_state_dict, strict=True)
        self.level1.blocks.load_state_dict(encoder2_state_dict, strict=True)
        self.level2.blocks.load_state_dict(encoder3_state_dict, strict=True)
        self.level3.blocks.load_state_dict(encoder4_state_dict, strict=True)
        # self.level0.fusion.down.load_state_dict(up1_state_dict, strict=True)
        self.level1.fusion.down.load_state_dict(up1_state_dict, strict=True)
        self.level2.fusion.down.load_state_dict(up2_state_dict, strict=True)
        self.level3.fusion.down.load_state_dict(up3_state_dict, strict=True)

        print('-----------load pretrained subencoder from ' + state_dict_pth + '----------------')
class UniEncoderKernelPriorV2(nn.Module):
    def __init__(self, channels, layers, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        num_heads = [8, 4, 2, 1]
        self.save_memory = save_memory
        self.kalpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, kernel_size**2, 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.kalpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, kernel_size**2, 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.kalpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, kernel_size**2, 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.levelk0 = Level_Encoder_3level(0, channels, layers, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)

        self.levelk1 = Level_Encoder_3level(1, channels, layers, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)

        self.levelk2 = Level_Encoder_3level(2, channels, layers, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level_Encoder_KernelPrior(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level1 = Level_Encoder_KernelPrior(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level2 = Level_Encoder_KernelPrior(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level3 = Level_Encoder(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, k, k0, k1, k2 = args
        k0 = (self.kalpha0) * k0 + self.levelk0(k, k1)
        k1 = (self.kalpha1) * k1 + self.levelk1(k0, k2)
        k2 = (self.kalpha2) * k2 + self.levelk2(k1, None)

        c0 = (self.alpha0) * c0 + self.level0(x, c1, k0)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2, k1)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3, k2)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3, k0, k1, k2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.levelk0, self.levelk1, self.levelk2]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.kalpha0, self.kalpha1, self.kalpha2]
        _, c0, c1, c2, c3, _, k0, k1, k2 = ReverseFunctionKernelPriorV2.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, k0, k1, k2

    def forward(self, *args):
        self._clamp_abs(self.kalpha0.data, 1e-3)
        self._clamp_abs(self.kalpha1.data, 1e-3)
        self._clamp_abs(self.kalpha2.data, 1e-3)

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
    def load_pretain_model(self, state_dict_pth):
        checkpoint = torch.load(state_dict_pth)

        state_dict = checkpoint["params"]
        encoder1_state_dict = OrderedDict()
        encoder2_state_dict = OrderedDict()
        encoder3_state_dict = OrderedDict()
        encoder4_state_dict = OrderedDict()
        up1_state_dict = OrderedDict()
        up2_state_dict = OrderedDict()
        up3_state_dict = OrderedDict()
        kdown1_state_dict = OrderedDict()
        kdown2_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)

            # v = repeat(v, )
            if k[:8] == 'encoders':
                name_a = k[9:]  # remove `module.`
                name_1 = int(name_a[0])
                name = k[11:]
                # print(v.shape)
                if name_1 == 0:
                    encoder1_state_dict[name] = v
                elif name_1 == 1:
                    encoder2_state_dict[name] = v
                elif name_1 == 2:
                    encoder3_state_dict[name] = v
                elif name_1 == 3:
                    encoder4_state_dict[name] = v
                    # print(name)
            elif k[:5] == 'downs':
                name_a = k[6:]
                idx = name_a.split('.')
                name = '0' + k[7:]
                name_1 = int(idx[0])
                if name_1 == 0:
                    up1_state_dict[name] = v
                elif name_1 == 1:
                    up2_state_dict[name] = v
                elif name_1 == 2:
                    up3_state_dict[name] = v
            elif k[:11] == 'kernel_down':
                name_a = k[12:]
                idx = name_a.split('.')
                name = '0' + k[13:]
                name_1 = int(idx[0])
                if name_1 == 0:
                    kdown1_state_dict[name] = v
                elif name_1 == 1:
                    kdown2_state_dict[name] = v
                # elif name_1 == 3:
                #     up4_state_dict[name] = v

        self.level0.blocks.load_state_dict(encoder1_state_dict, strict=True)
        self.level1.blocks.load_state_dict(encoder2_state_dict, strict=True)
        self.level2.blocks.load_state_dict(encoder3_state_dict, strict=True)
        self.level3.blocks.load_state_dict(encoder4_state_dict, strict=True)
        # self.level0.fusion.down.load_state_dict(up1_state_dict, strict=True)
        self.levelk1.fusion.down.load_state_dict(kdown1_state_dict, strict=True)
        self.levelk2.fusion.down.load_state_dict(kdown2_state_dict, strict=True)
        self.level1.fusion.down.load_state_dict(up1_state_dict, strict=True)
        self.level2.fusion.down.load_state_dict(up2_state_dict, strict=True)
        self.level3.fusion.down.load_state_dict(up3_state_dict, strict=True)

        print('-----------load pretrained subencoder from ' + state_dict_pth + '----------------')
class UniEncoderKernelPrior5Level(nn.Module):
    def __init__(self, channels, layers, layers_k, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory
        self.kalpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.kalpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.kalpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.levelk0 = Level_Encoder_3level(0, channels, layers_k, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)

        self.levelk1 = Level_Encoder_3level(1, channels, layers_k, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)

        self.levelk2 = Level_Encoder_3level(2, channels, layers_k, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha4 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_Encoder_KernelPrior(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level1 = Level_Encoder_KernelPrior(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level2 = Level_Encoder_KernelPrior(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level3 = Level_Encoder_5(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)
        self.level4 = Level_Encoder_5(4, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, c4, k, k0, k1, k2 = args
        k0 = (self.kalpha0) * k0 + self.levelk0(k, k1)
        k1 = (self.kalpha1) * k1 + self.levelk1(k0, k2)
        k2 = (self.kalpha2) * k2 + self.levelk2(k1, None)

        c0 = (self.alpha0) * c0 + self.level0(x, c1, k0)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2, k1)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3, k2)
        c3 = (self.alpha3) * c3 + self.level3(c2, c4)
        c4 = (self.alpha4) * c4 + self.level4(c3, None)
        return c0, c1, c2, c3, c4, k0, k1, k2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4, self.levelk0, self.levelk1, self.levelk2]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4, self.kalpha0, self.kalpha1, self.kalpha2]
        _, c0, c1, c2, c3, c4, _, k0, k1, k2 = ReverseFunctionKernelPrior5levelV2.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4, k0, k1, k2

    def forward(self, *args):
        self._clamp_abs(self.kalpha0.data, 1e-3)
        self._clamp_abs(self.kalpha1.data, 1e-3)
        self._clamp_abs(self.kalpha2.data, 1e-3)

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        self._clamp_abs(self.alpha4.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
    def load_pretain_model(self, state_dict_pth):
        checkpoint = torch.load(state_dict_pth)

        state_dict = checkpoint["params"]
        encoder1_state_dict = OrderedDict()
        encoder2_state_dict = OrderedDict()
        encoder3_state_dict = OrderedDict()
        encoder4_state_dict = OrderedDict()
        up1_state_dict = OrderedDict()
        up2_state_dict = OrderedDict()
        up3_state_dict = OrderedDict()
        kdown1_state_dict = OrderedDict()
        kdown2_state_dict = OrderedDict()
        for k, v in state_dict.items():
            # print(k)

            # v = repeat(v, )
            if k[:8] == 'encoders':
                name_a = k[9:]  # remove `module.`
                name_1 = int(name_a[0])
                name = k[11:]
                # print(v.shape)
                if name_1 == 0:
                    encoder1_state_dict[name] = v
                elif name_1 == 1:
                    encoder2_state_dict[name] = v
                elif name_1 == 2:
                    encoder3_state_dict[name] = v
                elif name_1 == 3:
                    encoder4_state_dict[name] = v
                    # print(name)
            elif k[:5] == 'downs':
                name_a = k[6:]
                idx = name_a.split('.')
                name = '0' + k[7:]
                name_1 = int(idx[0])
                if name_1 == 0:
                    up1_state_dict[name] = v
                elif name_1 == 1:
                    up2_state_dict[name] = v
                elif name_1 == 2:
                    up3_state_dict[name] = v
            elif k[:11] == 'kernel_down':
                name_a = k[12:]
                idx = name_a.split('.')
                name = '0' + k[13:]
                name_1 = int(idx[0])
                if name_1 == 0:
                    kdown1_state_dict[name] = v
                elif name_1 == 1:
                    kdown2_state_dict[name] = v
                # elif name_1 == 3:
                #     up4_state_dict[name] = v

        self.level0.blocks.load_state_dict(encoder1_state_dict, strict=True)
        self.level1.blocks.load_state_dict(encoder2_state_dict, strict=True)
        self.level2.blocks.load_state_dict(encoder3_state_dict, strict=True)
        self.level3.blocks.load_state_dict(encoder4_state_dict, strict=True)
        # self.level0.fusion.down.load_state_dict(up1_state_dict, strict=True)
        self.levelk1.fusion.down.load_state_dict(kdown1_state_dict, strict=True)
        self.levelk2.fusion.down.load_state_dict(kdown2_state_dict, strict=True)
        self.level1.fusion.down.load_state_dict(up1_state_dict, strict=True)
        self.level2.fusion.down.load_state_dict(up2_state_dict, strict=True)
        self.level3.fusion.down.load_state_dict(up3_state_dict, strict=True)

        print('-----------load pretrained subencoder from ' + state_dict_pth + '----------------')
class UniEncoderKernelFeature3Level(nn.Module):
    def __init__(self, channels, layers, layers_k, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        num_heads = [4, 2, 1]
        self.save_memory = save_memory
        self.kalpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.kalpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.kalpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, kernel_size**2, 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.levelk0 = Level_Encoder_3level(0, channels, layers_k, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)

        self.levelk1 = Level_Encoder_3level(1, channels, layers_k, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)

        self.levelk2 = Level_Encoder_3level_kernel(2, channels, layers_k, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)

    def _forward_nonreverse(self, *args):
        k, k0, k1, k2 = args
        k0 = (self.kalpha0) * k0 + self.levelk0(k, k1)
        k1 = (self.kalpha1) * k1 + self.levelk1(k0, k2)
        k2 = (self.kalpha2) * k2 + self.levelk2(k1, None)

        return k0, k1, k2

    def _forward_reverse(self, *args):

        local_funs = [self.levelk0, self.levelk1, self.levelk2]
        alpha = [self.kalpha0, self.kalpha1, self.kalpha2]
        _, k0, k1, k2 = ReverseFunctionKernelFeature3level.apply(
            local_funs, alpha, *args)

        return k0, k1, k2

    def forward(self, *args):
        self._clamp_abs(self.kalpha0.data, 1e-3)
        self._clamp_abs(self.kalpha1.data, 1e-3)
        self._clamp_abs(self.kalpha2.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniEncoderKernelFeature4Level(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf', num_kernel=5,
                 train_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4, 8]
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level_Encoder_nofuse(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level1 = Level_Encoder_nofuse(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level2 = Level_Encoder_nofuse(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        # self.level3 = Level_Pred_kernel(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
        #                             train_size, ffn_expansion_factor)
        self.level3 = Level_Pred_kernel_flow(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                        train_size, ffn_expansion_factor)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleEncoder3Level(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=0, downsample_mode='conv') -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4]
        self.save_memory = save_memory
        # self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.register_parameter('alpha0', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha1', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha2', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                                       requires_grad=True))
        # self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # Level_Encoder_nofuse_3level
        # print(dp_rates)
        self.level0 = Level_Encoder_nofuse_3level(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx, downsample_mode=downsample_mode)

        self.level1 = Level_Encoder_nofuse_3level(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx, downsample_mode=downsample_mode)

        self.level2 = Level_Encoder_nofuse_3level(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, sub_idx=sub_idx, downsample_mode=downsample_mode)

        # self.level3 = Level_Pred_kernel(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
        #                             train_size, ffn_expansion_factor)
        self.just_backward = False
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, None)
        # c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2]
        alpha = [self.alpha0, self.alpha1, self.alpha2]
        _, c0, c1, c2 = ReverseFunctionKernelFeature3level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2
    def _forward_backward(self, *args):
        with torch.no_grad():
            return self._forward_nonreverse(*args)
        # local_funs = [self.level0, self.level1, self.level2]
        # alpha = [self.alpha0, self.alpha1, self.alpha2]
        # _, c0, c1, c2 = ReverseFunctionKernelFeature3level_Adapter.apply(
        #     local_funs, alpha, *args)
        #
        # return c0, c1, c2
    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        # self._clamp_abs(self.alpha3.data, 1e-3)
        if self.just_backward:
            return self._forward_backward(*args)
        elif self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniFourierEncoder3Level(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4]
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # Level_Encoder_nofuse_3level
        # print(dp_rates)
        self.level0 = Level_FourierEncoder_nofuse_3level(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_FourierEncoder_nofuse_3level(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_FourierEncoder_nofuse_3level(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        # self.level3 = Level_Pred_kernel(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
        #                             train_size, ffn_expansion_factor)
        self.just_backward = False
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, None)
        # c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2]
        alpha = [self.alpha0, self.alpha1, self.alpha2]
        _, c0, c1, c2 = ReverseFunctionKernelFeature3level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2
    def _forward_backward(self, *args):
        with torch.no_grad():
            return self._forward_nonreverse(*args)
        # local_funs = [self.level0, self.level1, self.level2]
        # alpha = [self.alpha0, self.alpha1, self.alpha2]
        # _, c0, c1, c2 = ReverseFunctionKernelFeature3level_Adapter.apply(
        #     local_funs, alpha, *args)
        #
        # return c0, c1, c2
    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        # self._clamp_abs(self.alpha3.data, 1e-3)
        if self.just_backward:
            return self._forward_backward(*args)
        elif self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleKernelDownEncoder3Level(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4]
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # Level_Encoder_nofuse_3level
        # print(dp_rates)
        self.level0 = Level_KernelDownEncoder_nofuse_3level(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_KernelDownEncoder_nofuse_3level(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_KernelDownEncoder_nofuse_3level(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        # self.level3 = Level_Pred_kernel(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
        #                             train_size, ffn_expansion_factor)
        self.just_backward = False
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, None)
        # c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2]
        alpha = [self.alpha0, self.alpha1, self.alpha2]
        _, c0, c1, c2 = ReverseFunctionKernelFeature3level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2
    def _forward_backward(self, *args):
        with torch.no_grad():
            return self._forward_nonreverse(*args)
        # local_funs = [self.level0, self.level1, self.level2]
        # alpha = [self.alpha0, self.alpha1, self.alpha2]
        # _, c0, c1, c2 = ReverseFunctionKernelFeature3level_Adapter.apply(
        #     local_funs, alpha, *args)
        #
        # return c0, c1, c2
    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        # self._clamp_abs(self.alpha3.data, 1e-3)
        if self.just_backward:
            return self._forward_backward(*args)
        elif self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleMEncoder3Level(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1, downsample='downsample', downsample_mode='conv', data_mode='BCHW') -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4]
        self.save_memory = save_memory
        self.register_parameter('alpha0', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha1', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha2', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                                       requires_grad=True))
        # self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # Level_Encoder_nofuse_3level
        self.level0 = Level_Encoder_mfuse_3level(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample, downsample_mode=downsample_mode, data_mode=data_mode)

        self.level1 = Level_Encoder_mfuse_3level(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample, downsample_mode=downsample_mode, data_mode=data_mode)

        self.level2 = Level_Encoder_mfuse_3level(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample, downsample_mode=downsample_mode, data_mode=data_mode)

        # self.level3 = Level_Pred_kernel(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
        #                             train_size, ffn_expansion_factor)

        self.just_backward = False
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, u0, u1, u2 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, u0)
        c1 = (self.alpha1) * c1 + self.level1(c0, u1)
        c2 = (self.alpha2) * c2 + self.level2(c1, u2)
        # c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2]
        alpha = [self.alpha0, self.alpha1, self.alpha2]
        _, c0, c1, c2, _, _, _ = ReverseFunctionMultisacleSimpleEncoder3level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2
    def _forward_backward(self, *args):
        with torch.no_grad():
            return self._forward_nonreverse(*args)
        # local_funs = [self.level0, self.level1, self.level2]
        # alpha = [self.alpha0, self.alpha1, self.alpha2]
        # _, c0, c1, c2 = ReverseFunctionMultisacleSimpleEncoder3level_Adapter.apply(
        #     local_funs, alpha, *args)
        #
        # return c0, c1, c2
    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        # self._clamp_abs(self.alpha3.data, 1e-3)

        if self.just_backward:
            return self._forward_backward(*args)
        elif self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleMEncoder4Level(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1, downsample='downsample') -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4]
        self.save_memory = save_memory
        self.downsample = downsample
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        # Level_Encoder_nofuse_3level
        self.level0 = Level_Encoder_mfuse_4level(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample)

        self.level1 = Level_Encoder_mfuse_4level(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample)

        self.level2 = Level_Encoder_mfuse_4level(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample)

        self.level3 = Level_Encoder_mfuse_4level(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample)

        self.just_backward = False
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, u0, u1, u2, u3 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, u0)
        c1 = (self.alpha1) * c1 + self.level1(c0, u1)
        c2 = (self.alpha2) * c2 + self.level2(c1, u2)
        c3 = (self.alpha3) * c3 + self.level3(c2, u3)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, C3, _, _, _, _ = ReverseFunctionSimpleMEncoder4level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, C3
    def _forward_reverse_nofuse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]

        _, c0, c1, c2, c3 = ReverseFunction_nofuse.apply(
            local_funs, alpha, *args)
        return c0, c1, c2, c3
    def _forward_backward(self, *args):
        with torch.no_grad():
            return self._forward_nonreverse(*args)
        # local_funs = [self.level0, self.level1, self.level2]
        # alpha = [self.alpha0, self.alpha1, self.alpha2]
        # _, c0, c1, c2 = ReverseFunctionMultisacleSimpleEncoder3level_Adapter.apply(
        #     local_funs, alpha, *args)
        #
        # return c0, c1, c2
    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.just_backward:
            return self._forward_backward(*args)
        elif self.save_memory:
            if self.downsample == 'None':
                x, c0, c1, c2, c3, _, _, _, _ = args
                return self._forward_reverse_nofuse(x, c0, c1, c2, c3)
            else:
                return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign

class UniSimpleMEncoder4Level_(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1, downsample='downsample', downsample_mode='conv') -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4]
        self.save_memory = save_memory
        self.downsample = downsample
        self.first_col = first_col
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        # Level_Encoder_nofuse_3level
        self.level0 = Level_Encoder_mfuse_4level(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample, downsample_mode=downsample_mode)

        self.level1 = Level_Encoder_mfuse_4level(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample, downsample_mode=downsample_mode)

        self.level2 = Level_Encoder_mfuse_4level(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample, downsample_mode=downsample_mode)

        self.level3 = Level_Encoder_mfuse_4level(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample, downsample_mode=downsample_mode)

        self.just_backward = False
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, u0, u1, u2, u3 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, u0)
        c1 = (self.alpha1) * c1 + self.level1(c0, u1)
        c2 = (self.alpha2) * c2 + self.level2(c1, u2)
        c3 = (self.alpha3) * c3 + self.level3(c2, u3)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, C3, _, _, _, _ = ReverseFunctionSimpleMEncoder4level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, C3

    def _forward_reverse_first(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, C3, _, _, _, _ = ReverseFunctionSimpleMEncoder4level_first.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, C3
    def _forward_reverse_nofuse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]

        _, c0, c1, c2, c3 = ReverseFunction_nofuse.apply(
            local_funs, alpha, *args)
        return c0, c1, c2, c3

    def _forward_reverse_nofuse_first(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]

        _, c0, c1, c2, c3 = ReverseFunction_nofuse_first.apply(
            local_funs, alpha, *args)
        return c0, c1, c2, c3
    def _forward_backward(self, *args):
        with torch.no_grad():
            return self._forward_nonreverse(*args)
        # local_funs = [self.level0, self.level1, self.level2]
        # alpha = [self.alpha0, self.alpha1, self.alpha2]
        # _, c0, c1, c2 = ReverseFunctionMultisacleSimpleEncoder3level_Adapter.apply(
        #     local_funs, alpha, *args)
        #
        # return c0, c1, c2
    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.just_backward:
            return self._forward_backward(*args)
        elif self.save_memory:
            if self.downsample == 'None':
                x, c0, c1, c2, c3, _, _, _, _ = args
                if self.first_col:
                    return self._forward_reverse_nofuse_first(x, c0, c1, c2, c3)
                else:
                    return self._forward_reverse_nofuse(x, c0, c1, c2, c3)
            else:
                if self.first_col:
                    return self._forward_reverse_first(*args)
                else:
                    return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniFourierMEncoder3Level(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1, downsample=True) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4]
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # Level_Encoder_nofuse_3level
        self.level0 = Level_FourierEncoder_mfuse_3level(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample)

        self.level1 = Level_FourierEncoder_mfuse_3level(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample)

        self.level2 = Level_FourierEncoder_mfuse_3level(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample=downsample)

        # self.level3 = Level_Pred_kernel(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
        #                             train_size, ffn_expansion_factor)

        self.just_backward = False
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, u0, u1, u2 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, u0)
        c1 = (self.alpha1) * c1 + self.level1(c0, u1)
        c2 = (self.alpha2) * c2 + self.level2(c1, u2)
        # c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2]
        alpha = [self.alpha0, self.alpha1, self.alpha2]
        _, c0, c1, c2, _, _, _ = ReverseFunctionMultisacleSimpleEncoder3level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2
    def _forward_backward(self, *args):
        with torch.no_grad():
            return self._forward_nonreverse(*args)
        # local_funs = [self.level0, self.level1, self.level2]
        # alpha = [self.alpha0, self.alpha1, self.alpha2]
        # _, c0, c1, c2 = ReverseFunctionMultisacleSimpleEncoder3level_Adapter.apply(
        #     local_funs, alpha, *args)
        #
        # return c0, c1, c2
    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        # self._clamp_abs(self.alpha3.data, 1e-3)

        if self.just_backward:
            return self._forward_backward(*args)
        elif self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleMKerEncoder3Level(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4]
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # Level_Encoder_nofuse_3level
        self.level0 = Level_Encoder_mkerfuse_3level(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_Encoder_mkerfuse_3level(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_Encoder_mkerfuse_3level(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        # self.level3 = Level_Pred_kernel(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
        #                             train_size, ffn_expansion_factor)

        self.just_backward = False
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, u0, u1, u2 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, u0)
        c1 = (self.alpha1) * c1 + self.level1(c0, u1)
        c2 = (self.alpha2) * c2 + self.level2(c1, u2)
        # c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2]
        alpha = [self.alpha0, self.alpha1, self.alpha2]
        _, c0, c1, c2, _, _, _ = ReverseFunctionMultisacleSimpleEncoder3level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2
    def _forward_backward(self, *args):
        with torch.no_grad():
            return self._forward_nonreverse(*args)
        # local_funs = [self.level0, self.level1, self.level2]
        # alpha = [self.alpha0, self.alpha1, self.alpha2]
        # _, c0, c1, c2 = ReverseFunctionMultisacleSimpleEncoder3level_Adapter.apply(
        #     local_funs, alpha, *args)
        #
        # return c0, c1, c2
    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        # self._clamp_abs(self.alpha3.data, 1e-3)

        if self.just_backward:
            return self._forward_backward(*args)
        elif self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class Uni2ScaleEncoder3Level(nn.Module):
    def __init__(self, channels_max, channels_mid, layers_max, layers_mid, kernel_size, num_heads, first_col, dp_rates,
                 save_memory, baseblock_max='naf', baseblock_mid='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4]
        train_size_mid = [train_size[0] // 2, train_size[1] // 2]
        patch_size_mid = [patch_size[0] // 2, patch_size[1] // 2]
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels_max[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels_max[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels_max[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.alpha01 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels_mid[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha11 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels_mid[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha21 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels_mid[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # Level_Encoder_nofuse_3level
        self.level0 = Level_Encoder_maxmid_fuse_3level(0, channels_max, layers_max, kernel_size, first_col, dp_rates,
                                                 num_heads, baseblock_max,
                                                 train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_Encoder_maxmid_fuse_3level(1, channels_max, layers_max, kernel_size, first_col, dp_rates,
                                                 num_heads, baseblock_max,
                                                 train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_Encoder_maxmid_fuse_3level(2, channels_max, layers_max, kernel_size, first_col, dp_rates,
                                                 num_heads, baseblock_max,
                                                 train_size, patch_size, ffn_expansion_factor)

        self.level01 = Level_Encoder_midmax_fuse_3level(0, channels_mid, layers_mid, kernel_size, first_col, dp_rates,
                                                  num_heads,
                                                  baseblock_mid,
                                                  train_size_mid, patch_size_mid, ffn_expansion_factor)

        self.level11 = Level_Encoder_midmax_fuse_3level(1, channels_mid, layers_mid, kernel_size, first_col, dp_rates,
                                                  num_heads,
                                                  baseblock_mid,
                                                  train_size_mid, patch_size_mid, ffn_expansion_factor)

        self.level21 = Level_Encoder_midmax_fuse_3level(2, channels_mid, layers_mid, kernel_size, first_col, dp_rates,
                                                  num_heads,
                                                  baseblock_mid,
                                                  train_size_mid, patch_size_mid, ffn_expansion_factor)

        self.just_backward = False
    def _forward_nonreverse(self, *args):
        x_max, x_mid, max0, max1, max2, mid0, mid1, mid2 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        mid0 = (self.alpha01) * mid0 + self.level01(x_mid, x_max)
        max0 = (self.alpha0) * max0 + self.level0(x_max, x_mid)
        mid1 = (self.alpha11) * mid1 + self.level11(mid0, max0)
        max1 = (self.alpha1) * max1 + self.level1(max0, mid0)
        mid2 = (self.alpha21) * mid2 + self.level21(mid1, max1)
        max2 = (self.alpha2) * max2 + self.level2(max1, mid1)
        # c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return max0, max1, max2, mid0, mid1, mid2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level01, self.level11, self.level21]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha01, self.alpha11, self.alpha21]
        _,_, max0, max1, max2, mid0, mid1, mid2 = ReverseFunctionMultisacleEncoder3level.apply(
            local_funs, alpha, *args)

        return max0, max1, max2, mid0, mid1, mid2
    def _forward_backward(self, *args):
        with torch.no_grad():
            return self._forward_nonreverse(*args)
        # local_funs = [self.level0, self.level1, self.level2]
        # alpha = [self.alpha0, self.alpha1, self.alpha2]
        # _, c0, c1, c2 = ReverseFunctionMultisacleSimpleEncoder3level_Adapter.apply(
        #     local_funs, alpha, *args)
        #
        # return c0, c1, c2
    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        # self._clamp_abs(self.alpha3.data, 1e-3)

        if self.just_backward:
            return self._forward_backward(*args)
        elif self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniEncoder3Level(nn.Module):
    def __init__(self, channels, layers, layers_k, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        num_heads = [4, 2, 1]
        self.save_memory = save_memory
        self.kalpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.kalpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.kalpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.levelk0 = Level_Encoder_3level(0, channels, layers_k, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)

        self.levelk1 = Level_Encoder_3level(1, channels, layers_k, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)

        self.levelk2 = Level_Encoder_3level(2, channels, layers_k, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)

    def _forward_nonreverse(self, *args):
        k, k0, k1, k2 = args
        k0 = (self.kalpha0) * k0 + self.levelk0(k, k1)
        k1 = (self.kalpha1) * k1 + self.levelk1(k0, k2)
        k2 = (self.kalpha2) * k2 + self.levelk2(k1, None)

        return k0, k1, k2

    def _forward_reverse(self, *args):

        local_funs = [self.levelk0, self.levelk1, self.levelk2]
        alpha = [self.kalpha0, self.kalpha1, self.kalpha2]
        _, k0, k1, k2 = ReverseFunctionKernelFeature3level.apply(
            local_funs, alpha, *args)

        return k0, k1, k2

    def forward(self, *args):
        self._clamp_abs(self.kalpha0.data, 1e-3)
        self._clamp_abs(self.kalpha1.data, 1e-3)
        self._clamp_abs(self.kalpha2.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign

class UniDecoderKernelFeature3Level(nn.Module):
    def __init__(self, channels, layers, layers_k, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        num_heads = [4, 2, 1]
        self.save_memory = save_memory
        self.kalpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.kalpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.kalpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, kernel_size**2, 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.levelk0 = Level_Decoder_3level(0, channels, layers_k, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)

        self.levelk1 = Level_Decoder_3level(1, channels, layers_k, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)

        self.levelk2 = Level_Encoder_3level_kernel(2, channels, layers_k, kernel_size, first_col, dp_rates, num_heads,
                                                baseblock,
                                                train_size, ffn_expansion_factor)

    def _forward_nonreverse(self, *args):
        k, k0, k1, k2 = args
        k0 = (self.kalpha0) * k0 + self.levelk0(k, k1)
        k1 = (self.kalpha1) * k1 + self.levelk1(k0, k2)
        k2 = (self.kalpha2) * k2 + self.levelk2(k1, None)

        return k0, k1, k2

    def _forward_reverse(self, *args):

        local_funs = [self.levelk0, self.levelk1, self.levelk2]
        alpha = [self.kalpha0, self.kalpha1, self.kalpha2]
        _, k0, k1, k2 = ReverseFunctionKernelFeature3level.apply(
            local_funs, alpha, *args)

        return k0, k1, k2

    def forward(self, *args):
        self._clamp_abs(self.kalpha0.data, 1e-3)
        self._clamp_abs(self.kalpha1.data, 1e-3)
        self._clamp_abs(self.kalpha2.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniEncoderKernelAttn5Level(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[96, 96], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha4 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_Encoder_KernelPrior(0, channels, layers, kernel_size[0], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_Encoder_KernelPrior(1, channels, layers, kernel_size[1], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_Encoder_KernelPrior(2, channels, layers, kernel_size[2], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level3 = Level_SimpleEncoder_5(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)
        self.level4 = Level_SimpleEncoder_5(4, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, c4, k0, k1, k2 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1, k0)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2, k1)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3, k2)
        c3 = (self.alpha3) * c3 + self.level3(c2, c4)
        c4 = (self.alpha4) * c4 + self.level4(c3, None)
        return c0, c1, c2, c3, c4

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4, _, _, _ = ReverseFunctionKernelAttn5level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        self._clamp_abs(self.alpha4.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniDecoderKernelAttn5Level(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[96, 96], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha4 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_SimpleDecoder_5(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_SimpleDecoder_5(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_Decoder_KernelPrior(2, channels, layers, kernel_size[0], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level3 = Level_Decoder_KernelPrior(3, channels, layers, kernel_size[1], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level4 = Level_Decoder_KernelPrior(4, channels, layers, kernel_size[2], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, c4, k0, k1, k2 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3, k2)
        c3 = (self.alpha3) * c3 + self.level3(c2, c4, k1)
        c4 = (self.alpha4) * c4 + self.level4(c3, None, k0)
        return c0, c1, c2, c3, c4

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4, _, _, _ = ReverseFunctionDecKernelAttn5level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        self._clamp_abs(self.alpha4.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniDyDecoderKernelAttn5Level(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[96, 96], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha4 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_SimpleDyDecoder_5(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_SimpleDyDecoder_5(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_DyDecoder_KernelPrior(2, channels, layers, kernel_size[0], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level3 = Level_DyDecoder_KernelPrior(3, channels, layers, kernel_size[1], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level4 = Level_DyDecoder_KernelPrior(4, channels, layers, kernel_size[2], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, c4, k0, k1, k2, exit_index = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1, exit_index)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2, exit_index)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3, k2, exit_index)
        c3 = (self.alpha3) * c3 + self.level3(c2, c4, k1, exit_index)
        c4 = (self.alpha4) * c4 + self.level4(c3, None, k0, exit_index)
        return c0, c1, c2, c3, c4

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4, _, _, _, _ = ReverseFunctionDyDecKernelAttn5level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        self._clamp_abs(self.alpha4.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniDecoderKernelAttn5Level0(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha4 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_SimpleDecoder_5(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level1 = Level_SimpleDecoder_5(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level2 = Level_SimpleDecoder_5(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level3 = Level_SimpleDecoder_5(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level4 = Level_Decoder_KernelPrior(4, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, c4, k0 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, c4)
        c4 = (self.alpha4) * c4 + self.level4(c3, None, k0)
        return c0, c1, c2, c3, c4

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4, _ = ReverseFunctionDecKernelAttn5level0.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        self._clamp_abs(self.alpha4.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniEncoderKernelAttn5LevelXF(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[96, 96], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha4 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_Encoder_KernelFeaturePrior(0, channels, layers, kernel_size[2], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_Encoder_KernelFeaturePrior(1, channels, layers, kernel_size[1], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_Encoder_KernelFeaturePrior(2, channels, layers, kernel_size[0], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level3 = Level_SimpleEncoder_5(3, channels, layers, kernel_size[0], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)
        self.level4 = Level_SimpleEncoder_5(4, channels, layers, kernel_size[0], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1, k0, kf0)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2, k1, kf1)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3, k2, kf2)
        c3 = (self.alpha3) * c3 + self.level3(c2, c4)
        c4 = (self.alpha4) * c4 + self.level4(c3, None)
        return c0, c1, c2, c3, c4

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4, _, _, _, _, _, _ = ReverseFunctionKernelAttn5levelXF.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        self._clamp_abs(self.alpha4.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniDecoderKernelAttn5LevelXF(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[96, 96], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha4 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_SimpleDecoder_5(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_SimpleDecoder_5(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_Decoder_KernelFeaturePrior(2, channels, layers, kernel_size[0], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level3 = Level_Decoder_KernelFeaturePrior(3, channels, layers, kernel_size[1], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level4 = Level_Decoder_KernelFeaturePrior(4, channels, layers, kernel_size[2], first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, c4, k0, k1, k2, kf0, kf1, kf2 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3, k2, kf2)
        c3 = (self.alpha3) * c3 + self.level3(c2, c4, k1, kf1)
        c4 = (self.alpha4) * c4 + self.level4(c3, None, k0, kf0)
        return c0, c1, c2, c3, c4

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4, _, _, _, _, _, _ = ReverseFunctionDecKernelAttn5levelXF.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        self._clamp_abs(self.alpha4.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleEncoder4Level(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4, 8]
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level_Encoder_nofuse(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_Encoder_nofuse(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_Encoder_nofuse(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level3 = Level_Encoder_nofuse(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleEncoder4Level_(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1, downsample_mode='conv') -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4, 8]
        self.first_col = first_col
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level_Encoder_nofuse(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample_mode=downsample_mode)

        self.level1 = Level_Encoder_nofuse(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample_mode=downsample_mode)

        self.level2 = Level_Encoder_nofuse(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample_mode=downsample_mode)

        self.level3 = Level_Encoder_nofuse(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, downsample_mode=downsample_mode)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3
    def _forward_reverse_first(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction_first.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3
    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            if self.first_col:
                return self._forward_reverse_first(*args)
            else:
                return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleResEncoder3Level(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4, 8]
        kernel_size = [kernel_size[0], kernel_size[0], kernel_size[1], kernel_size[2]]
        self.save_memory = save_memory
        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # print(kernel_size)
        self.level0 = Level_Encoder_nodownup(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level1 = Level_Encoder_nodownup(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level2 = Level_Encoder_nodownup(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        # self.level3 = Level_Encoder_nodownup(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
        #                             train_size, ffn_expansion_factor)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        # print(x.shape, c1.shape, self.level1(c0, c2).shape)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, None)
        # c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2]
        alpha = [self.alpha0, self.alpha1, self.alpha2]
        _, c0, c1, c2 = ReverseFunctionSimpleDecoder3level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2

    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        # self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleResEncoder4Level(nn.Module):
    def __init__(self, channels, layers, kernel_size, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        # shortcut_scale_init_value = 0.5
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4, 8]
        self.save_memory = save_memory
        self.register_parameter('alpha0', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha1', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha2', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha3', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                                       requires_grad=True))
        # self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level_Encoder_nodownup(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level1 = Level_Encoder_nodownup(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level2 = Level_Encoder_nodownup(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level3 = Level_Encoder_nodownup(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleResNet4Level(nn.Module):
    def __init__(self, channels, layers, num_heads, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[256, 256], flip=[False, False, False, False], transpose=[False, False, False, False], sub_idx=1, skip_connect=False) -> None:
        super().__init__()
        # shortcut_scale_init_value = 0.5
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        kernel_size = 3
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [1, 2, 4, 8]
        self.save_memory = save_memory
        self.register_parameter('alpha0', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha1', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha2', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha3', nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                                       requires_grad=True))
        # self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None
        # self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
        #                            requires_grad=True) if shortcut_scale_init_value > 0 else None

        self.level0 = Level_ResNet_nodownup(0, channels, layers, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, flip=flip[0], transpose=transpose[0], skip_connect=skip_connect)

        self.level1 = Level_ResNet_nodownup(1, channels, layers, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, flip=flip[1], transpose=transpose[1], skip_connect=skip_connect)

        self.level2 = Level_ResNet_nodownup(2, channels, layers, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, flip=flip[2], transpose=transpose[2], skip_connect=skip_connect)

        self.level3 = Level_ResNet_nodownup(3, channels, layers, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor, flip=flip[3], transpose=transpose[3], skip_connect=skip_connect)

    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3 = args
        # print(x.shape, c0.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, None)
        return c0, c1, c2, c3

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3]
        _, c0, c1, c2, c3 = ReverseFunction.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3

    def forward(self, *args):

        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)

        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniSimpleEncoder5Level(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[96, 96], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha4 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_SimpleEncoder_5(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_SimpleEncoder_5(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_SimpleEncoder_5(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level3 = Level_SimpleEncoder_5(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level4 = Level_SimpleEncoder_5(4, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, c4 = args
        # print((self.alpha0 * c0).shape, x.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, c4)
        c4 = (self.alpha4) * c4 + self.level4(c3, None)
        return c0, c1, c2, c3, c4

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4 = ReverseFunction5Level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4

    def _forward_reverse_adapter(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4 = ReverseFunction5Level_Adapter.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        self._clamp_abs(self.alpha4.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign

class UniSimpleFusionEncoder5Level(nn.Module):
    def __init__(self, channels, layers, num_heads, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], patch_size=[96, 96], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        # num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha4 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_SimpleFusionEncoder_5(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level1 = Level_SimpleFusionEncoder_5(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level2 = Level_SimpleFusionEncoder_5(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level3 = Level_SimpleFusionEncoder_5(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)

        self.level4 = Level_SimpleFusionEncoder_5(4, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, patch_size, ffn_expansion_factor)
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, c4 = args
        # print((self.alpha0 * c0).shape, x.shape, self.level0(x, c1).shape)
        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, c4)
        c4 = (self.alpha4) * c4 + self.level4(c3, None)
        return c0, c1, c2, c3, c4

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4 = ReverseFunction5Level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4

    def _forward_reverse_adapter(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4 = ReverseFunction5Level_Adapter.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        self._clamp_abs(self.alpha4.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
class UniEncoder5Level(nn.Module):
    def __init__(self, channels, layers, layers_k, kernel_size, first_col, dp_rates, save_memory, baseblock='naf',
                 train_size=[256, 256], sub_idx=1) -> None:
        super().__init__()
        shortcut_scale_init_value = 0.5
        # layers = [1,1,1,1]
        ffn_expansion_factor = 1. + 1.66 * sub_idx
        num_heads = [16, 8, 4, 2, 1]
        self.save_memory = save_memory

        self.alpha0 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[0], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha1 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[1], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha2 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[2], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha3 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[3], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.alpha4 = nn.Parameter(shortcut_scale_init_value * torch.ones((1, channels[4], 1, 1)),
                                   requires_grad=True) if shortcut_scale_init_value > 0 else None
        self.level0 = Level_Encoder_5(0, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level1 = Level_Encoder_5(1, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level2 = Level_Encoder_5(2, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)

        self.level3 = Level_Encoder_5(3, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)
        self.level4 = Level_Encoder_5(4, channels, layers, kernel_size, first_col, dp_rates, num_heads, baseblock,
                                    train_size, ffn_expansion_factor)
    def _forward_nonreverse(self, *args):
        x, c0, c1, c2, c3, c4 = args

        c0 = (self.alpha0) * c0 + self.level0(x, c1)
        c1 = (self.alpha1) * c1 + self.level1(c0, c2)
        c2 = (self.alpha2) * c2 + self.level2(c1, c3)
        c3 = (self.alpha3) * c3 + self.level3(c2, c4)
        c4 = (self.alpha4) * c4 + self.level4(c3, None)
        return c0, c1, c2, c3, c4

    def _forward_reverse(self, *args):

        local_funs = [self.level0, self.level1, self.level2, self.level3, self.level4]
        alpha = [self.alpha0, self.alpha1, self.alpha2, self.alpha3, self.alpha4]
        _, c0, c1, c2, c3, c4 = ReverseFunction5Level.apply(
            local_funs, alpha, *args)

        return c0, c1, c2, c3, c4

    def forward(self, *args):
        self._clamp_abs(self.alpha0.data, 1e-3)
        self._clamp_abs(self.alpha1.data, 1e-3)
        self._clamp_abs(self.alpha2.data, 1e-3)
        self._clamp_abs(self.alpha3.data, 1e-3)
        self._clamp_abs(self.alpha4.data, 1e-3)
        if self.save_memory:
            return self._forward_reverse(*args)
        else:
            return self._forward_nonreverse(*args)

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign

class UFPNet(nn.Module):
    def __init__(self, img_channel=3, width=64, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1],
                 kernel_size=19):
        super().__init__()

        self.kernel_size = kernel_size
        self.kernel_extra = code_extra_mean_var(kernel_size)

        self.flow = KernelPrior(n_blocks=5, input_size=19 ** 2, hidden_size=25, n_hidden=1, kernel_size=19)
        # for k, v in self.flow.named_parameters():
        #     v.requires_grad = False
        # for k, v in self.kernel_extra.named_parameters():
        #     v.requires_grad = False
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        # self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
        #                         groups=1,
        #                         bias=True)

        self.encoders = nn.ModuleList()
        # self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        # self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.kernel_down = nn.ModuleList()

        chan = width
        for i, num in enumerate(enc_blk_nums):
            if num == 1:
                self.encoders.append(
                    nn.Sequential(*[NAFBlock_kernel(chan, kernel_size=kernel_size) for _ in range(num)]))
            else:
                self.encoders.append(nn.Sequential(*[NAFBlock(chan) for _ in range(num)]))
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            if i < 2:
                self.kernel_down.append(nn.Conv2d(kernel_size * kernel_size, kernel_size * kernel_size, 2, 2))
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )
        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        # with torch.no_grad():
        B = inp.shape[0]
        # with torch.no_grad():
        # kernel estimation: size [B, H*W, 19, 19]
        kernel_code, kernel_var = self.kernel_extra(inp)
        kernel_code = (kernel_code - torch.mean(kernel_code, dim=[2, 3], keepdim=True)) / torch.std(kernel_code,
                                                                                                    dim=[2, 3],
                                                                                                    keepdim=True)
        # code uncertainty
        sigma = kernel_var
        kernel_code_uncertain = kernel_code * torch.sqrt(1 - torch.square(sigma)) + torch.randn_like(
            kernel_code) * sigma

        kernel = generate_k(self.flow,
                            kernel_code_uncertain.reshape(kernel_code.shape[0] * kernel_code.shape[1], -1))
        kernel = kernel.reshape(kernel_code.shape[0], kernel_code.shape[1], self.kernel_size, self.kernel_size)
        # kernel_blur = kernel

        kernel = kernel.permute(0, 2, 3, 1).reshape(B, self.kernel_size * self.kernel_size, inp.shape[2],
                                                    inp.shape[3])

        x = self.intro(inp)

        encs = []
        kernels = [kernel]
        for i, encoder, down in zip(range(len(self.encoders)), self.encoders, self.downs):
            if len(encoder) == 1:
                x = encoder[0]([x, kernel])
                if i < 2:
                    kernel = self.kernel_down[i](kernel)
                    kernels.append(kernel)
            else:
                x = encoder(x)
            encs.append(x)
            x = down(x)
        # print(x.shape)
        # encs.reverse()
        e4 = self.middle_blks(x)
        encs.append(e4)
        encs.extend(kernels)
        return encs

class UFPNetKernel(nn.Module):
    def __init__(self, width=64, enc_blk_nums=[1, 1], kernel_size=19):
        super().__init__()

        self.kernel_size = kernel_size
        self.kernel_extra = code_extra_mean_var(kernel_size)
        for k, v in self.kernel_extra.named_parameters():
            v.requires_grad = False
        self.flow = KernelPrior(n_blocks=5, input_size=19 ** 2, hidden_size=25, n_hidden=1, kernel_size=19)
        for k, v in self.flow.named_parameters():
            v.requires_grad = False
        # self.kernel_down = nn.ModuleList()
        #
        # for _ in range(2):
        #     self.kernel_down.append(nn.Conv2d(kernel_size * kernel_size, kernel_size * kernel_size, 2, 2, bias=False))
        # for k, v in self.kernel_down.named_parameters():
        #     v.requires_grad = False
    def forward(self, inp):
        B = inp.shape[0]
        with torch.no_grad():
        # kernel estimation: size [B, H*W, 19, 19]
            kernel_code, kernel_var = self.kernel_extra(inp)
            kernel_code = (kernel_code - torch.mean(kernel_code, dim=[2, 3], keepdim=True)) / torch.std(kernel_code,
                                                                                                        dim=[2, 3],
                                                                                                        keepdim=True)
            # code uncertainty
            sigma = kernel_var
            kernel_code_uncertain = kernel_code * torch.sqrt(1 - torch.square(sigma)) + torch.randn_like(
                kernel_code) * sigma

            kernel = generate_k(self.flow,
                                kernel_code_uncertain.reshape(kernel_code.shape[0] * kernel_code.shape[1], -1))
            kernel = kernel.reshape(kernel_code.shape[0], kernel_code.shape[1], self.kernel_size, self.kernel_size)

            kernel = kernel.permute(0, 2, 3, 1).reshape(B, self.kernel_size * self.kernel_size, inp.shape[2],
                                                    inp.shape[3])

            # kernels = [kernel]
        # kernel1 = self.kernel_down[0](kernel)
        # kernel2 = self.kernel_down[1](kernel1)
        # kernel1 = torch.nn.functional.interpolate(kernel, scale_factor=[0.5, 0.5])
        # kernel2 = torch.nn.functional.interpolate(kernel1, scale_factor=[0.5, 0.5])
        # kernel3 = self.kernel_down[2](kernel2)
        # kernels.append(kernel1)
        # print('kernels: ', len(kernels))
        return kernel # [kernel, kernel1, kernel2]
class NAFNet(nn.Module):

    def __init__(self, width=64, img_channel=3, out_channels=3, middle_blk_num=1,
                 enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1], train_size=None,
                 num_heads_e=[1, 2, 4, 8], num_heads_m=[16], window_size_e=[64, 32, 16, 8], window_size_m=[8],
                 window_size_e_fft=[-1, -1, -1, -1], window_size_m_fft=[-1],
                 return_feat=False, attn_type='SCA'):
        super().__init__()
        self.grid = True
        self.train_size = train_size
        self.overlap_size = (32, 32)
        print(self.overlap_size)
        self.kernel_size = [train_size, train_size]
        self.return_feat = return_feat

        num_heads_d = num_heads_e[::-1]  # [8, 4, 2, 1]
        window_size_d = window_size_e[::-1]
        window_size_d_fft = window_size_e_fft[::-1]

        shift_size_e = [0, 0, 0, -1]
        shift_size_m = [-1]
        # shift_size_d = [-1, -1, -1, -1]
        shift_size_d = shift_size_e[::-1]

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)

        # self.fft_head = FFT_head()
        self.encoders = nn.ModuleList()

        self.middle_blks = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for i in range(len(enc_blk_nums)):
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, num_heads_e[i], window_size_e[i], window_size_e_fft[i], shift_size_e[i],
                               attn_type=attn_type) for _ in range(enc_blk_nums[i])]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2 * chan, 2, 2)
            )
            chan = chan * 2
        # print(NAFBlock)
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, num_heads_m[0], window_size_m[0], window_size_m_fft[0], shift_size_m[0],
                           attn_type=attn_type) for _ in range(middle_blk_num)]
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)
        # print(x.shape)
        x = self.middle_blks(x)
        # encs.reverse()
        encs.append(x)
        return encs


class Generate_Potential_Kernel(nn.Module):

    def __init__(self, width=64, num_kernel=5,
                 enc_blk_nums=[2, 2, 2],
                 num_heads=[4, 2, 1],
                 baseblock=[], train_size=256):
        super().__init__()

        # self.fft_head = FFT_head()
        self.encoders = nn.ModuleList()
        self.exs = nn.ModuleList()

        # chan = width
        channels = [width * 4, width * 2, width]
        self.intro = nn.Conv2d(in_channels=channels[0], out_channels=channels[0], kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        for i in range(len(enc_blk_nums)):
            if i < len(enc_blk_nums) - 1:
                self.exs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=3, padding=1,
                                  stride=1,
                                  groups=1,
                                  bias=True))
                )
            else:
                self.exs.append(
                    nn.Sequential(
                        nn.Identity())
                )

            modules = get_modules(baseblock[i], i, channels, enc_blk_nums, num_heads, train_size)
            self.encoders.append(
                nn.Sequential(*modules)
            )

        alpha = 1e-6
        normalization = 1.
        self.register_buffer('alpha', torch.ones(1) * alpha)
        self.register_buffer('normalization', torch.ones(1) * normalization)

    def post_process(self, x):
        # inverse process of pre_process in dataloader
        # x = x.view(x.shape[0], 1, int(self.kernel_size), int(self.kernel_size))
        x = ((torch.sigmoid(x) - self.alpha) / (1 - 2 * self.alpha))
        x = x * self.normalization
        return x  # / torch.sum(x, dim=[-2, -1], keepdim=True)
    def forward(self, inp):
        # print(inp)
        x = self.intro(inp)
        encs = []

        for encoder, ex in zip(self.encoders, self.exs):
            x = encoder(x)
            # x = torch.sigmoid(x) - 1.
            # x = kornia.geometry.spatial_softmax2d(x)
            
            x = self.post_process(x)
            encs.append(x)
            x = ex(x)
        encs.reverse()
        # print(len(encs))
        return encs
class Generate_Potential_Kernel1(nn.Module):

    def __init__(self, width=64, num_kernel=5,
                 enc_blk_nums=[2, 2, 2],
                 num_heads=[1, 2, 4],
                 baseblock=[], train_size=256):
        super().__init__()




        # self.fft_head = FFT_head()
        self.encoders = nn.ModuleList()
        self.exs = nn.ModuleList()

        # chan = width
        channels = [width, width * 2, width * 4]
        self.intro = nn.Conv2d(in_channels=channels[-1], out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)
        for i in range(len(enc_blk_nums)):
            if i < len(enc_blk_nums)-1:
                self.exs.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels=channels[i], out_channels=channels[i + 1], kernel_size=3, padding=1, stride=1,
                                  groups=1,
                                  bias=True))
                )
            else:
                self.exs.append(
                    nn.Sequential(
                        nn.Identity())
                )

            modules = get_modules(baseblock[i], i, channels, enc_blk_nums, num_heads, train_size)
            self.encoders.append(
                nn.Sequential(*modules)
            )



    def forward(self, inp):
        x = self.intro(inp)
        encs = []

        for encoder, ex in zip(self.encoders, self.exs):
            x = encoder(x)
            # x = torch.sigmoid(x) - 1.
            x = kornia.geometry.spatial_softmax2d(x)
            encs.append(x)
            x = ex(x)

        # print(len(encs))
        return encs
class DegradationModel(nn.Module):
    def __init__(self, kernel_size=19, divisor_=False):
        super(DegradationModel, self).__init__()
        self.blur_layer = BatchBlur_SV(l=kernel_size)# , divisor_=divisor_) # replication

    def forward(self, image, kernel):
        return self.blur_layer(image, kernel)
def _compute_padding(kernel_size: List[int]) -> List[int]:
    """Compute padding tuple."""
    # 4 or 6 ints:  (padding_left, padding_right,padding_top,padding_bottom)
    # https://pytorch.org/docs/stable/nn.html#torch.nn.functional.pad
    if len(kernel_size) < 2:
        raise AssertionError(kernel_size)
    computed = [k - 1 for k in kernel_size]

    # for even kernels we need to do asymmetric padding :(
    out_padding = 2 * len(kernel_size) * [0]

    for i in range(len(kernel_size)):
        computed_tmp = computed[-(i + 1)]

        pad_front = computed_tmp // 2
        pad_rear = computed_tmp - pad_front

        out_padding[2 * i + 0] = pad_front
        out_padding[2 * i + 1] = pad_rear

    return out_padding
def mfilter2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    border_type: str = 'reflect',
    normalized: bool = False,
    padding: str = 'same',
) -> torch.Tensor:

    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input input is not torch.Tensor. Got {type(input)}")

    if not isinstance(kernel, torch.Tensor):
        raise TypeError(f"Input kernel is not torch.Tensor. Got {type(kernel)}")

    if not isinstance(border_type, str):
        raise TypeError(f"Input border_type is not string. Got {type(border_type)}")

    if border_type not in ['constant', 'reflect', 'replicate', 'circular']:
        raise ValueError(
            f"Invalid border type, we expect 'constant', \
        'reflect', 'replicate', 'circular'. Got:{border_type}"
        )

    if not isinstance(padding, str):
        raise TypeError(f"Input padding is not string. Got {type(padding)}")

    if padding not in ['valid', 'same']:
        raise ValueError(f"Invalid padding mode, we expect 'valid' or 'same'. Got: {padding}")

    if not len(input.shape) == 4:
        raise ValueError(f"Invalid input shape, we expect BxCxHxW. Got: {input.shape}")

    if (not len(kernel.shape) == 3) and not ((kernel.shape[0] == 0) or (kernel.shape[0] == input.shape[0])):
        raise ValueError(f"Invalid kernel shape, we expect 1xHxW or BxHxW. Got: {kernel.shape}")

    # prepare kernel
    b, c, h, w = input.shape
    # tmp_kernel: torch.Tensor = kernel.unsqueeze(1).to(input)

    tmp_kernel = kernel # tmp_kernel.expand(-1, c, -1, -1)

    height, width = tmp_kernel.shape[-2:]

    # pad the input tensor
    if padding == 'same':
        padding_shape: List[int] = _compute_padding([height, width])
        input = F.pad(input, padding_shape, mode=border_type)

    # kernel and input tensor reshape to align element-wise or batch-wise params
    tmp_kernel = tmp_kernel.reshape(-1, 1, height, width)
    input = input.view(-1, tmp_kernel.size(0), input.size(-2), input.size(-1))

    # convolve the tensor with the kernel.
    output = F.conv2d(input, tmp_kernel, groups=tmp_kernel.size(0), padding=0, stride=1)

    if padding == 'same':
        out = output.view(b, c, h, w)
    else:
        out = output.view(b, c, h - height + 1, w - width + 1)

    return out


def reblurfilter(image, kernel, mask):
    num_k = kernel.shape[1]
    ch = image.shape[1]
    image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
    kernel = kernel.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    image = rearrange(image, 'b k c h w -> b (k c) h w')
    kernel = rearrange(kernel, 'b k c h w -> b (k c) h w')
    image_blur = mfilter2d(image, kernel)
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    image_blur = rearrange(image_blur, 'b (k c) h w -> b k c h w', k=num_k, c=ch)
    mask = mask.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    image_blur = torch.sum(image_blur * mask, dim=1)
    return image_blur
def reblurfilter_fft(image, kernel, mask=None, method='reblur', NSR=1e-5):
    num_k = kernel.shape[1]
    ch = image.shape[1]
    ks = max(kernel.shape[-1], kernel.shape[-2])
    ks = ks // 2
    dim = (ks, ks, ks, ks)
    if mask is not None:
        mask = mask.unsqueeze(2).expand(-1, -1, ch, -1, -1)

    b, nk = kernel.shape[:2]
    h, w = image.shape[-2:]
    otf = convert_psf2otf(kernel, (b, nk, h+2*ks, w+2*ks))
    otf = otf.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    otf = rearrange(otf, 'b k c h w -> b (k c) h w')
    if method == 'deblur':
        # otf = otf / (torch.abs(otf)+1e-7)
        image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
        # if mask is not None:
        #     image = image * mask
        image = rearrange(image, 'b k c h w -> b (k c) h w')
        image = torch.nn.functional.pad(image, dim, "replicate")

        otf = torch.conj(otf)
        otf = torch.exp(1j*torch.angle(otf))
        Image_blur = torch.fft.rfft2(image) * otf
        # Image_blur = torch.fft.rfft2(image)
        image_blur = torch.fft.irfft2(Image_blur)[:, :, ks:-ks, ks:-ks].contiguous()
        # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
        image_blur = rearrange(image_blur, 'b (k c) h w -> b k c h w', k=num_k, c=ch)
        if mask is not None:
            image_blur = image_blur * mask
        # image_blur = torch.sum(image_blur, dim=1)
        # if mask is not None:
        #     image = image * mask
        # if mask is not None:
        #     image_blur = torch.sum(image_blur * mask, dim=1)
        # else:
        #     image_blur = torch.sum(image_blur, dim=1)
        return torch.sum(image_blur, dim=1)
    else:
        # otf = otf / (torch.abs(otf)+1e-7)
        image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
        # if mask is not None:
        #     image = image * mask
        image = rearrange(image, 'b k c h w -> b (k c) h w')
        image = torch.nn.functional.pad(image, dim, "replicate")
        Image_blur = torch.fft.rfft2(image) * otf
        # Image_blur = torch.fft.rfft2(image)
        image_blur = torch.fft.irfft2(Image_blur)[:, :, ks:-ks, ks:-ks].contiguous()
        # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
        image_blur = rearrange(image_blur, 'b (k c) h w -> b k c h w', k=num_k, c=ch)
        if mask is not None:
            image_blur = image_blur * mask
        # image_blur = torch.sum(image_blur, dim=1)
        # if mask is not None:
        #     image = image * mask
        # if mask is not None:
        #     image_blur = torch.sum(image_blur * mask, dim=1)
        # else:
        #     image_blur = torch.sum(image_blur, dim=1)
        return torch.sum(image_blur, dim=1)

def reblurfilter_fftv1(image, kernel, mask=None, method='reblur', NSR=1e-5):
    num_k = kernel.shape[1]
    ch = image.shape[1]
    ks = max(kernel.shape[-1], kernel.shape[-2])
    ks = ks // 2
    dim = (ks, ks, ks, ks)
    if mask is not None:
        mask = mask.unsqueeze(2).expand(-1, -1, ch, -1, -1)

    b, nk = kernel.shape[:2]
    h, w = image.shape[-2:]
    otf = convert_psf2otf(kernel, (b, nk, h+2*ks, w+2*ks))
    otf = otf.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    otf = rearrange(otf, 'b k c h w -> b (k c) h w')
    if method == 'deblur':

        image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
        # if mask is not None:
        #     image = image * mask
        # image = image / (mask*num_k)
        image = rearrange(image, 'b k c h w -> b (k c) h w')
        image = torch.nn.functional.pad(image, dim, "replicate")
        # NSR = NSR.unsqueeze(1).expand(-1, num_k, -1, -1, -1) replicate
        #
        # NSR = rearrange(NSR, 'b k c h w -> b (k c) h w')
        # otf = torch.conj(otf) / (torch.abs(otf) ** 2 + NSR)
        otf = torch.conj(otf) / (torch.abs(otf) ** 2 + 1e-7)
        # otf = torch.conj(otf) / (torch.abs(otf) + NSR)
        # otf = torch.conj(otf) / (torch.abs(otf)+1e-7)
        # otf_abs = torch.abs(otf)
        # otf = torch.conj(otf) / (otf_abs*otf_abs*otf_abs + NSR)
        Image_deblur = torch.fft.rfft2(image) * otf

        Image_deblur = torch.fft.irfft2(Image_deblur)[:, :, ks:-ks, ks:-ks].contiguous()
        # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
        Image_deblur = rearrange(Image_deblur, 'b (k c) h w -> b k c h w', k=num_k, c=ch)
        if mask is not None:
            Image_deblur = torch.sum(Image_deblur * mask, dim=1)
        else:
            Image_deblur = torch.sum(Image_deblur, dim=1)
        # Image_deblur = torch.sum(Image_deblur * mask, dim=1)
        return Image_deblur
    else:
        # otf = otf / (torch.abs(otf)+1e-7)
        image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
        # if mask is not None:
        #     image = image * mask
        image = rearrange(image, 'b k c h w -> b (k c) h w')
        image = torch.nn.functional.pad(image, dim, "replicate")
        Image_blur = torch.fft.rfft2(image) * otf
        # Image_blur = torch.fft.rfft2(image)
        image_blur = torch.fft.irfft2(Image_blur)[:, :, ks:-ks, ks:-ks].contiguous()
        # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
        image_blur = rearrange(image_blur, 'b (k c) h w -> b k c h w', k=num_k, c=ch)
        if mask is not None:
            image_blur = image_blur * mask
        # image_blur = torch.sum(image_blur, dim=1)
        # if mask is not None:
        #     image = image * mask
        # if mask is not None:
        #     image_blur = torch.sum(image_blur * mask, dim=1)
        # else:
        #     image_blur = torch.sum(image_blur, dim=1)
        return torch.sum(image_blur, dim=1)
def deblurfilter_fft(image, kernel, mask, NSR):
    num_k = kernel.shape[1]
    ch = image.shape[1]
    ks = max(kernel.shape[-1], kernel.shape[-2])
    dim = (ks, ks, ks, ks)
    image = torch.nn.functional.pad(image, dim, "replicate")
    image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    image = rearrange(image, 'b k c h w -> b (k c) h w')

    otf = convert_psf2otf(kernel, image.size())
    otf = torch.conj(otf) / (otf_mag ** 2 + NSR)
    otf = otf.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    otf = rearrange(otf, 'b k c h w -> b (k c) h w')
    Image_blur = torch.fft.rfft2(image) * otf

    image_blur = torch.fft.irfft2(Image_blur)[:, :, ks:-ks, ks:-ks].contiguous()
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    image_blur = rearrange(image_blur, 'b (k c) h w -> b k c h w', k=num_k, c=ch)
    mask = mask.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    image_blur = torch.sum(image_blur * mask, dim=1)
    return image_blur
def deblurfeaturefilter_fft(image, kernel, groups, NSR):
    # num_k = kernel.shape[1]
    # ch = image.shape[1]
    ks = max(kernel.shape[-1], kernel.shape[-2])
    dim = (ks, ks, ks, ks)
    image = torch.nn.functional.pad(image, dim, "replicate")
    # image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    image = rearrange(image, 'b (g c) h w -> b g c h w', g=groups)

    otf = convert_psf2otf(kernel, image.size())
    otf = torch.conj(otf) / (otf_mag ** 2 + NSR)
    otf = otf.unsqueeze(1).expand(-1, groups, -1, -1, -1)
    # otf = rearrange(otf, 'b k c h w -> b (k c) h w')
    Image_blur = torch.fft.rfft2(image) * otf

    image_blur = torch.fft.irfft2(Image_blur)[:, :, ks:-ks, ks:-ks].contiguous()
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    image_blur = rearrange(image_blur, 'b g c h w -> b (g c) h w')

    return image_blur
def deblurWeinerfilter(image, kernel, mask, maxiter=100):
    ks_h, ks_w = kernel.shape[-2:]
    ks_h = ks_h // 2
    ks_w = ks_w // 2
    dim = (ks_h, ks_w, ks_h, ks_w)
    # print(dim)
    image = F.pad(image, dim, "replicate")
    num_k = kernel.shape[1]
    ch = image.shape[1]
    mask = mask.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
    kernel = kernel.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    image = rearrange(image, 'b k c h w -> (b k c) h w')
    kernel = rearrange(kernel, 'b k c h w -> (b k c) h w')
    image_deblur = ModifiedWiener(image.unsqueeze(1), kernel.unsqueeze(1), maxiter=maxiter)
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    image_deblur = rearrange(image_deblur, '(b k c) z h w -> b k (c z) h w', k=num_k, c=ch)

    # print(image_deblur.shape)
    image_deblur = image_deblur[..., ks_h:-ks_h, ks_w:-ks_w].contiguous()
    image_deblur = torch.sum(image_deblur * mask, dim=1)
    return image_deblur

def deblurfilter(image, kernel, mask):
    ks_h, ks_w = kernel.shape[-2:]
    ks_h = ks_h // 2
    ks_w = ks_w // 2
    dim = (ks_h, ks_w, ks_h, ks_w)
    # print(dim)
    image = F.pad(image, dim, "replicate")
    num_k = kernel.shape[1]
    ch = image.shape[1]
    mask = mask.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
    kernel = kernel.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    image = rearrange(image, 'b k c h w -> (b k c) h w')
    kernel = rearrange(kernel, 'b k c h w -> (b k c) h w')
    # Fx = torch.fft.rfft2()
    image_deblur = direct_deconv(image.unsqueeze(1), kernel.unsqueeze(1))
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    image_deblur = rearrange(image_deblur, '(b k c) z h w -> b k (c z) h w', k=num_k, c=ch)

    # print(image_deblur.shape)
    image_deblur = image_deblur[..., ks_h:-ks_h, ks_w:-ks_w].contiguous()
    image_deblur = torch.sum(image_deblur * mask, dim=1)
    return image_deblur

def deblurDWeinerfilter(image, kernel, mask, noise):
    ks_h, ks_w = kernel.shape[-2:]
    ks_h = ks_h // 2
    ks_w = ks_w // 2
    dim = (ks_h, ks_w, ks_h, ks_w)
    # print(dim)
    image = F.pad(image, dim, "replicate")
    num_k = kernel.shape[1]
    ch = image.shape[1]
    mask = mask.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
    kernel = kernel.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    image = rearrange(image, 'b k c h w -> (b k c) h w')
    kernel = rearrange(kernel, 'b k c h w -> (b k c) h w')
    noise = noise.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    noise = rearrange(noise, 'b k c h w -> (b k c) h w')
    # Fx = torch.fft.rfft2()
    image_deblur = get_uperleft_denominator(image.unsqueeze(1), kernel.unsqueeze(1), noise.unsqueeze(1))
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    image_deblur = rearrange(image_deblur, '(b k c) z h w -> b k (c z) h w', k=num_k, c=ch)

    # print(image_deblur.shape)
    image_deblur = image_deblur[..., ks_h:-ks_h, ks_w:-ks_w].contiguous()
    image_deblur = torch.sum(image_deblur * mask, dim=1)
    return image_deblur
if __name__ == '__main__':
    chan = 65
    num_heads = 1
    ker_size = 65
    modelA = UniSimpleMEncoder3Level([16, 32, 64], [1, 1, 1], 3, [1, 2, 4], False, [0., 0., 0.], save_memory=False, downsample='concat')

    modelA = modelA.cuda()

    x0, x1, x2, x3 = torch.randn((1, 16, 256, 256)).cuda(), 0, 0, 0
    u1, u2, u3 = torch.randn((1, 16, 256, 256)).cuda(), torch.randn((1, 32, 128, 128)).cuda(), torch.randn((1, 64, 64, 64)).cuda()

    y1, y2, y3 = modelA(x0, x1, x2, x3, u1, u2, u3)


    loss1 = torch.mean(y1)
    loss1.backward()

    x0_ = x0.clone()
    x1_, x2_, x3_ = 0, 0, 0
    u1_, u2_, u3_ = u1.clone(), u2.clone(), u3.clone()

    modelB = UniSimpleMEncoder3Level([16, 32, 64], [1, 1, 1], 3, [1, 2, 4], False, [0., 0., 0.], save_memory=True,
                                     downsample='concat')
    modelB.load_state_dict(modelA.state_dict())
    modelB = modelB.cuda()
    y1_, y2_, y3_ = modelB(x0_, x1_, x2_, x3_, u1_, u2_, u3_)

    print(torch.allclose(y1, y1_, 1e-5))
    loss2 = torch.ones(1, requires_grad=True)
    loss2.backward()
    print(y1.grad, y1_.grad)
    print(torch.allclose(x0.grad, x0_.grad, 1e-5))
    # kernel = kornia.filters.get_motion_kernel2d(ker_size, angle=90)
    # # print(kernel.shape)
    # kernel[:, ker_size//2:, :] *= 0
    # kernel = kernel / torch.sum(kernel)
    #
    # otf = torch.fft.rfft2(kernel)
    # a = torch.abs(otf) * torch.exp(1j*torch.angle(otf))
    # print(torch.sum(torch.abs(a-otf)))
    # kernel = kernel.cuda()
    # print(kernel.shape)
    # # print(kernel)
    # x = torch.randn(3, 4, 4)
    # print(x)
    # y = torch.randn(4, 4)
    # mask = y.ge(0.5)
    # print(mask)
    # z = torch.masked_select(x, mask)
    # z_ = x.masked_scatter(mask, z+1)
    # print(z.view(3, -1), z.view(3, -1).shape)
    # print(z_)
    # print(z.shape, z_.shape)
    # kernel = kernel.unsqueeze(1)
    # kernel = repeat(kernel, 'b c h w -> b (r c) h w', r=num_heads)
    # image = cv2.imread('/home/ubuntu/90t/personal_data/mxt/Dataset/MotionBlur/GoPro/val/target_crops/0.png')
    # x = kornia.image_to_tensor(image / 255., keepdim=False)
    # x_gray = kornia.color.bgr_to_grayscale(x)
    # x = x_gray.cuda()
    # x = repeat(x, 'b c h w -> b (r c) h w', r=chan*num_heads)
    # # x = torch.randn((1, 32, 48, 64)).cuda()
    #
    # net = Ker_Recoord(chan, num_heads, ker_size, div_ker=False, pad=7).cuda()
    # y = net(x, kernel)
    # # y = kornia.enhance.normalize_min_max(y, 0., 1.)
    # y = y * 255.
    # print(y.shape)
    # y = kornia.tensor_to_image(y, keepdim=False)
    # out_root = '/home/ubuntu/90t/personal_data/mxt/MXT/RevIR/Motion_Deblurring/results_redeconv'
    # os.makedirs(out_root, exist_ok=True)
    # kernel = kornia.tensor_to_image(kernel/kernel.max() * 255., keepdim=False)
    # cv2.imwrite(os.path.join(out_root, 'ker.png'), kernel)
    # x_gray = kornia.tensor_to_image(x_gray * 255., keepdim=False)
    # cv2.imwrite(os.path.join(out_root, '0x.png'), x_gray)
    # for c in range(y.shape[-1]):
    #     cv2.imwrite(os.path.join(out_root, str(c) +'.png'), y[:, :, c])

    # x = kornia.enhance.normalize_min_max(x, 0., 1.)
    # print(kernel)
    # otf = convert_psf2otf_dim3(kernel, (1, 64, 64))
    # # kernel_shift = torch.fft.ifftshift(kernel, dim=[-2, -1])
    # # otf = torch.fft.fft2(kernel_shift, dim=[-2, -1])
    # otf_mag = torch.abs(otf)
    # otf_phase = torch.angle(otf)
    # # out = blur_fft / (torch.exp(otf_phase * 1j) * (otf_mag + 1e-7))
    # # ker_inv_fft = 1 / (torch.exp(otf_phase * 1j) * (otf_mag + 1e-5))
    # b_i = kornia.filters.median_blur(x, [3, 3])
    # NSR = torch.std(x - b_i, dim=[-2, -1], keepdim=True) / torch.var(x, dim=[-2, -1], keepdim=True)
    # print(NSR)
    # ker_inv_fft = torch.conj(otf) / (otf_mag ** 2 + 1e-7)
    # ker_inv = torch.fft.irfft2(ker_inv_fft, dim=[-2, -1]) # .real
    # ker_inv = torch.fft.fftshift(ker_inv, dim=[-2, -1])
    # # ker_inv = kornia.geometry.center_crop(ker_inv.unsqueeze(0), [19, 19]).squeeze(0)
    # # print(ker_inv)
    #
    # x_f = torch.fft.rfft2(x)
    # y_f = x_f * otf
    # y_rf = y_f * (ker_inv_fft)
    # y_r = torch.fft.irfft2(y_rf)
    # y_1 = torch.fft.irfft2(y_f)
    # y = kornia.filters.filter2d(x, kernel)
    # z = kornia.filters.filter2d(y, ker_inv)
    # z = torch.clamp(z, 0., 1.)
    # x = x[:, :, 19:-19, 19:-19]
    # y = y[:, :, 19:-19, 19:-19]
    # y_1 = y_1[:, :, 19:-19, 19:-19]
    # y_r = y_r[:, :, 19:-19, 19:-19]
    # z = z[:, :, 19:-19, 19:-19]
    # # print(x)
    # # print(x - z)
    #
    # print(torch.mean(torch.abs(y_1 - y)))
    # print(torch.mean(torch.abs(y_rf - x_f)))
    # print(torch.mean(torch.abs(y_r - x)))
    # print(torch.mean(torch.abs(x-z)))

# class DeepRev(nn.Module):
#     def __init__(self, width=64, in_channels=3, out_channels=3, num_kernel=5, encoder_layers=[1, 1, 1, 28, 1], kernel_layers=[2, 2, 2],
#                  decoder_layers=[1, 1, 1, 1], baseblock_kernel=['FFTRes'], baseblock_enc=['naf'], baseblock_dec=['naf'],
#                  kernel_size=19, drop_path=0.0, train_size=[256, 256],  # save_memory=True,
#                  save_memory=True,
#                  pretrain=False,
#                  state_dict_pth=None,
#                  # state_dict_pth_subdecoder=None,
#                  # state_dict_pth_subkernel=None,
#                  combine_train=False, test_only=True,
#                  exit_threshold=0.5,
#                  reblur_loss_weight=0.5,
#                  deblur_loss_weight=0.25) -> None:
#         super().__init__()
#         # self.num_subnet = 2
#         self.num_kernel_extra = len(baseblock_kernel)
#         self.num_subnet = len(baseblock_enc)
#         self.num_subdenet = len(baseblock_dec)
#         self.channels = width
#         self.test_only = test_only
#         self.exit_threshold = exit_threshold
#         self.pretrain = pretrain
#         self.out_channels = out_channels
#         self.num_kernel = num_kernel
#         # self.use_amp = use_amp
#         # self.num_subnet = num_subnet
#         self.combine_train = combine_train
#         channels_kernel_dec = [width, width * 2]
#         channels_dec = [width, width * 2, width * 4, width * 8]
#         channels_kernel_enc = [width, width * 2, width * 4, num_kernel]
#         channels_enc = [width, width * 2, width * 4, width * 8, width * 16]
#         channels_dec.reverse()
#         channels_kernel_dec.reverse()
#         self.subdenets = nn.ModuleList()
#         self.subnets = nn.ModuleList()
#         self.subknets = nn.ModuleList()
#         # self.flow = nn.ModuleList()
#         # self.Conv_tail = nn.ModuleList()
#         # self.kernel_down = nn.ModuleList()
#         pad_1 = kernel_size // 2
#         pad_2 = kernel_size // 2 if kernel_size % 2 == 1 else kernel_size // 2 - 1
#         self.pad_1 = pad_1
#         self.pad_2 = pad_2
#         # self.DegradationModel = DegradationModel(kernel_size=kernel_size, divisor_=False)
#         dp_rate = [x.item() for x in torch.linspace(0, drop_path, sum(decoder_layers))]
#         for i in range(self.num_kernel_extra):
#             first_col = True if i == 0 else False
#             # first_col = False
#             # if first_col:
#             self.subknets.append(UniEncoderKernelFeature4Level(
#                 channels=channels_kernel_enc, layers=kernel_layers, kernel_size=kernel_size,
#                 first_col=first_col, dp_rates=dp_rate, save_memory=save_memory, num_kernel=num_kernel,
#                 baseblock=baseblock_kernel[i], train_size=train_size, sub_idx=(i + 1) / self.num_subdenet))
#             # self.flow.append(KernelPrior(n_blocks=5, input_size=kernel_size ** 2, hidden_size=25, n_hidden=1, kernel_size=kernel_size))
#             # self.Conv_tail.append(kernel_extra_conv_tail_mean_var(channels_kernel_enc[-1], self.kernel_size * self.kernel_size))
#             self.subnets.append(UniEncoderKernelAttn5Level(
#                 channels=channels_enc, layers=encoder_layers, layers_k=kernel_layers, kernel_size=kernel_size,
#                 first_col=first_col, dp_rates=dp_rate, save_memory=save_memory,
#                 baseblock=baseblock_enc[i], train_size=train_size, sub_idx=(i + 1) / self.num_subdenet))
#         for i in range(self.num_kernel_extra, self.num_subnet):
#             first_col = True if i == 0 else False
#
#             self.subnets.append(UniEncoder5Level(
#                 channels=channels_enc, layers=encoder_layers, layers_k=kernel_layers, kernel_size=kernel_size,
#                 first_col=first_col, dp_rates=dp_rate, save_memory=save_memory,
#                 baseblock=baseblock_enc[i], train_size=train_size, sub_idx=(i + 1) / self.num_subdenet))
#         for i in range(self.num_subdenet):
#             first_col = False
#             self.subdenets.append(UniDecoder(channels=channels_dec, layers=decoder_layers, kernel_size=kernel_size,
#                                              first_col=first_col, dp_rates=dp_rate, save_memory=save_memory,
#                                              baseblock=baseblock_dec[i], train_size=train_size, sub_idx=(i+1)/self.num_subdenet))
#         # if state_dict_pth_subkernel is not None:
#         #     for i in range(self.num_kernel_extra):
#         #         self.subknets[i].load_pretain_model(state_dict_pth_subkernel)
#         # if state_dict_pth_subencoder is not None:
#         #     for i in range(self.num_subnet):
#         #         self.subnets[i].load_pretain_model(state_dict_pth_subencoder)
#         # if state_dict_pth_subdecoder is not None:
#         #     for i in range(self.num_subdenet):
#         #         self.subdenets[i].load_pretain_model(state_dict_pth_subdecoder)
#
#
#         self.outputs = nn.ModuleList()
#
#
#         self.stems = nn.ModuleList()
#         for i in range(self.num_subnet):
#             self.stems.append(nn.Sequential(
#                 nn.Conv2d(in_channels, width, kernel_size=3, stride=1, padding=1, bias=True)
#                 # LayerNorm(channels[0], eps=1e-6, data_format="channels_first")
#             ))
#         self.kernel_stems = nn.ModuleList()
#         # self.kernel_extras = nn.ModuleList()
#         for i in range(self.num_kernel_extra):
#             self.kernel_stems.append(nn.Sequential(
#                 nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=True)
#             # PromptGenBlock(3, prompt_dim=kernel_size**2, prompt_len=5, prompt_size=train_size, num_heads=1)
#             ))
#             # self.kernel_extras.append(nn.Sequential(
#             #     nn.AdaptiveAvgPool2d((kernel_size, kernel_size)),
#             #     nn.Conv2d(channels_kernel_enc[-1], num_kernel, kernel_size=3, stride=1, padding=1, bias=True)
#             #     # PromptGenBlock(3, prompt_dim=kernel_size**2, prompt_len=5, prompt_size=train_size, num_heads=1)
#             # ))
#             self.outputs.append(nn.Sequential(
#                 nn.Conv2d(width, out_channels+num_kernel, kernel_size=3, stride=1, padding=1, bias=True)
#                 # LayerNorm(channels[0], eps=1e-6, data_format="channels_first")
#             ))
#         for i in range(self.num_kernel_extra, self.num_subdenet):
#             self.outputs.append(nn.Sequential(
#                 nn.Conv2d(width, out_channels, kernel_size=3, stride=1, padding=1, bias=True)
#                 # LayerNorm(channels[0], eps=1e-6, data_format="channels_first")
#             ))
#         # alpha = 1e-6
#         # normalization = 1.
#         # self.register_buffer('alpha', torch.ones(1) * alpha)
#         # self.register_buffer('normalization', torch.ones(1) * normalization)
#         if self.num_kernel_extra > 0:
#             self.loss_reblur = FreqLoss(loss_weight=reblur_loss_weight)
#             self.loss_deblur = FreqLoss(loss_weight=deblur_loss_weight)
#         if state_dict_pth is not None:
#             self.load_pretain_model(state_dict_pth)
#
#
#     def load_pretain_model(self, state_dict_pth):
#         checkpoint = torch.load(state_dict_pth)
#
#         state_dict = checkpoint["params"]
#         kernel_state_dict = OrderedDict()
#         subnet_state_dict = OrderedDict()
#         subdenet_state_dict = OrderedDict()
#         for k, v in state_dict.items():
#             # print(k)
#
#             # v = repeat(v, )
#             if k[:8] == 'subknets':
#                 name = k[11:]  # remove `module.`
#                 kernel_state_dict[name] = v
#             if k[:7] == 'subnets':
#                 name = k[10:]  # remove `module.`
#                 subnet_state_dict[name] = v
#             if k[:9] == 'subdenets':
#                 name = k[12:]  # remove `module.`
#                 subdenet_state_dict[name] = v
#         strict = False
#         for i in range(self.num_kernel_extra):
#             self.subknets[i].load_state_dict(kernel_state_dict, strict=strict)
#
#         for i in range(self.num_subnet):
#             self.subnets[i].load_state_dict(subnet_state_dict, strict=strict)
#
#         for i in range(self.num_subdenet):
#             self.subdenets[i].load_state_dict(subdenet_state_dict, strict=strict)
#         print('-----------load pretrained sub-models from ' + state_dict_pth + '----------------')
#     def post_process(self, x):
#         # inverse process of pre_process in dataloader
#         # x = x.view(x.shape[0], 1, int(self.kernel_size), int(self.kernel_size))
#         x = ((torch.sigmoid(x) - self.alpha) / (1 - 2 * self.alpha))
#         x = x * self.normalization
#         return x # / torch.sum(x, dim=[-2, -1], keepdim=True)
#     def forward(self, img, gt=None):
#         x_tmp_out_list = []
#
#         # e0, e1, e2, e3, e4, k0, k1, k2 = features
#         k0, k1, k2, kernel = 0, 0, 0, 0
#         e0, e1, e2, e3, e4 = 0, 0, 0, 0, 0
#         if self.num_kernel_extra > 0:
#             reblur_loss = 0.
#             deblur_loss = 0.
#         # img_blur3 = torch.nn.functional.interpolate(img, scale_factor=[0.25, 0.25])
#         assert self.num_subdenet >= self.num_subnet
#         for i in range(self.num_kernel_extra):
#
#             k = self.kernel_stems[i](img)
#             k0, k1, k2, kernel = self.subknets[i](k, k0, k1, k2, kernel)
#             x = self.stems[i](img)
#             # kernel1 = torch.nn.functional.interpolate(k2, scale_factor=[2, 2])
#             # kernel0 = torch.nn.functional.interpolate(k2, scale_factor=[4, 4])
#             e0, e1, e2, e3, e4, k0, k1, k2 = self.subnets[i](x, e0, e1, e2, e3, e4, k0, k1, k2)
#
#             e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)
#             d3 = self.outputs[i](e0)
#             # kernel = self.kernel_extras[i](k2)
#             # print(d3_img.max())
#             d3, m3 = torch.split(d3, [self.out_channels, self.num_kernel], dim=1)
#             d3_img = d3 + img
#             x_tmp_out_list.append(d3_img)
#             if gt is None:
#                 img_3 = d3_img # torch.nn.functional.interpolate(d3_img, scale_factor=[0.25, 0.25])
#                 # x_reblur = self.DegradationModel(img_3, kernel2)
#             else:
#                 img_3 = gt  # torch.nn.functional.interpolate(gt, scale_factor=[0.25, 0.25])
#             # m3 = torch.softmax(m3, dim=1)
#             m3 = torch.sigmoid(m3)
#             # kernel = self.post_process(kernel)
#             # kernel = kornia.geometry.subpix.spatial_softmax2d(kernel)
#             # print(kernel)
#             # blur_kernel, noise = torch.chunk(kernel, 2, dim=1)
#             # blur_kernel = kernel
#             blur_kernel = torch.softmax(kernel, dim=1)
#             x_reblur = reblurfilter(img_3, blur_kernel, m3)
#             # x_reblur = kornia.enhance.normalize_min_max(x_reblur, 0., 1.)
#             x_reblur = x_reblur / (self.num_kernel)
#             # x_reblur = torch.clamp(x_reblur, 0., 1.)
#             # x_deblur = deblurfilter(img, blur_kernel, m3)
#             # x_deblur = torch.clamp(x_deblur, 0., 1.)
#             # x_deblur = deblurDWeinerfilter(img, blur_kernel, m3, noise)
#             reblur_loss += self.loss_reblur(x_reblur[:, :, self.pad_1:-self.pad_2, self.pad_1:-self.pad_2].contiguous(),
#                                      img[:, :, self.pad_1:-self.pad_2, self.pad_1:-self.pad_2].contiguous())
#             # deblur_loss += self.loss_deblur(x_deblur, img_3)
#         for i in range(self.num_kernel_extra, self.num_subnet):
#             x = self.stems[i](img)
#             e0, e1, e2, e3, e4 = self.subnets[i](x, e0, e1, e2, e3, e4)
#             e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)
#             d3_img = self.outputs[i](e0) + img
#             # print(d3_img.max())
#             x_tmp_out_list.append(d3_img)
#         # if self.num_subdenet > self.num_subnet:
#         for i in range(self.num_subnet, self.num_subdenet):
#             e3, e2, e1, e0 = self.subdenets[i](e4, e3, e2, e1, e0)
#             d3_img = self.outputs[i](e0) + img
#             # print(d3_img.max())
#             x_tmp_out_list.append(d3_img)
#         # print('loss_reblur: ',  reblur_loss.item() / self.num_kernel_extra)
#         # print('loss_deblur: ', deblur_loss.item() / self.num_kernel_extra)
#         if self.num_kernel_extra > 0:
#             return {'loss_reblur': reblur_loss / self.num_kernel_extra,
#                     # 'loss_deblur': deblur_loss / self.num_kernel_extra,
#                     'img': x_tmp_out_list, 'kernel': kernel} # , 'x_deblur': x_deblur}
#         else:
#             return {'img': x_tmp_out_list}  # , 'x_deblur': x_deblur}
#
#
# class UFPNetLocal(Local_Base, UFPNet):
#     def __init__(self, *args, fast_imp=False, **kwargs):
#         Local_Base.__init__(self)
#         UFPNet.__init__(self, *args, **kwargs)
#
#         train_size = (1, 3, 256, 256)
#         N, C, H, W = train_size
#         base_size = (int(H * 1.5), int(W * 1.5))
#
#         self.eval()
#         with torch.no_grad():
#             self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


# class DeepRevLocal(Local_Base, DeepRev):
#     def __init__(self, *args, fast_imp=False, **kwargs):
#         Local_Base.__init__(self)
#         DeepRev.__init__(self, *args, **kwargs)
#
#         train_size = (1, 3, 256, 256)
#         N, C, H, W = train_size
#         base_size = (int(H * 1.5), int(W * 1.5))
#
#         self.eval()
#         with torch.no_grad():
#             self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


# if __name__ == '__main__':
#     pth = '/home/ubuntu/90t/personal_data/mxt/MXT/RevIR/experiments/Deblurring_DeepRev_1e1d_avgUFPNet_flowtrain_reblur_predx_freqloss_1e-3_ema/models/net_g_200000.pth'
#     pth = None
#     net = DeepRev(test_only=True, pretrain=False, save_memory=False, train_size=[128, 128], state_dict_pth=pth).cuda()
#
#     # for n, p in net.named_parameters()
#     #     print(n, p.shape)
#
#     inp = torch.randn((1, 3, 128, 128)).cuda()
#     print(inp.shape, inp.max())
#     out = net(inp)
#     print(len(out))
#     # print(torch.mean(torch.abs(out['img'][-1] - inp)))
#     # print(out['loss'])
#     print(torch.mean(torch.abs(out['img'][0] - inp)))
#     inp_shape = (3, 256, 256)

    # from ptflops import get_model_complexity_info
    #
    # macs, params = get_model_complexity_info(net, inp_shape, verbose=False, print_per_layer_stat=False)
    #
    # # params = float(params[:-3])
    # # macs = float(macs[:-4])
    #
    # print(macs, params)


