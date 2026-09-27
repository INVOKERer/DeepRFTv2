'''
An official Pytorch impl of `Transcending the Limit of Local Window: 
Advanced Super-Resolution Transformer with Adaptive Token Dictionary`.

Arxiv: 'https://arxiv.org/abs/2401.08209'
'''

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from basicsr.models.archs.arch_mambaIR_util import to_2tuple, trunc_normal_
from fairscale.nn import checkpoint_wrapper
from basicsr.models.archs.DeepRFTv2_util import *
# from basicsr.utils.registry import ARCH_REGISTRY

class FeatureExtract(nn.Module):
    def __init__(self, in_channel=32, out_channel=32, inp_feature='inp_latentsharp_feature', deconv_feature='inp_latentsharp_feature'):
        super().__init__()

        self.inp_feature = inp_feature
        self.deconv_feature = deconv_feature
        if inp_feature in ['inp', 'latentsharp']:
            in_c = 3
        elif inp_feature == 'inp_latentsharp':
            in_c = 6
        elif inp_feature == 'inp_feature':
            in_c = 3 + in_channel
        elif inp_feature == 'latentsharp_feature':
            in_c = 3 + in_channel
        elif inp_feature == 'inp_latentsharp_feature':
            in_c = 6 + in_channel
        elif inp_feature == 'feature':
            in_c = in_channel


        self.intro_inp = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_channel, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=False)
        )

        if deconv_feature in ['inp', 'latentsharp']:
            in_c = 3
        elif deconv_feature == 'inp_latentsharp':
            in_c = 6
        elif deconv_feature == 'inp_feature':
            in_c = 3 + in_channel
        elif deconv_feature == 'latentsharp_feature':
            in_c = 3 + in_channel
        elif deconv_feature == 'inp_latentsharp_feature':
            in_c = 6 + in_channel
        elif deconv_feature == 'feature':
            in_c = in_channel

        self.intro_deconv = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=in_channel, kernel_size=1, padding=0, stride=1, groups=1,
                      bias=False)
        )
    def forward(self, inp_img, latent_sharp, feature):
        if self.inp_feature == 'inp':
            inp = inp_img
        elif self.inp_feature == 'latentsharp':
            inp = latent_sharp
        elif self.inp_feature == 'feature':
            inp = feature
        elif self.inp_feature == 'inp_latentsharp':
            inp = torch.cat([inp_img, latent_sharp], dim=1)
        elif self.inp_feature == 'inp_feature':
            inp = torch.cat([inp_img, feature], dim=1)
        elif self.inp_feature == 'latentsharp_feature':
            inp = torch.cat([latent_sharp, feature], dim=1)
        elif self.inp_feature == 'inp_latentsharp_feature':
            inp = torch.cat([inp_img, latent_sharp, feature], dim=1)

        x = self.intro_inp(inp)
        # x = self.pad(x)
        if self.deconv_feature == 'inp':
            inp = inp_img
        elif self.deconv_feature == 'latentsharp':
            inp = latent_sharp
        elif self.deconv_feature == 'feature':
            inp = feature
        elif self.deconv_feature == 'inp_latentsharp':
            inp = torch.cat([inp_img, latent_sharp], dim=1)
        elif self.deconv_feature == 'inp_feature':
            inp = torch.cat([inp_img, feature], dim=1)
        elif self.deconv_feature == 'latentsharp_feature':
            inp = torch.cat([latent_sharp, feature], dim=1)
        elif self.deconv_feature == 'inp_latentsharp_feature':
            inp = torch.cat([inp_img, latent_sharp, feature], dim=1)

        y = self.intro_deconv(inp)

        return torch.cat([x, y], dim=1)  # *self.alpha

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
class GetPSF(nn.Module):
    def __init__(self, out_channel=32, in_channel=32, norm='backward', shift=True, spatial=False):
        super().__init__()
        self.norm = norm
        self.shift = shift
        self.spatial = spatial
        self.out_channel = out_channel

        self.in_channel = in_channel
        out_channel = in_channel
        if spatial:
            in_channel = in_channel // 2

        self.ending = nn.Sequential(nn.Conv2d(in_channels=in_channel*2, out_channels=out_channel, kernel_size=1, padding=0,
                                              stride=1,
                                              groups=1,
                                              bias=False)
                                    )
        # inpx, batch_list = window_partitionx(inp[0], self.patch_size)
        # self.alpha = nn.Parameter(torch.ones([1, out_channel*2, 1, 1]), requires_grad=True)
        # self.beta = nn.Parameter(torch.ones([1, num_ker * 2, 1, 1]), requires_grad=True)

    def forward(self, x):
        x, y = x.split([self.in_channel, self.out_channel], dim=1)
        # inp_freq = fft_ri_(x, clamp_min=math.exp(-10), norm=self.norm)
        if self.spatial:
            inp_enc_level1_ker = x
        else:
            inp_freq = torch.fft.rfft2(x, norm=self.norm)
            inp_enc_level1_ker = torch.cat([inp_freq.real, inp_freq.imag], dim=1)
        deconv_freq = torch.fft.rfft2(y, norm=self.norm)


        # x = torch.cat([inp_mag_log, inp_phase], dim=1) * self.alpha
        return self.ending(inp_enc_level1_ker), deconv_freq # *self.alpha

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
class EstOTF(nn.Module):
    def __init__(self, width=32, num_ker=4, kernel_size=19, norm='backward', shift=True, mode='rfft', norm_type='None', spatial=False):
        super().__init__()
        self.norm = norm
        self.shift = shift
        self.kernel_size = [kernel_size, kernel_size]
        self.spatial = spatial
        if norm_type == 'LayerNorm':
            self.norm = LayerNorm2d(width)
        elif norm_type == 'BatchNorm':
            self.norm = nn.BatchNorm2d(width)
        elif norm_type == 'InstanceNorm':
            self.norm = nn.InstanceNorm2d(width)
        else:
            self.norm = nn.Identity()

        if spatial:
            num_ker = num_ker // 2
        self.ending = nn.Sequential(nn.Conv2d(in_channels=width, out_channels=num_ker * 2, kernel_size=1, padding=0,
                                              stride=1,
                                              groups=1,
                                              bias=True)
                                    )

    def forward(self, inp):
        inp = self.norm(inp)
        x = self.ending(inp)
        if self.spatial:
            otf_deblur = torch.fft.rfft2(x)
        else:
            ker_real, ker_imag = torch.chunk(x, 2, dim=1)

            # ker_mag, ker_phase = ker_mag, ker_phase * torch.pi
            otf_deblur = torch.complex(ker_real, ker_imag)

        return otf_deblur
class KernelEstimator(nn.Module):
    def __init__(self, img_channel=3, num_ker=3, width=[32, 32, 32], width_d=[48, 48, 48], baseblock_enc_ker=['naf', 'naf', 'naf', 'naf'],
                 baseblock_dec_ker=['naf', 'naf', 'naf'], enc_ker_layers=[1, 1, 1, 1], num_heads_dec=[4, 2, 1], pad_size=0, downsample='noconcat_nodownsmaple',
                 dec_ker_layers=[3, 2, 1], num_heads_enc=[1, 1, 1, 1], train_size=[256, 256], patch_size=[256, 256], overlap_size=[16,16], ks=[32, 16, 8],
                 save_memory=False, drop_path=0., kernel_size=[65, 33, 17], k_len=61, upx=False, scale=0, first_col=True, spatial_add=False, get_ker=False,
                 frozen_kernel=False, sub_image_pad=False, select_range=None, flip_transpose=False, inp_feature='feature', deconv_feature='feature', norm='backward', mode='rfft', norm_type='LayerNorm', mask=False, spatial_decoder=False):
        super().__init__()
        # self.grid = grid
        self.train_size = train_size
        self.spatial_decoder = spatial_decoder
        self.overlap_size = overlap_size
        self.pad_size = pad_size
        self.flip_transpose=flip_transpose
        self.select_range=select_range
        self.in_channel = width_d[0]
        self.out_channel = width[0]
        # self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
        # self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
        # self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
        # self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
        # print(self.overlap_size)
        self.mask = mask
        self.sub_image_pad = sub_image_pad
        self.k_len = k_len
        self.upx = upx
        self.num_ker = num_ker
        self.kernel_size = kernel_size
        self.frozen_kernel = frozen_kernel
        self.ks = ks
        self.scale = scale
        self.first_col = first_col
        self.up_scale = 1
        self.patch_size = patch_size
        # self.patch_size = [256, 256]
        self.enc_ker_layers = enc_ker_layers
        self.spatial_add = spatial_add
        # print(spatial_add, spatial_decoder)
        self.get_ker = get_ker
        self.h = -1
        self.w = -1
        self.inp_feature = inp_feature
        # self.inp_feature = inp_feature
        # self.kernel_size_up = patch_size
        self.spatial_concat = True # False # True
        # norm = 'backward'
        # self.output_feature_level1 = nn.Sequential(
        #     nn.Conv2d(in_channels=width_d[0], out_channels=width_d[0], kernel_size=1, padding=0, stride=1, groups=1, bias=True),
        # )
        if self.mask:
            self.mask_feature1 = nn.Sequential(
                # nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=width_d[0]*2, out_channels=width_d[0]*2, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
                nn.GELU(),
                nn.Conv2d(in_channels=width_d[0]*2, out_channels=width_d[0]*2, kernel_size=1, padding=0, stride=1, groups=1,
                          bias=True),
            )
        # self.intro = FeatureExt_ri_rfft(6, width[0], norm=norm, shift=False)
        self.feature_extract = FeatureExtract(width_d[0], width[0], inp_feature=inp_feature, deconv_feature=deconv_feature)
        spatial = False
        self.intro = GetPSF(width_d[0], width[0], norm=norm, shift=False, spatial=spatial)
        # self.intro1 = FeatureExt_ri_rfft(width_d[0], width[0], norm=norm, shift=False)
        self.end_ker1 = EstOTF(width[0], width_d[0], kernel_size[0], norm=norm, shift=False, mode=mode, norm_type=norm_type, spatial=spatial)
        # if self.spatial_add:
        #     self.out_channels = width_d[0] * 2
        #     # self.intro_spatial = nn.Sequential(
        #     #     nn.Conv2d(in_channels=width_d[0]*2, out_channels=width[0], kernel_size=1, padding=0, stride=1, groups=1,
        #     #               bias=False),
        #     # )
        #     # self.end_spatial = nn.Sequential(
        #     #     nn.Conv2d(in_channels=width[0], out_channels=width_d[0]*2, kernel_size=1, padding=0, stride=1, groups=1, bias=False),
        #     # )
        #     self.deblur_feature1 = nn.Sequential(
        #         nn.Conv2d(in_channels=width_d[0] * 3, out_channels=width_d[0], kernel_size=1, padding=0, stride=1,
        #                   groups=1, bias=False),
        #     )
        # else:
        self.out_channels = width_d[0]
        if self.spatial_concat:
            self.deblur_feature1 = nn.Sequential(
                nn.Conv2d(in_channels=width_d[0]*2, out_channels=width_d[0], kernel_size=1, padding=0, stride=1, groups=1, bias=False),
            )

        # self.end_feature1 = nn.Sequential(
        #     nn.Conv2d(in_channels=width_d[0], out_channels=3, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
        # )

        # self.intro_img1 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=width_d[0], kernel_size=3, padding=1, stride=1, groups=1,
        #               bias=True),
        #     # nn.Conv2d(in_channels=width_d, out_channels=width_d, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
        #     # nn.Conv2d(in_channels=width_d, out_channels=width_d, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
        #     # nn.ReLU()
        # )
        # self.intro_inp1 = nn.Sequential(
        #     nn.Conv2d(in_channels=width_d[0], out_channels=width_d[0], kernel_size=3, padding=1, stride=1, groups=1,
        #               bias=True),
        #     nn.GELU()
        # )

        # self.conv_intro1 = nn.Conv2d(in_channels=width[0]*2, out_channels=width[0], kernel_size=1, padding=0, stride=1, groups=1, bias=False)
            # if not self.return_feat:

        # self.fft_head = FFT_head()
        # self.subnets_ker = nn.ModuleList()
        # self.subdenets_ker = nn.ModuleList()
        # self.middle_blks = nn.ModuleList()
        # self.ups = nn.ModuleList()
        # self.downs = nn.ModuleList()
        channels_enc_ker = [width[0], width[0], width[0], width[0]]
        channels_dec_ker = [width[0], width[0], width[0], width[0]]
        dp_rate_e = [x.item() for x in torch.linspace(0, drop_path, len(enc_ker_layers))]
        dp_rate_d = [x.item() for x in torch.linspace(0, drop_path, len(dec_ker_layers))]
        dp_rate_d.reverse()

        # first_col = True
        if self.spatial_decoder:
            # save_memory = True
            self.subdenets_ker = UniSimpleResEncoder4Level(
                channels=channels_dec_ker, layers=dec_ker_layers, num_heads=num_heads_dec,
                kernel_size=kernel_size,
                first_col=first_col, dp_rates=dp_rate_d, save_memory=save_memory,
                baseblock=baseblock_dec_ker, train_size=[train_size[0], train_size[1]],
                patch_size=[patch_size[0], patch_size[1]], sub_idx=1)
        if enc_ker_layers[-1] != -1:
            self.subnets_ker=UniSimpleResEncoder4Level(
                channels=channels_enc_ker, layers=enc_ker_layers, num_heads=num_heads_enc,
                kernel_size=kernel_size,
                first_col=first_col, dp_rates=dp_rate_e, save_memory=save_memory,
                baseblock=baseblock_enc_ker, train_size=[train_size[0], train_size[1]],
                patch_size=[patch_size[0], patch_size[1]], sub_idx=1)
        else:

            modules = get_modules(baseblock_enc_ker[0], 0, channels_enc_ker, enc_ker_layers, num_heads_enc, train_size)
            self.subnets_ker = nn.Sequential(*modules)

        # self.subnets_ker = get_modules()
        # self.subdenets_ker=UniSimpleDecoder3Level(
        #     channels=channels_dec_ker, layers=dec_ker_layers, num_heads=num_heads_dec,
        #     kernel_size=kernel_size,
        #     first_col=first_col, dp_rates=dp_rate_d, save_memory=save_memory,
        #     baseblock=baseblock_dec_ker, train_size=[train_size[0], train_size[1] // 4],
        #     patch_size=[patch_size[0], patch_size[1]], sub_idx=1)


        # self.padder_size = 2 ** len(enc_ker_layers)
        # self.alpha = nn.Parameter(torch.ones([1, width // 2, 1, 1]) * 0.1, requires_grad=True)
        # self.beta = nn.Parameter(torch.ones([1, num_ker, 1, 1]) * 2., requires_grad=True)
        # self.alpha = nn.Parameter(torch.cat([torch.ones([1, width, 1, 1]) / 10.,torch.ones([1, width, 1, 1])], dim=1), requires_grad=True)

        # self.ker_loss = MSELoss(100.) # MSELoss(1./self.k_len, reduction='sum')
        # self.ker_loss = MSELoss(0.01 / self.k_len, reduction='sum')
        # self.deblur_loss_ = FreqLoss(0.5)
        # self.deblur_loss = FreqLoss(1.)
        # self.reblur_loss = FreqLoss(1.)
        # self.deform_conv = DCNv3_kernel_pytorch(3, group=3, kernel_size=9, pad=4, motion_blur_kernel_size=kernel_size)
        self.padder_size=2
    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign
    def grids(self, x, level=0):
        b, c, h, w = x.shape
        n_level = 2 ** level

        # assert b == 1
        k1, k2 = self.patch_size[0] // n_level, self.patch_size[1] // n_level
        # if h < 1.25 * k1:
        #     k1 = h
        # if w < 1.25 * k2:
        #     k2 = w
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = [self.overlap_size[0] // n_level, self.overlap_size[1] // n_level]  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])

        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1

        # print(h, stride[0])
        # print(num_row, num_col)
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

    def get_overlap_matrix(self, h, w):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        self.ek1 = self.nr * self.stride[0] + self.overlap_size[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.overlap_size[1] * 2 - w
        if self.h != h or self.w != w:
            self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
            self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
            self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
            self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
        if self.h != h:
            self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1)
            self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1)
        if self.w != w:
            self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2)
            self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2)
        self.h = h
        self.w = w


        # self.ek1, self.ek2 = 48, 224
        # print(self.ek1, self.ek2, self.nr)
        # print(self.overlap_size)
        # self.overlap_size = [8, 8]
        # self.overlap_size = [self.overlap_size[0] * 2, self.overlap_size[1] * 2]


    def grids_inverse(self, outs, type_out, out_device=None, pix_max=1.):
        if out_device is None:
            out_device = outs.device
        # print(out_device)

        # type_out = torch.uint8 if pix_max == 255. else torch.float32
        # type_out = torch.int16
        preds = torch.zeros(self.original_size, device=out_device, dtype=type_out)  # .to(out_device)
        b, c, h, w = self.original_size
        # print(self.original_size)
        # outs = torch.clamp(outs, 0, 1) # * pix_max
        # outs = outs.type(type_out)
        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.patch_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)
        # rounding_mode = 'floor'
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            if i != 0 and i + k1 != h:
                outs[cnt, :, :self.overlap_size[0], :] = torch.mul(outs[cnt, :, :self.overlap_size[0], :],
                                                                      self.fuse_matrix_h2.to(outs.device))
                # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
            if i + k1 * 2 - self.ek1 < h:
                outs[cnt, :, -self.overlap_size[0]:, :] = torch.mul(outs[cnt, :, -self.overlap_size[0]:, :],
                                                                       self.fuse_matrix_h1.to(outs.device))
            # print(self.fuse_matrix_eh1.dtype)
            if i + k1 == h and i != 0:
                outs[cnt, :, :self.ek1, :] = torch.mul(outs[cnt, :, :self.ek1, :], self.fuse_matrix_eh2.to(outs.device))
            if i + k1 * 2 - self.ek1 == h:
                outs[cnt, :, -self.ek1:, :] = torch.mul(outs[cnt, :, -self.ek1:, :],
                                                        self.fuse_matrix_eh1.to(outs.device))

            if j != 0 and j + k2 != w:
                outs[cnt, :, :, :self.overlap_size[1]] = torch.mul(outs[cnt, :, :, :self.overlap_size[1]],
                                                                      self.fuse_matrix_w2.to(outs.device))
            if j + k2 * 2 - self.ek2 < w:
                # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, -self.overlap_size[1]:] = torch.mul(outs[cnt, :, :, -self.overlap_size[1]:],
                                                                       self.fuse_matrix_w1.to(outs.device))
            if j + k2 == w and j != 0:
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

    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign

    def forward(self, inp, blur_img=None, deblur_img=None, global_img=None, ker_feature=[0, 0, 0, 0], spatial_feature=[0, 0, 0, 0], gt=None, batch_size=None):

        inpx = inp
        _, _, H, W = inp.shape
        # inpx = self.check_image_size(inp)
        # print(inp[0].shape)
        # c_in = torch.cat([deblur_img, blur_img], dim=1)
        # if self.select_range is not None:
        #     h_start = self.select_range[0]
        #     h_end = self.select_range[1]
        #     w_start = self.select_range[2]
        #     w_end = self.select_range[3]
        #     center_range = [self.select_range[1]-self.select_range[0], self.select_range[3]-self.select_range[2]]
        #     blur_img = kornia.geometry.center_crop(blur_img, center_range)
        #     # print(blur_img_.shape)
        #     deblur_img_ = kornia.geometry.center_crop(deblur_img, center_range)
        #     inpx = kornia.geometry.center_crop(inpx, center_range)
        #     fx = self.feature_extract(blur_img, deblur_img_, inpx)
        # else:

        fx = self.feature_extract(blur_img, deblur_img, inpx)
        if self.flip_transpose:
            x0 = fx
            x1 = torch.rot90(fx, k=2, dims=[-2, -1])
            x2 = torch.flipud(fx)
            x3 = torch.flipud(x1)

            fx = torch.cat([x0, x1, x2, x3], dim=0)
        # fx = self.feature_extract(blur_img, deblur_img, inp[1])
        # print(fx.shape)

            # feature_deblur1 = feature_deblur1 * mask1 + x_ * mask2
        if H > self.patch_size[0] or W > self.patch_size[1]:
            # fx, batch_list = window_partitionx(fx, self.patch_size)
            if self.training or (self.overlap_size[0] == 0 and self.overlap_size[1] == 0):
                fx, batch_list = window_partitionx(fx, self.patch_size)
                # print(fx.shape)
            else:
                fx = self.grids(fx)
        h, w = fx.shape[-2:]
        fx = self.check_image_size(fx)
        # if self.mask:
        #     # b, c, h, w = x_.shape
        #     # print(x.shape)
        #     x_, _ = fx.split([self.in_channel, self.out_channel], dim=1)
        #     b, c, h, w = x_.shape
        #     # print(self.mask_feature1(x_).shape, x_.shape)
        #     mask = F.gumbel_softmax(self.mask_feature1(x_), tau=1, hard=True, dim=1)
        #     mask1, mask2 = mask.chunk(2, dim=1)
        #     print(mask1.shape)
            # c_in = self.grids(c_in)
        # if self.mask:
        #     mask = F.gumbel_softmax(self.mask_feature1(fx),tau=1,hard=True,dim=1)
        #     fx = fx * mask
        if self.pad_size > 0:
            fx = F.pad(fx, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), mode='replicate')
        # print(H, W, fx.shape, self.overlap_size)
        # out_dict = self.forward_(fx, ker_feature)
        # feature_deblur1 = out_dict['feature_deblur']
        if batch_size is None:
            out_dict = self.forward_(fx, ker_feature)
            feature_deblur1 = out_dict['feature_deblur']
        else:
            all_batch = inpx.shape[0]
            batchs = range(0, all_batch, batch_size)

            feature_deblur1s = []
            ker_feature_x = [ [], [], [], [] ]
            # print(ker_feature_x, len(ker_feature_x))
            # print(type(ker_feature[0]))
            for batch_z in batchs:
                ker_feature_ = []
                if batch_z + batch_size <= all_batch:
                    fx_ = fx[batch_z:batch_z + batch_size, ...]
                    # inpx_ = inpx[batch_z:batch_z + batch_size, ...]
                    # print(len(ker_feature))
                    if type(ker_feature[0]) != int:
                        for k in range(len(ker_feature)):
                            x = ker_feature[k][batch_z:batch_z + batch_size, ...]
                            ker_feature_.append(x)
                    else:
                        ker_feature_ = [0, 0, 0, 0]
                else:
                    fx_ = fx[batch_z:, ...]

                    if len(fx_.shape) == 3:
                        fx_ = fx_.unsqueeze(0)

                    # inpx_ = inpx[batch_z:, ...]
                    #
                    # if len(inpx_.shape) == 3:
                    #     inpx_ = inpx_.unsqueeze(0)
                    if type(ker_feature[0]) != int:
                        for k in range(len(ker_feature)):
                            x = ker_feature[k][batch_z:, ...]
                            if len(x.shape) == 3:
                                x = x.unsqueeze(0)
                            ker_feature_.append(x)
                    else:
                        ker_feature_ = [0, 0, 0, 0]
                # print(len(ker_feature_), len(ker_feature), type(ker_feature[0]))
                out_dict = self.forward_(fx_, ker_feature_)
                feature_deblur1s.append(out_dict['feature_deblur'])
                ker_feature_s = out_dict['ker_feature']

                for k in range(len(ker_feature_x)):
                    ker_feature_x[k].append(ker_feature_s[k])

            feature_deblur1 = torch.cat(feature_deblur1s, dim=0)
            ker_feature = []
            for k in range(len(ker_feature_x)):
                ker_feature.append(torch.cat(ker_feature_x[k], dim=0))
            # print(len(ker_feature))
            out_dict['ker_feature'] = ker_feature
        if self.pad_size > 0:
            feature_deblur1 = feature_deblur1[:, :, self.pad_size:-self.pad_size, self.pad_size:-self.pad_size]

        feature_deblur1 = feature_deblur1[:, :, :h, :w]
        if H > self.patch_size[0] or W > self.patch_size[1]:
            # feature_deblur1 = window_reversex(feature_deblur1, self.patch_size, H, W, batch_list)
            if self.training or (self.overlap_size[0] == 0 and self.overlap_size[1] == 0):
                # print(feature_deblur1.shape)
                feature_deblur1 = window_reversex(feature_deblur1, self.patch_size, H, W, batch_list)
            else:
                feature_deblur1 = self.grids_inverse(feature_deblur1, type_out=inp.dtype)
        # feature_deblur1 = feature_deblur1[:, :, :H, :W]
        # if self.mask:
        #     feature_deblur1 = feature_deblur1 * mask1 + x_ * mask2
            # feature_deblur1 = feature_deblur1 * self.mask_feature1(inp[0])
        # if self.select_range is not None:
        #     feature_deblur1_ = inp[0]
        #     # print(h_start, h_end, w_start, w_end, inp[0].shape, feature_deblur1_.shape, feature_deblur1_[..., h_start: h_end, w_start: w_end].shape, feature_deblur1.shape)
        #     feature_deblur1_[..., h_start: h_end, w_start: w_end] = feature_deblur1
        #     feature_deblur1 = feature_deblur1_.contiguous()
        if self.flip_transpose:
            x0, x1, x2, x3 = feature_deblur1.chunk(4, dim=0)
            x1 = torch.rot90(x1, k=2, dims=[-2, -1])
            x2 = torch.flipud(x2)
            x3 = torch.flipud(x3)
            x3 = torch.rot90(x3, k=2, dims=[-2, -1])
            feature_deblur1 = (x0 + x1 + x2 + x3) / 4.0
        if self.spatial_concat:

            feature_deblur1 = self.deblur_feature1(torch.cat([feature_deblur1, inp], dim=1))
        if self.spatial_add or self.spatial_decoder:
            out_enc_level3, out_enc_level2, out_enc_level1, out_dec_level1 = spatial_feature
            inp_enc_level1 = feature_deblur1 # self.intro_spatial(feature_deblur1)
            # if self.flip_transpose and type(out_dec_level1) != int:
            #     out_enc_level1 = torch.flip(out_enc_level1, dims=(-2, -1)).contiguous()
            #     out_enc_level2 = torch.transpose(torch.flip(out_enc_level2, dims=(-2, -1)), dim0=-2, dim1=-1).contiguous()
            #     out_enc_level3 = torch.transpose(out_enc_level3, dim0=-2, dim1=-1).contiguous()
            out_enc_level1, out_enc_level2, out_enc_level3, out_dec_level1 = self.subdenets_ker(
                inp_enc_level1,
                out_enc_level1,
                out_enc_level2,
                out_enc_level3, out_dec_level1)

            # out_dec_level1_spatial = out_dec_level1
            out_dict['spatial_feature'] = out_enc_level1, out_enc_level2, out_enc_level3, out_dec_level1
            feature_deblur1 = out_dec_level1 # torch.cat([feature_deblur1, out_dec_level1], dim=1)
        else:
            out_dict['spatial_feature'] = [0, 0, 0, 0]
        out_dict['feature_deblur'] = feature_deblur1
        # blur_pattern = self.end_feature1(feature_deblur1)
        # # if self.flip_transpose:
        # #     x0, x1, x2, x3 = blur_pattern.chunk(4, dim=0)
        # #     x1 = torch.rot90(x1, k=2, dims=[-2, -1])
        # #     x2 = torch.flipud(x2)
        # #     x3 = torch.flipud(x3)
        # #     x3 = torch.rot90(x3, k=2, dims=[-2, -1])
        # #     blur_pattern = (x0 + x1 + x2 + x3) / 4.0
        # results_deblur_max = blur_pattern + blur_img
        # # if self.select_range is not None:
        # #     results_deblur_max_ = deblur_img
        # #     h_start = self.select_range[0]
        # #     h_end = self.select_range[1]
        # #     w_start = self.select_range[2]
        # #     w_end = self.select_range[3]
        # #     results_deblur_max_[..., h_start: h_end, w_start: w_end] = results_deblur_max[..., h_start: h_end, w_start: w_end]
        # #     results_deblur_max = results_deblur_max_
        # # results_deblur_max = torch.clamp(results_deblur_max, 0., 1.)
        # out_dict['img_deblur'] = results_deblur_max

        return out_dict
    def forward_(self, x, ker_feature=[0, 0, 0, 0]):
    # def forward(self, inp, blur_img, deblur_img, global_img=None, ker_feature=None, gt=None):
        out_dict = {}
        # print(x.shape)
        # if self.first_col:
        #     out_enc_level3_ker, out_enc_level2_ker, out_enc_level1_ker, out_dec_level1_ker = 0, 0, 0, 0
        # else:
        out_enc_level3_ker, out_enc_level2_ker, out_enc_level1_ker, out_dec_level1_ker = ker_feature

        # print(x.shape)
        # inp_enc_level1_ker, _ = self.intro(x)
        inp_enc_level1_ker, inp_enc_level1_freq = self.intro(x)
        # inp_enc_level1_ker = self.conv_intro1(torch.cat([inp_enc_level1_ker, inp_enc_level1_ker1], dim=1))

        if self.enc_ker_layers[-1] != -1:
            # print(111)
            out_enc_level1_ker, out_enc_level2_ker, out_enc_level3_ker, out_dec_level1_ker = self.subnets_ker(inp_enc_level1_ker,
                                                                                          out_enc_level1_ker,
                                                                                          out_enc_level2_ker,
                                                                                          out_enc_level3_ker, out_dec_level1_ker)
        else:
            out_dec_level1_ker = self.subnets_ker(inp_enc_level1_ker)
        # print(out_dec_level1_ker.shape, inp_enc_level1_freq.shape, out_enc_level2_ker.shape, out_enc_level3_ker.shape)
        # if self.spatial_add:
        #     out_dec_level1_spatial = out_dec_level1_ker

        out_dict['ker_feature'] = out_enc_level3_ker, out_enc_level2_ker, out_enc_level1_ker, out_dec_level1_ker

        out_dict['visual_feature'] = [inp_enc_level1_ker, out_enc_level1_ker, out_enc_level2_ker, out_enc_level3_ker, out_dec_level1_ker]

        otf_deblur_max = self.end_ker1(out_dec_level1_ker)



            # inp_enc_level1_spatial = torch.fft.irfft2(inp_enc_level1_spatial)
            #
            # out_dict['feature_spatial_deblur'] = inp_enc_level1_spatial
        feature_deblur1 = torch.fft.irfft2(inp_enc_level1_freq * otf_deblur_max)
        # feature_deblur1 = torch.fft.irfft2(inp_enc_level1_freq * torch.conj(otf_deblur_max))

        if self.mask:
            # b, c, h, w = x_.shape
            # print(x.shape)
            x_, _ = x.split([self.in_channel, self.out_channel], dim=1)
            b, c, h, w = x_.shape
            # print(x_.shape, x_.shape)
            # mask = F.gumbel_softmax(self.mask_feature1(x_).view(b, 2, c, 1, 1), tau=1, hard=True, dim=1)

            mask = torch.softmax(self.mask_feature1(torch.cat([feature_deblur1, x_], dim=1)).view(b, 2, c, h, w), dim=1)
            mask1, mask2 = mask[:, 0, ...], mask[:, 1, ...]
            # print(mask1.shape)
            feature_deblur1 = feature_deblur1 * mask1 + x_ * mask2

        if self.get_ker:
            dict_visual = {}
            dict_visual['kernel_fft'] = torch.log(torch.fft.fftshift(torch.abs(torch.fft.fft2(torch.fft.irfft2(otf_deblur_max))) + 1e-7, dim=[-2, -1]))
            kernel = convert_otf2psf(otf_deblur_max, [65, 65])

            dict_visual['kernel'] = kernel

            dict_visual['feature'] = torch.fft.irfft2(inp_enc_level1_freq)
            dict_visual['deconv_feature'] = feature_deblur1
            out_dict['feature_visual'] = dict_visual

            # out_dec_level1_spatial = out_dec_level1_ker
            # # out_dec_level1_spatial = out_dec_level1_ker
            # # featrue_spatial = self.end_spatial(out_dec_level1_spatial)
            # # featrue_spatial_real, featrue_spatial_imag = featrue_spatial.chunk(2, dim=1)
            # # deblur_spatial = torch.complex(featrue_spatial_real, featrue_spatial_imag)
            # feature_deblur2 = out_dec_level1_spatial #  torch.fft.irfft2(inp_enc_level1_freq + deblur_spatial)


        out_dict['feature_deblur'] = feature_deblur1

        return out_dict

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def check_image_size_(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.patch_size[0] - h % self.patch_size[0]) % self.patch_size[0]
        mod_pad_w = (self.patch_size[1] - w % self.patch_size[1]) % self.patch_size[1]
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
# Shuffle operation for Categorization and UnCategorization operations.
def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def feature_shuffle(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim-1, index=index)
    return shuffled_x


class dwconv(nn.Module):
    def __init__(self, hidden_features, kernel_size=5):
        super(dwconv, self).__init__()
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=kernel_size, stride=1, padding=(kernel_size - 1) // 2, dilation=1,
                      groups=hidden_features), nn.GELU())
        self.hidden_features = hidden_features

    def forward(self,x,x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.hidden_features, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        x = self.depthwise_conv(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


class ConvFFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, kernel_size=5, act_layer=nn.GELU):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.dwconv = dwconv(hidden_features=hidden_features, kernel_size=kernel_size)
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x, x_size):
        x = self.fc1(x)
        x = self.act(x)
        x = x + self.dwconv(x, x_size)
        x = self.fc2(x)
        return x

def window_partition_(x, window_size):
    # if isinstance(window_size, int):
    #     window_size = [window_size, window_size]
    _, H, W, _ = x.shape
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    x_main = window_partition(x[:, :h, :w, :], window_size)
    b_main = x_main.shape[0]
    if h == H and w == W:
        return x_main, [b_main]
    if h != H and w != W:
        x_r = window_partition(x[:, :h, -window_size:, :], window_size)
        b_r = x_r.shape[0] + b_main
        x_d = window_partition(x[:, -window_size:, :w, :], window_size)
        b_d = x_d.shape[0] + b_r
        x_dd = x[:, -window_size:, -window_size:, :]
        b_dd = x_dd.shape[0] + b_d
        # batch_list = [b_main, b_r, b_d, b_dd]
        return torch.cat([x_main, x_r, x_d, x_dd], dim=0), [b_main, b_r, b_d, b_dd]
    if h == H and w != W:
        x_r = window_partition(x[:, :h, -window_size:, :], window_size)
        b_r = x_r.shape[0] + b_main
        return torch.cat([x_main, x_r], dim=0), [b_main, b_r]
    if h != H and w == W:
        x_d = window_partition(x[:, -window_size:, :w, :], window_size)
        b_d = x_d.shape[0] + b_main
        return torch.cat([x_main, x_d], dim=0), [b_main, b_d]
def window_reverse_(windows, window_size, H, W, batch_list):
    # if isinstance(window_size, int):
    #     window_size = [window_size, window_size]
    h, w = window_size * (H // window_size), window_size * (W // window_size)
    # print(windows[:batch_list[0], ...].shape)
    # print(windows.shape, window_size, h, w)
    x_main = window_reverse(windows[:batch_list[0], ...], window_size, h, w)
    B, _, _, C = x_main.shape
    # print(x_main.shape)
    # print('windows: ', windows.shape)
    # print('batch_list: ', batch_list)
    if torch.is_complex(windows):
        res = torch.complex(torch.zeros([B, H, W, C]), torch.zeros([B, H, W, C]))
        res = res.to(windows.device)
    else:
        res = torch.zeros([B, H, W, C], dtype=windows.dtype, device=windows.device)

    res[:, :h, :w, :] = x_main
    if h == H and w == W:
        return res
    if h != H and w != W and len(batch_list) == 4:
        x_dd = window_reverse(windows[batch_list[2]:, ...], window_size, window_size, window_size)
        res[:, h:, w:, :] = x_dd[:, h - H:, w - W:, :]
        x_r = window_reverse(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :h, w:, :] = x_r[:, :, w - W:, :]
        x_d = window_reverse(windows[batch_list[1]:batch_list[2], ...], window_size, window_size, w)
        res[:, h:, :w, :] = x_d[:, h - H:, :, :]
        return res
    if w != W and len(batch_list) == 2:
        x_r = window_reverse(windows[batch_list[0]:batch_list[1], ...], window_size, h, window_size)
        res[:, :h, w:, :] = x_r[:, :, w - W:, :]
    if h != H and len(batch_list) == 2:
        x_d = window_reverse(windows[batch_list[0]:batch_list[1], ...], window_size, window_size, w)
        res[:, h:, :w, :] = x_d[:, h - H:, :, :]
    return res
def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size

    Returns:
        windows: (num_windows*b, window_size, window_size, c)
    """
    b, h, w, c = x.shape
    x = x.view(b, h // window_size, window_size, w // window_size, window_size, c)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, c)
    return windows

def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*b, window_size, window_size, c)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image

    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    x = windows.view(b, h // window_size, w // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(b, h, w, -1)
    return x


class WindowAttention(nn.Module):
    r"""
    Shifted Window-based Multi-head Self-Attention

    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """
    def __init__(self, dim, window_size, num_heads, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        self.qkv_bias = qkv_bias
        head_dim = dim // num_heads
        self.scale = head_dim**-0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        self.proj = nn.Linear(dim, dim)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, rpi, mask=None):
        r"""
        Args:
            qkv: Input query, key, and value tokens with shape of (num_windows*b, n, c*3)
            rpi: Relative position index
            mask (0/-inf):  Mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        b_, n, c3 = qkv.shape
        c = c3 // 3
        qkv = qkv.reshape(b_, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[rpi.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nw = mask.shape[0]
            attn = attn.view(b_ // nw, nw, self.num_heads, n, n) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, n, n)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        x = (attn @ v).transpose(1, 2).reshape(b_, n, c)
        x = self.proj(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}, qkv_bias={self.qkv_bias}'

    def flops(self, n):
        flops = 0
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * n * (self.dim // self.num_heads) * n
        #  x = (attn @ v)
        flops += self.num_heads * n * n * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += n * self.dim * self.dim
        return flops


class ATD_CA(nn.Module):
    r""" 
    Adaptive Token Dictionary Cross-Attention.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_tokens (int): Number of tokens in external token dictionary. Default: 64
        reducted_dim (int, optional): Reducted dimension number for query and key matrix. Default: 4
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_tokens=64, reducted_dim=10, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.rc = reducted_dim
        self.qkv_bias = qkv_bias

        self.wq = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = nn.Parameter(torch.ones([self.num_tokens]) * 0.5, requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, td, x_size):
        r"""
        Args:
            x: input features with shape of (b, n, c)
            td: token dicitionary with shape of (b, m, c)
            x_size: size of the input x (h, w)
        """
        h, w = x_size
        b, n, c = x.shape
        b, m, c = td.shape
        rc = self.rc
        
        # Q: b, n, c
        q = self.wq(x)
        # K: b, m, c
        k = self.wk(td)
        # V: b, m, c
        v = self.wv(td)

        # Q @ K^T
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))  # b, n, n_tk
        scale = torch.clamp(self.scale, 0, 1)
        attn = attn * (1 + scale * np.log(self.num_tokens))
        attn = self.softmax(attn)
        
        # Attn * V
        x = (attn @ v).reshape(b, n, c)

        return x, attn

    def flops(self, n):
        n_tk = self.num_tokens
        flops = 0
        # qkv = self.wq(x)
        flops += n * self.dim * self.rc
        # k = self.wk(gc)
        flops += n_tk * self.dim * self.rc
        # v = self.wv(gc)
        flops += n_tk * self.dim * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += n * self.dim * self.rc
        #  x = (attn @ v)
        flops += n * n_tk * self.dim

        return flops
    

class AC_MSA(nn.Module):
    r""" 
    Adaptive Category-based Multihead Self-Attention.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_tokens (int): Number of tokens in external dictionary. Default: 64
        num_heads (int): Number of attention heads. Default: 4
        category_size (int): Number of tokens in each group for global sparse attention. Default: 128
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, input_resolution, num_tokens=64, num_heads=4, category_size=128, qkv_bias=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.num_heads = num_heads
        self.category_size = category_size

        # self.wqkv = nn.Linear(dim, 3 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((1, 1))), requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, qkv, sim, x_size):
        """
        Args:
            x: input features with shape of (b, HW, c)
            mask: similarity map with shape of (b, HW, m)
            x_size: size of the input x
        """

        H, W = x_size
        b, n, c3 = qkv.shape
        c = c3 // 3
        b, n, m = sim.shape
        gs = min(n, self.category_size)  # group size
        ng = (n + gs - 1) // gs
        
        # classify features into groups based on similarity map (sim)
        tk_id = torch.argmax(sim, dim=-1, keepdim=False)
        # sort features by type
        x_sort_values, x_sort_indices = torch.sort(tk_id, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)
        shuffled_qkv = feature_shuffle(qkv, x_sort_indices)  # b, n, c3
        pad_n = ng * gs - n
        paded_qkv = torch.cat((shuffled_qkv, torch.flip(shuffled_qkv[:, n-pad_n:n, :], dims=[1])), dim=1)
        y = paded_qkv.reshape(b, -1, gs, c3)

        qkv = y.reshape(b, ng, gs, 3, self.num_heads, c//self.num_heads).permute(3, 0, 1, 4, 2, 5)  # 3, b, ng, nh, gs, c//nh
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Q @ K^T
        attn = (q @ k.transpose(-2, -1))  # b, ng, nh, gs, gs

        logit_scale = torch.clamp(self.logit_scale, max=torch.log(torch.tensor(1. / 0.01)).to(qkv.device)).exp()
        attn = attn * logit_scale

        # softmax
        attn = self.softmax(attn)  # b, ng, nh, gs, gs

        # Attn * V
        y = (attn @ v).permute(0, 1, 3, 2, 4).reshape(b, n+pad_n, c)[:, :n, :]

        x = feature_shuffle(y, x_sort_indices_reverse)
        x = self.proj(x)

        return x


    def flops(self, n):
        flops = 0

        # attn = (q @ k.transpose(-2, -1))
        flops += n * self.dim * self.category_size
        #  x = (attn @ v)
        flops += n * self.dim * self.category_size
        # x = self.proj(x)
        flops += n * self.dim * self.dim

        return flops


class ATDTransformerLayer(nn.Module):
    r"""
    ATD Transformer Layer

    Args:
        dim (int): Number of input channels.
        idx (int): Layer index.
        input_resolution (tuple[int]): Input resolution.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        category_size (int): Category size for AC-MSA.
        num_tokens (int): Token number for each token dictionary.
        reducted_dim (int): Reducted dimension number for query and key matrix.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
        is_last (bool): True if this layer is the last of a ATD Block. Default: False 
    """

    def __init__(self,
                 dim,
                 idx,
                 input_resolution,
                 num_heads,
                 window_size,
                 shift_size,
                 category_size,
                 num_tokens,
                 reducted_dim,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 is_last=False,
                 ):
        super().__init__()

        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        self.convffn_kernel_size = convffn_kernel_size
        self.num_tokens=num_tokens
        self.softmax = nn.Softmax(dim=-1)
        self.lrelu = nn.LeakyReLU()
        self.sigmoid = nn.Sigmoid()
        self.reducted_dim = reducted_dim
        self.is_last = is_last

        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)
        if not is_last:
            self.norm3 = nn.InstanceNorm1d(num_tokens, affine=True)
            self.sigma = nn.Parameter(torch.zeros([num_tokens, 1]), requires_grad=True)

        self.wqkv = nn.Linear(dim, 3*dim, bias=qkv_bias)

        self.attn_win = WindowAttention(
            self.dim,
            window_size=to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
        )
        self.attn_atd = ATD_CA(
            self.dim,
            input_resolution=input_resolution,
            qkv_bias=qkv_bias,
            num_tokens=num_tokens,
            reducted_dim=reducted_dim
        )
        self.attn_aca = AC_MSA(
            self.dim,
            input_resolution=input_resolution,
            num_tokens=num_tokens,
            num_heads=num_heads,
            category_size=category_size,
            qkv_bias=qkv_bias,
        )

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.convffn = ConvFFN(in_features=dim, hidden_features=mlp_hidden_dim, kernel_size=convffn_kernel_size, act_layer=act_layer)


    def forward(self, x, td, x_size, params):
        h, w = x_size
        b, n, c = x.shape
        c3 = 3 * c

        shortcut = x
        x = self.norm1(x)
        qkv = self.wqkv(x)

        # ATD_CA
        x_atd, sim_atd = self.attn_atd(x, td, x_size)  # x_atd: (b, n, c)  sim_atd: (b, n,)

        # AC_MSA
        x_aca = self.attn_aca(qkv, sim_atd, x_size) 

        # SW-MSA
        qkv = qkv.reshape(b, h, w, c3)

        # cyclic shift
        if self.shift_size > 0:
            shifted_qkv = torch.roll(qkv, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
            attn_mask = params['attn_mask']
        else:
            shifted_qkv = qkv
            attn_mask = None

        # partition windows
        # x_windows, batch_list = window_partitionx(shifted_qkv.permute(0, 3, 1, 2), self.window_size)
        # x_windows = x_windows.permute(0, 2, 3, 1)
        x_windows, batch_list = window_partition_(shifted_qkv, self.window_size)  # nw*b, window_size, window_size, c
        x_windows = x_windows.view(-1, self.window_size * self.window_size, c3)  # nw*b, window_size*window_size, c

        # W-MSA/SW-MSA (to be compatible for testing on images whose shapes are the multiple of window size
        attn_windows = self.attn_win(x_windows, rpi=params['rpi_sa'], mask=attn_mask)

        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, c)
        shifted_x = window_reverse_(attn_windows, self.window_size, h, w, batch_list)  # b h' w' c
        # shifted_x = window_reversex(attn_windows.permute(0, 3, 1, 2), self.window_size, h, w, batch_list).permute(0, 2, 3, 1)
        # reverse cyclic shift
        if self.shift_size > 0:
            attn_x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            attn_x = shifted_x
        x_win = attn_x
        x = shortcut + x_win.view(b, n, c) + x_atd + x_aca

        # FFN
        x = x + self.convffn(self.norm2(x), x_size)

        b, N, c = x.shape
        b, n, c = td.shape
        
        # Adaptive Token Refinement
        if not self.is_last:
            mask_soft = self.softmax(self.norm3(sim_atd.transpose(-1, -2)))
            mask_x = x.reshape(b, N, c)
            s = self.sigmoid(self.sigma)
            td = s*td + (1-s)*torch.einsum('btn,bnc->btc', mask_soft, mask_x)

        return x, td


    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.input_resolution if input_resolution is None else input_resolution

        # qkv = self.wqkv(x)
        flops += self.dim * 3 * self.dim * h * w

        # W-MSA/SW-MSA, ATD-CA, AC-MSA
        nw = h * w / self.window_size / self.window_size
        flops += nw * self.attn_win.flops(self.window_size * self.window_size)
        flops += self.attn_atd.flops(h * w)
        flops += self.attn_aca.flops(h * w)

        # mlp
        flops += 2 * h * w * self.dim * self.dim * self.mlp_ratio
        flops += h * w * self.dim * self.convffn_kernel_size**2 * self.mlp_ratio

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
        x: b, h*w, c
        """
        h, w = self.input_resolution
        b, seq_len, c = x.shape
        assert seq_len == h * w, 'input feature has wrong size'
        assert h % 2 == 0 and w % 2 == 0, f'x size ({h}*{w}) are not even.'

        x = x.view(b, h, w, c)

        x0 = x[:, 0::2, 0::2, :]  # b h/2 w/2 c
        x1 = x[:, 1::2, 0::2, :]  # b h/2 w/2 c
        x2 = x[:, 0::2, 1::2, :]  # b h/2 w/2 c
        x3 = x[:, 1::2, 1::2, :]  # b h/2 w/2 c
        x = torch.cat([x0, x1, x2, x3], -1)  # b h/2 w/2 4*c
        x = x.view(b, -1, 4 * c)  # b h/2*w/2 4*c

        x = self.norm(x)
        x = self.reduction(x)

        return x

    def extra_repr(self) -> str:
        return f'input_resolution={self.input_resolution}, dim={self.dim}'

    def flops(self, input_resolution=None):
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops = h * w * self.dim
        flops += (h // 2) * (w // 2) * 4 * self.dim * 2 * self.dim
        return flops


class BasicBlock(nn.Module):
    """ A basic ATD Block for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        idx (int): Block index.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        category_size (int): Category size for AC-MSA.
        num_tokens (int): Token number for each token dictionary.
        reducted_dim (int): Reducted dimension number for query and key matrix.
        convffn_kernel_size (int): Convolutional kernel size for ConvFFN.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 idx,
                 depth,
                 num_heads,
                 window_size,
                 category_size,
                 num_tokens,
                 convffn_kernel_size,
                 reducted_dim,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False, ):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.idx = idx

        self.layers = nn.ModuleList()
        for i in range(depth):
            self.layers.append(
                ATDTransformerLayer(
                    dim=dim,
                    idx=i,
                    input_resolution=input_resolution,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,
                    category_size=category_size,
                    num_tokens=num_tokens,
                    convffn_kernel_size=convffn_kernel_size,
                    reducted_dim=reducted_dim, 
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                    is_last=i == depth-1,
                )
            )

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

        # Token Dictionary
        self.td = nn.Parameter(torch.randn([num_tokens, dim]), requires_grad=True)

    def forward(self, x, x_size, params):
        b, n, c = x.shape
        td = self.td.repeat([b, 1, 1])
        for layer in self.layers:
            # adjust the value of idx_checkpoint to change the number of layers processed by checkpoint_wrapper
            # increase the value of idx_checkpoint could save more GPU memory footprint but slow down the training
            # idx_checkpoint need to be set as at least 4 for eight 24G GPU when training ATD
            idx_checkpoint = 4
            if self.use_checkpoint and self.idx < idx_checkpoint:
                layer = checkpoint_wrapper(layer, offload_to_cpu=False)
            x, td = layer(x, td, x_size, params)
        if self.downsample is not None:
            x = self.downsample(x)
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}'

    def flops(self, input_resolution=None):
        flops = 0
        for layer in self.layers:
            flops += layer.flops(input_resolution)
        if self.downsample is not None:
            flops += self.downsample.flops(input_resolution)
        return flops


class ATDB(nn.Module):
    """Adaptive Token Dictionary Block (ATDB).

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
        img_size: Input image size.
        patch_size: Patch size.
        resi_connection: The convolutional block before residual connection.
    """

    def __init__(self,
                 dim,
                 idx,
                 input_resolution,
                 depth,
                 num_heads,
                 window_size,
                 category_size,
                 num_tokens,
                 reducted_dim,
                 convffn_kernel_size,
                 mlp_ratio,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 downsample=None,
                 use_checkpoint=False,
                 img_size=224,
                 patch_size=4,
                 resi_connection='1conv', ):
        super(ATDB, self).__init__()

        self.dim = dim
        self.input_resolution = input_resolution

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.patch_unembed = PatchUnEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=0, embed_dim=dim, norm_layer=None)

        self.residual_group = BasicBlock(
            dim=dim,
            input_resolution=input_resolution,
            idx=idx,
            depth=depth,
            num_heads=num_heads,
            window_size=window_size,
            num_tokens=num_tokens,
            category_size=category_size,
            reducted_dim=reducted_dim,
            convffn_kernel_size=convffn_kernel_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            norm_layer=norm_layer,
            downsample=downsample,
            use_checkpoint=use_checkpoint,
        )

        if resi_connection == '1conv':
            self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv = nn.Sequential(
                nn.Conv2d(dim, dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(dim // 4, dim, 3, 1, 1))

    def forward(self, x, x_size, params):
        return self.patch_embed(self.conv(self.patch_unembed(self.residual_group(x, x_size, params), x_size))) + x

    def flops(self, input_resolution=None):
        flops = 0
        flops += self.residual_group.flops(input_resolution)
        h, w = self.input_resolution if input_resolution is None else input_resolution
        flops += h * w * self.dim * self.dim * 9
        flops += self.patch_embed.flops(input_resolution)
        flops += self.patch_unembed.flops(input_resolution)

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

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()  # b Ph*Pw c
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self, input_resolution=None):
        flops = 0
        h, w = self.img_size if input_resolution is None else input_resolution
        if self.norm is not None:
            flops += h * w * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

    def forward(self, x, x_size):
        x = x.transpose(1, 2).view(x.shape[0], self.embed_dim, x_size[0], x_size[1]).contiguous()  # b Ph*Pw c
        return x

    def flops(self, input_resolution=None):
        flops = 0
        return flops


class Upsample(nn.Sequential):
    """Upsample module.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.
    """

    def __init__(self, scale, num_feat):
        m = []
        self.scale = scale
        self.num_feat = num_feat
        if (scale & (scale - 1)) == 0:  # scale = 2^n
            for _ in range(int(math.log(scale, 2))):
                m.append(nn.Conv2d(num_feat, 4 * num_feat, 3, 1, 1))
                m.append(nn.PixelShuffle(2))
        elif scale == 3:
            m.append(nn.Conv2d(num_feat, 9 * num_feat, 3, 1, 1))
            m.append(nn.PixelShuffle(3))
        else:
            raise ValueError(f'scale {scale} is not supported. Supported scales: 2^n and 3.')
        super(Upsample, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        x, y = input_resolution
        if (self.scale & (self.scale - 1)) == 0:
            flops += self.num_feat * 4 * self.num_feat * 9 * x * y * int(math.log(self.scale, 2))
        else:
            flops += self.num_feat * 9 * self.num_feat * 9 * x * y
        return flops


class UpsampleOneStep(nn.Sequential):
    """UpsampleOneStep module (the difference with Upsample is that it always only has 1conv + 1pixelshuffle)
       Used in lightweight SR to save parameters.

    Args:
        scale (int): Scale factor. Supported scales: 2^n and 3.
        num_feat (int): Channel number of intermediate features.

    """

    def __init__(self, scale, num_feat, num_out_ch, input_resolution=None):
        self.num_feat = num_feat
        self.input_resolution = input_resolution
        m = []
        m.append(nn.Conv2d(num_feat, (scale ** 2) * num_out_ch, 3, 1, 1))
        m.append(nn.PixelShuffle(scale))
        super(UpsampleOneStep, self).__init__(*m)

    def flops(self, input_resolution):
        flops = 0
        h, w = self.patches_resolution if input_resolution is None else input_resolution
        flops = h * w * self.num_feat * 3 * 9
        return flops


# @ARCH_REGISTRY.register()
class ATD(nn.Module):
    r""" ATD
        A PyTorch impl of : `Transcending the Limit of Local Window: Advanced Super-Resolution Transformer 
                             with Adaptive Token Dictionary`.

    Args:
        img_size (int | tuple(int)): Input image size. Default 64
        patch_size (int | tuple(int)): Patch size. Default: 1
        in_chans (int): Number of input image channels. Default: 3
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 2
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
        upscale: Upscale factor. 2/3/4/8 for image SR, 1 for denoising and compress artifact reduction
        img_range: Image range. 1. or 255.
        upsampler: The reconstruction reconstruction module. 'pixelshuffle'/'pixelshuffledirect'/'nearest+conv'/None
        resi_connection: The convolutional block before residual connection. '1conv'/'3conv'
    """

    def __init__(self,
                 img_size=64,
                 patch_size=1,
                 in_chans=3,
                 embed_dim=90,
                 depths=(6, 6, 6, 6),
                 num_heads=(6, 6, 6, 6),
                 window_size=8,
                 category_size=128,
                 num_tokens=64,
                 ker_size=64,
                 reducted_dim=4,
                 convffn_kernel_size=5,
                 mlp_ratio=2.,
                 qkv_bias=True,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 use_checkpoint=False,
                 upscale=2,
                 img_range=1.,
                 upsampler='',
                 resi_connection='1conv',
                 FKE=True,
                 FKE_last=False,
                 sub_image=True,
                 get_ker=False,
                 norm_type='LayerNorm',
                 overlap_size=[16, 16],
                 enc_ker_layers=[1,1,1,1],
                 last_ker_layers=[2, 2, 2, 2],
                 baseblock_enc_ker=['EVSS', 'EVSS','EVSS','EVSS'],
                 save_memorys=[False, False,False,False,False,False,False,False],
                 **kwargs):
        super().__init__()
        # overlap_size = [img_size//4, img_size//4]
        num_in_ch = in_chans
        num_out_ch = in_chans
        num_feat = 64
        self.h, self.w = 0, 0
        self.get_ker = get_ker
        self.FKE_last = FKE_last
        self.sub_image = sub_image
        self.image_size = [img_size, img_size]
        self.FKE = FKE
        self.up_scale = upscale
        self.ker_size = [ker_size, ker_size]
        self.ker_size_up = [ker_size * self.up_scale, ker_size * self.up_scale]
        self.overlap_size = overlap_size
        self.overlap_size_up = [overlap_size[0] * self.up_scale, overlap_size[1] * self.up_scale]
        self.dynic_overlap = False
        self.img_range = img_range
        if in_chans == 3:
            rgb_mean = (0.4488, 0.4371, 0.4040)
            self.mean = torch.Tensor(rgb_mean).view(1, 3, 1, 1)
        else:
            self.mean = torch.zeros(1, 1, 1, 1)
        self.upscale = upscale
        self.upsampler = upsampler

        # ------------------------- 1, shallow feature extraction ------------------------- #
        self.conv_first = nn.Conv2d(num_in_ch, embed_dim, 3, 1, 1)

        # ------------------------- 2, deep feature extraction ------------------------- #
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = embed_dim
        self.mlp_ratio = mlp_ratio
        self.window_size = window_size

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # merge non-overlapping patches into image
        self.patch_unembed = PatchUnEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=embed_dim,
            embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # relative position index
        relative_position_index_SA = self.calculate_rpi_sa()
        self.register_buffer('relative_position_index_SA', relative_position_index_SA)
        if FKE_last:
            self.conv_last_x = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.last_ker_est = KernelEstimator(num_out_ch, width=[num_feat, num_feat, num_feat], first_col=True,
                                      save_memory=save_memorys[-1],
                                      width_d=[num_feat, num_feat, num_feat], enc_ker_layers=last_ker_layers,
                                      baseblock_enc_ker=baseblock_enc_ker,
                                      norm_type=norm_type, train_size=self.ker_size_up,
                                      patch_size=self.ker_size_up, overlap_size=self.overlap_size_up)
        # build Residual Adaptive Token Dictionary Blocks (ATDB)
        self.layers = nn.ModuleList()
        self.ker_ests = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = ATDB(
                dim=embed_dim,
                idx=i_layer,
                input_resolution=(patches_resolution[0], patches_resolution[1]),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                window_size=window_size,
                category_size=category_size,
                num_tokens=num_tokens,
                reducted_dim=reducted_dim,
                convffn_kernel_size=convffn_kernel_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=qkv_bias,
                norm_layer=norm_layer,
                downsample=None,
                use_checkpoint=use_checkpoint,
                img_size=img_size,
                patch_size=patch_size,
                resi_connection=resi_connection,
            )
            self.layers.append(layer)

            if FKE:
                first_col = True if i_layer == 0 else False
                ker_est = KernelEstimator(num_out_ch, width=[embed_dim, embed_dim, embed_dim], first_col=first_col, get_ker=get_ker, save_memory=save_memorys[i_layer],
                                          width_d=[embed_dim, embed_dim, embed_dim], enc_ker_layers=enc_ker_layers, baseblock_enc_ker=baseblock_enc_ker,
                                          norm_type=norm_type, train_size=[ker_size, ker_size], patch_size=[ker_size, ker_size], overlap_size=overlap_size)
                self.ker_ests.append(ker_est)
        self.norm = norm_layer(self.num_features)

        # build the last conv layer in deep feature extraction
        if resi_connection == '1conv':
            self.conv_after_body = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        elif resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim // 4, 3, 1, 1), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim // 4, 1, 1, 0), nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(embed_dim // 4, embed_dim, 3, 1, 1))

        # ------------------------- 3, high quality image reconstruction ------------------------- #
        if self.upsampler == 'pixelshuffle':
            self.out_channels = num_feat
            # for classical SR
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.upsample = Upsample(upscale, num_feat)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        elif self.upsampler == 'pixelshuffledirect':
            self.out_channels = num_out_ch
            # for lightweight SR (to save parameters)
            self.upsample = UpsampleOneStep(upscale, embed_dim, num_out_ch,
                                            (patches_resolution[0], patches_resolution[1]))
        elif self.upsampler == 'nearest+conv':
            # for real-world SR (less artifacts)
            self.out_channels = num_feat
            assert self.upscale == 4, 'only support x4 now.'
            self.conv_before_upsample = nn.Sequential(
                nn.Conv2d(embed_dim, num_feat, 3, 1, 1), nn.LeakyReLU(inplace=True))
            self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
            self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
            self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        else:
            # for image denoising and JPEG compression artifact reduction
            self.out_channels = num_feat
            self.conv_last = nn.Conv2d(embed_dim, num_out_ch, 3, 1, 1)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    def grids(self, x, level=0):
        b, c, h, w = x.shape
        n_level = 2 ** level

        assert b == 1
        k1, k2 = self.ker_size[0] // (n_level), self.ker_size[1] // (n_level)
        # print(h, w, k1, k2)
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if h < 1.25 * k1:
        #     k1 = h
        # if w < 1.25 * k2:
        #     k2 = w
        # print(k1, k2)
        if self.dynic_overlap and n_level == 1:
            num_row = h // self.ker_size[0] + 1
            num_col = w // self.ker_size[1] + 1
            self.overlap_size[0] = (num_row * k1 - h) // (num_row - 1)
            self.overlap_size[1] = (num_col * k2 - w) // (num_col - 1)

        overlap_size = [self.overlap_size[0] // (n_level), self.overlap_size[1] // (n_level)]  # (64, 64)

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
                idxes.append({'i': i*self.up_scale, 'j': j*self.up_scale})
                j = j + step_j
            i = i + step_i
        # print(idxes)
        parts = torch.cat(parts, dim=0)
        if level == 0:
            self.original_size = (b, self.out_channels, h*self.up_scale, w*self.up_scale)
            self.stride = (stride[0]*self.up_scale, stride[1]*self.up_scale)
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
        self.ek1 = self.nr * self.stride[0] + self.overlap_size_up[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.overlap_size_up[1] * 2 - w
        if self.h != h or self.w != w:
            self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
            self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
            self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
            self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
        if self.h != h:
            self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1)
            self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1)
        if self.w != w:
            self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2)
            self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2)
        self.h = h
        self.w = w

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
        k1, k2 = self.ker_size_up
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if h < 1.25 * k1:
        #     k1 = h
        # if w < 1.25 * k2:
        #     k2 = w
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)
        # rounding_mode = 'floor'
        # print(self.idxes)
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']

            if i != 0 and i + k1 != h:
                # print(1)
                outs[cnt, :, :self.overlap_size_up[0], :] = torch.mul(outs[cnt, :, :self.overlap_size_up[0], :],
                                                                      self.fuse_matrix_h2.to(outs.device))
                # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
            if i + k1 * 2 - self.ek1 < h:
                # print(2)
                outs[cnt, :, -self.overlap_size_up[0]:, :] = torch.mul(outs[cnt, :, -self.overlap_size_up[0]:, :],
                                                                       self.fuse_matrix_h1.to(outs.device))
            # print(self.fuse_matrix_eh1.dtype)
            if i + k1 == h and i !=0:
                # print(3)
                outs[cnt, :, :self.ek1, :] = torch.mul(outs[cnt, :, :self.ek1, :], self.fuse_matrix_eh2.to(outs.device))
            if i + k1 * 2 - self.ek1 == h:
                # print(4)
                outs[cnt, :, -self.ek1:, :] = torch.mul(outs[cnt, :, -self.ek1:, :],
                                                        self.fuse_matrix_eh1.to(outs.device))

            if j != 0 and j + k2 != w:
                # print(1)
                outs[cnt, :, :, :self.overlap_size_up[1]] = torch.mul(outs[cnt, :, :, :self.overlap_size_up[1]],
                                                                      self.fuse_matrix_w2.to(outs.device))
            if j + k2 * 2 - self.ek2 < w:
                # print(2)
                # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, -self.overlap_size_up[1]:] = torch.mul(outs[cnt, :, :, -self.overlap_size_up[1]:],
                                                                       self.fuse_matrix_w1.to(outs.device))
            if j + k2 == w and j != 0:
                # print(3)
                # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                outs[cnt, :, :, :self.ek2] = torch.mul(outs[cnt, :, :, :self.ek2], self.fuse_matrix_ew2.to(outs.device))
            if j + k2 * 2 - self.ek2 == w:
                # print(4)
                # print('j + k2*2 - self.ek2 == w: ')
                outs[cnt, :, :, -self.ek2:] = torch.mul(outs[cnt, :, :, -self.ek2:],
                                                        self.fuse_matrix_ew1.to(outs.device))
            # print(i, j)
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
    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def forward_features(self, x, params):
        x_size = (x.shape[2], x.shape[3])
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        if self.FKE:
            ker_feature = [0, 0, 0, 0]
            for layer, ker_est in zip(self.layers, self.ker_ests):
                x = layer(x, x_size, params)
                x = self.patch_unembed(x, x_size)
                out_dict = ker_est(x, ker_feature=ker_feature)
                # out_dict = self.ker_est(x)
                x = out_dict['feature_deblur']
                ker_feature = out_dict['ker_feature']
                x = self.patch_embed(x)
            if self.get_ker:
                visual_feature = out_dict['feature_visual']
        else:
            for layer in self.layers:
                x = layer(x, x_size, params)
        # for layer in self.layers:
        #     x = layer(x, x_size, params)

        x = self.norm(x)  # b seq_len c
        x = self.patch_unembed(x, x_size)
        if self.get_ker:
            return x, visual_feature
        else:
            return x
    
    def calculate_rpi_sa(self):
        # calculate relative position index for SW-MSA
        coords_h = torch.arange(self.window_size)
        coords_w = torch.arange(self.window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        return relative_position_index
    
    def calculate_mask(self, x_size):
        # calculate attention mask for SW-MSA
        h, w = x_size
        img_mask = torch.zeros((1, h, w, 1))  # 1 h w 1
        h_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        w_slices = (slice(0, -self.window_size), slice(-self.window_size,
                                                       -(self.window_size // 2)), slice(-(self.window_size // 2), None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        # mask_windows, _ = window_partitionx(img_mask.permute(0, 3, 1, 2), self.window_size)
        # mask_windows = mask_windows.permute(0, 2, 3, 1)
        mask_windows, _ = window_partition_(img_mask, self.window_size)  # nw, window_size, window_size, 1
        mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))

        return attn_mask

    def forward(self, x, gt=None, batch_size=None):
        # padding
        h_ori, w_ori = x.size()[-2], x.size()[-1]
        # mod = self.window_size
        # h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
        # w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
        # h, w = h_ori + h_pad, w_ori + w_pad
        # x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
        # x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]
        out_dict_ = {}
        self.mean = self.mean.type_as(x)
        x = (x - self.mean) * self.img_range

        x = self.conv_first(x)
        # print(x.shape)
        if self.sub_image and not self.training:
            x = self.grids(x, 0)
            attn_mask = self.calculate_mask([x.shape[-2], x.shape[-1]]).to(x.device)
            # attn_mask = self.calculate_mask(self.ker_size).to(x.device)
            params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}
            # params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}
            # params = {'attn_mask': self.grids(attn_mask, 0), 'rpi_sa': self.grids(self.relative_position_index_SA, 0)}
            # print(x.shape,'grid')
            if batch_size is None:
                x = self.forward_subimage(x, params)
                x = self.grids_inverse(x)
            else:
                all_batch = x[0].shape[0]
                batchs = range(0, all_batch, batch_size)

                out_parts = []
                # print(inp_img_.shape)
                for batch_z in batchs:
                    inps_pry_ = []
                    if batch_z + batch_size <= all_batch:
                        x_ = x[batch_z:batch_z + batch_size, ...]
                    else:
                        x_ = x[batch_z:, ...]
                        if len(x_.shape) == 3:
                            x_ = x_.unsqueeze(0)

                    x_ = self.forward_subimage(x_, params)
                    out_parts.append(x_)
                x = torch.cat(out_parts, dim=0)
                if self.sub_image and not self.training:
                    x = self.grids_inverse(x)
        else:

            # mod = self.window_size
            # h_pad = ((h_ori + mod - 1) // mod) * mod - h_ori
            # w_pad = ((w_ori + mod - 1) // mod) * mod - w_ori
            # h, w = h_ori + h_pad, w_ori + w_pad
            # x = torch.cat([x, torch.flip(x, [2])], 2)[:, :, :h, :]
            # x = torch.cat([x, torch.flip(x, [3])], 3)[:, :, :, :w]
            # print(x.shape, 'no grid')
            attn_mask = self.calculate_mask([x.shape[-2], x.shape[-1]]).to(x.device)
            # attn_mask = self.calculate_mask(self.ker_size).to(x.device)
            params = {'attn_mask': attn_mask, 'rpi_sa': self.relative_position_index_SA}
            if self.get_ker:
                x, visual_feature= self.forward_subimage(x, params)
            else:
                x = self.forward_subimage(x, params)


        if self.upsampler != 'pixelshuffledirect':
            if self.FKE_last:
                latent_sr = self.conv_last_x(x)
                latent_sr = latent_sr / self.img_range + self.mean
                latent_sr = latent_sr[..., :h_ori * self.upscale, :w_ori * self.upscale]
                out_dict = self.last_ker_est(x, ker_feature=[0, 0, 0, 0])
                # out_dict = ker_est(x, ker_feature=ker_feature)
                # out_dict = self.ker_est(x)
                x = out_dict['feature_deblur']
                # ker_feature = out_dict['ker_feature']
            x = self.conv_last(x)
        x = x / self.img_range + self.mean

        # unpadding
        x = x[..., :h_ori * self.upscale, :w_ori * self.upscale]
        if self.get_ker:
            return {'img': x, 'feature_visual': [visual_feature]}
        if self.FKE_last:
            return [latent_sr, x]
        else:
            return x
    def forward_subimage(self, x, params):

        if self.upsampler == 'pixelshuffle':
            # for classical SR
            x_body = self.forward_features(x, params)
            # if self.FKE:
            #     # print(x.shape)
            #     out_dict = self.ker_est(x_body)
            #     x_body = out_dict['feature_deblur']
            x = self.conv_after_body(x_body) + x

            # if self.FKE:
            #     # print(x.shape)
            #     out_dict = self.ker_est(x)
            #     x = out_dict['feature_deblur']
            x = self.conv_before_upsample(x)

            x = self.upsample(x)

        elif self.upsampler == 'pixelshuffledirect':
            # for lightweight SR
            # x = self.conv_first(x)
            if self.get_ker:
                x_body, visual_feature = self.forward_features(x, params)
            else:
                x_body = self.forward_features(x, params)
            # v3
            # if self.FKE:
            #     # print(x.shape)
            #     out_dict = self.ker_est(x_body)
            #     x_body = out_dict['feature_deblur']
            x = self.conv_after_body(x_body) + x
            # if self.FKE:
            #     # print(x.shape)
            #     out_dict = self.ker_est(x)
            #     x = out_dict['feature_deblur']
            # if self.FKE:
            #     # print(x.shape)
            #     out_dict = self.ker_est(x)
            #     x = out_dict['feature_deblur']
            x = self.upsample(x)

        elif self.upsampler == 'nearest+conv':
            # for real-world SR
            # x = self.conv_first(x)
            x = self.conv_after_body(self.forward_features(x, params)) + x
            x = self.conv_before_upsample(x)
            # if self.FKE:
            #     # print(x.shape)
            #     out_dict = self.ker_est(x)
            #     x = out_dict['feature_deblur']
            x = self.lrelu(self.conv_up1(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_up2(torch.nn.functional.interpolate(x, scale_factor=2, mode='nearest')))
            x = self.lrelu(self.conv_hr(x))
        # else:
        #     # for image denoising and JPEG compression artifact reduction
        #     x_first = x # self.conv_first(x)
        #     res = self.conv_after_body(self.forward_features(x_first, params)) + x_first
        #     x = x + self.conv_last(res)

        if self.get_ker:
            return x, visual_feature
        else:
            return x
    def flops(self, input_resolution=None):
        flops = 0
        resolution = self.patches_resolution if input_resolution is None else input_resolution
        h, w = resolution
        flops += h * w * 3 * self.embed_dim * 9
        flops += self.patch_embed.flops(resolution)
        for layer in self.layers:
            flops += layer.flops(resolution)

        flops += h * w * 3 * self.embed_dim * self.embed_dim
        if self.upsampler == 'pixelshuffle':
            flops += self.upsample.flops(resolution)
        else:
            flops += self.upsample.flops(resolution)

        return flops


if __name__ == '__main__':
    # upscale = 4
    #
    # model = ATD(
    #     upscale=4,
    #     img_size=64,
    #     embed_dim=210,
    #     depths=[6, 6, 6, 6, 6, 6, ],
    #     num_heads=[6, 6, 6, 6, 6, 6, ],
    #     window_size=16,
    #     category_size=128,
    #     num_tokens=128,
    #     reducted_dim=20,
    #     convffn_kernel_size=5,
    #     img_range=1.,
    #     down_c=20,
    #     mlp_ratio=2,
    #     upsampler='pixelshuffle')
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    # yaml_file = '../../../Motion_Deblurring/Options/Abalation-DeepRFTv2-200k-4gpu.yml'
    yaml_file = '../../../SR/Options/101_ATD_light_SRx2_scratch.yml'
    # yaml_file = '../../../SR/Options/000_ATD_SRx2_scratch.yml'
    # yaml_file = '../../../Motion_Deblurring/Options/Abalation-DeepRFTv2-postrain-200k-4gpu.yml'
    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    s = x['network_g'].pop('type')
    ##########################
    print(x['network_g'])

    inp = torch.randn((1, 3, 256, 256)).cuda()
    model = ATD(**x['network_g']).cuda()
    model.eval()
    # Model Size
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.3fM" % (total / 1e6))
    # print(128, 128, model.flops([128, 128]) / 1e9, 'G')
    # print(256, 256, model.flops([256, 256]) / 1e9, 'G')
    #
    # # Test
    # _input = torch.randn([2, 3, 64, 64])
    # output = model(_input)
    # print(output.shape)
