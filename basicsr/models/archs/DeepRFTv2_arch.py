import kornia.geometry
from basicsr.models.archs.DeepRFTv2_util import *
from basicsr.models.archs.utils_logarithmicfft import *

# Global Local Hierarchical Fourier Network

class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        x = self.proj(x)

        return x


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)
class Downsample1(nn.Module):
    def __init__(self):
        super(Downsample1, self).__init__()

    def forward(self, x):
        return F.interpolate(x, 0.5)
class Downsample2(nn.Module):
    def __init__(self, n_feat, out_feat):
        super(Downsample2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, out_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)

class Downsample3(nn.Module):
    def __init__(self, n_feat, out_feat):
        super(Downsample3, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, out_feat // 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelUnshuffle(2),
                                  LayerNorm2d(out_feat))

    def forward(self, x):
        return self.body(x)

class Downsample4(nn.Module):
    def __init__(self, n_feat, out_feat):
        super(Downsample4, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, out_feat, kernel_size=2, stride=2, padding=0, bias=False))

    def forward(self, x):
        return self.body(x)
class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)

class Upsample2(nn.Module):
    def __init__(self, n_feat, out_feat):
        super(Upsample2, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, out_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)
class Upsample3(nn.Module):
    def __init__(self, n_feat, out_feat):
        super(Upsample3, self).__init__()

        self.body = nn.Sequential(nn.Conv2d(n_feat, out_feat * 4, kernel_size=3, stride=1, padding=1, bias=False),
                                  nn.PixelShuffle(2),
                                  LayerNorm2d(out_feat))

    def forward(self, x):
        return self.body(x)

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
                 dec_ker_layers=[3, 2, 1], num_heads_enc=[1, 1, 1, 1], train_size=[256, 256], patch_size=[256, 256], overlap_size=[48,48], ks=[32, 16, 8],
                 save_memory=True, drop_path=0., kernel_size=[65, 33, 17], k_len=61, upx=False, scale=0, first_col=False, spatial_add=False, get_ker=False,
                 frozen_kernel=False, sub_image_pad=False, select_range=None, flip_transpose=False, inp_feature='inp_latentsharp_feature', deconv_feature='feature', norm='backward', mode='rfft', norm_type='None', mask=False, spatial_decoder=False):
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

        self.end_feature1 = nn.Sequential(
            nn.Conv2d(in_channels=width_d[0], out_channels=3, kernel_size=3, padding=1, stride=1, groups=1, bias=True),
        )

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

    def forward(self, inp, blur_img, deblur_img, global_img=None, ker_feature=[0, 0, 0, 0], spatial_feature=[0, 0, 0, 0], gt=None, batch_size=None):
        inpx = inp[0]

        _, _, H, W = inp[0].shape
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
        if H > self.patch_size[0] or W > self.patch_size[1]:
            # feature_deblur1 = window_reversex(feature_deblur1, self.patch_size, H, W, batch_list)
            if self.training or (self.overlap_size[0] == 0 and self.overlap_size[1] == 0):
                # print(feature_deblur1.shape)
                feature_deblur1 = window_reversex(feature_deblur1, self.patch_size, H, W, batch_list)
            else:
                feature_deblur1 = self.grids_inverse(feature_deblur1, type_out=inp[0].dtype)
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

            feature_deblur1 = self.deblur_feature1(torch.cat([feature_deblur1, inp[0]], dim=1))
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
        blur_pattern = self.end_feature1(feature_deblur1)
        # if self.flip_transpose:
        #     x0, x1, x2, x3 = blur_pattern.chunk(4, dim=0)
        #     x1 = torch.rot90(x1, k=2, dims=[-2, -1])
        #     x2 = torch.flipud(x2)
        #     x3 = torch.flipud(x3)
        #     x3 = torch.rot90(x3, k=2, dims=[-2, -1])
        #     blur_pattern = (x0 + x1 + x2 + x3) / 4.0
        results_deblur_max = blur_pattern + blur_img
        # if self.select_range is not None:
        #     results_deblur_max_ = deblur_img
        #     h_start = self.select_range[0]
        #     h_end = self.select_range[1]
        #     w_start = self.select_range[2]
        #     w_end = self.select_range[3]
        #     results_deblur_max_[..., h_start: h_end, w_start: w_end] = results_deblur_max[..., h_start: h_end, w_start: w_end]
        #     results_deblur_max = results_deblur_max_
        # results_deblur_max = torch.clamp(results_deblur_max, 0., 1.)
        out_dict['img_deblur'] = results_deblur_max

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
class BaseGoKModel(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dims=[48, 48, 48],
                 num_ker=48,
                 dims_ker=[48, 48, 48],
                 fuse_method_encoder='DownSampleAdd',
                 fuse_method_decoder='UpSampleAdd',
                 downsample='downsample',
                 num_heads_enc=[1, 2, 4],
                 num_heads_dec=[4, 2, 1],
                 baseblock_enc_small=['naf', 'naf', 'naf'],
                 baseblock_dec_small=['naf', 'naf', 'naf', 'naf'],
                 baseblock_enc_mid=['naf', 'naf', 'naf'],
                 baseblock_dec_mid=['naf', 'naf', 'naf', 'naf'],
                 baseblock_enc_max=['naf', 'naf', 'naf'],
                 baseblock_dec_max=['naf', 'naf', 'naf', 'naf'],
                 deblur_kernel=False,
                 latent_sharp=True,
                 spatial_decoder=False,
                 get_ker=False,
                 save_memory_global_encoder=[False, False],
                 save_memory_global_decoder=[False, False],
                 save_memory_kernel=[False],
                 deblur_kernel_mode='no_ker', inp_feature='inp_latentsharp_feature', deconv_feature='feature',
                 downsample_ker='noconcat_nodownsmaple',
                 upsample_mode='conv',
                 downsample_mode='conv',
                 fuse_upsample_mode='conv', fuse_downsample_mode='conv',
                 data_mode='BCHW',
                 baseblock_enc_ker=['naf', 'naf', 'naf'],
                 baseblock_dec_ker=['naf', 'naf', 'naf'],
                 enc_ker_layers=[1, 2, 3],  # 256 128 64
                 dec_ker_layers=[3, 2, 1],
                 enc_small_layers=[3, 6, 8],  # 64 32 16
                 dec_small_layers=[8, 6, 3],
                 enc_mid_layers=[2, 3, 4],  # 128 64 32
                 dec_mid_layers=[4, 3, 2],
                 enc_max_layers=[1, 2, 3],  # 256 128 64
                 dec_max_layers=[3, 2, 1],
                 num_global_unet=1,
                 train_size=[256, 256],
                 patch_size=[256, 256],
                 sub_image_size=[256, 256],
                 overlap_size=[48, 48],
                 drop_path=0.,
                 save_memory=True,
                 flip_transpose=False,
                 mid_flip=False,
                 select_range=None,
                 bias=False,
                 deblur_loss_weight=1.,
                 frozen_encoder=False,
                 up2scale=False,
                 small_scale=True,
                 mid_scale=True,
                 deblur_mask=False, norm_type='None', spatial_add=False,
                 kernel_size=65,
                 pad_size=0,
                 ks=[32, 16, 8],
                 scale_size=[256, 128, 64],
                 first_col_=False,
                 multiscale_enc=False, prior_model=None
                 ):
        super(BaseGoKModel, self).__init__()
        shortcut_scale_init_value = 0.5
        self.latent_sharp = latent_sharp
        self.ks = ks
        self.prior_model = prior_model
        self.deblur_kernel = deblur_kernel
        self.overlap_size = overlap_size
        self.ker_size = [int(train_size[0] * 1.), int(train_size[0] * 1.)]
        self.up_scale = 1
        self.overlap_size_up = (self.overlap_size[0] * self.up_scale, self.overlap_size[1] * self.up_scale)
        # self.kernel_size = [train_size, train_size]
        self.kernel_size_up = [self.ker_size[0] * self.up_scale, self.ker_size[1] * self.up_scale]
        self.out_channels = out_channels
        # print()
        self.get_ker = get_ker
        self.scale_size = scale_size
        self.mid_scale = mid_scale
        self.small_scale = small_scale

        self.up2scale = up2scale
        self.frozen_encoder = frozen_encoder

        self.mid_flip = mid_flip
        self.kernel_estimator = nn.ModuleList()
        self.mid_kernel_estimator = nn.ModuleList()

        self.small_kernel_estimator = nn.ModuleList()
        self.num_global_unet = num_global_unet
        self.subnets_small = nn.ModuleList()
        self.subdenets_small = nn.ModuleList()
        self.subnets_mid = nn.ModuleList()
        self.subdenets_mid = nn.ModuleList()
        self.subnets_max = nn.ModuleList()
        self.subdenets_max = nn.ModuleList()
        self.down_max1 = nn.ModuleList()
        self.down_max2 = nn.ModuleList()
        self.down_max3 = nn.ModuleList()
        self.down_max4 = nn.ModuleList()
        self.up_max0 = nn.ModuleList()
        self.up_max1 = nn.ModuleList()
        self.up_max2 = nn.ModuleList()
        self.up_max3 = nn.ModuleList()
        self.down_mid1 = nn.ModuleList()
        self.down_mid2 = nn.ModuleList()
        self.down_mid3 = nn.ModuleList()
        self.up_mid0 = nn.ModuleList()
        self.up_mid1 = nn.ModuleList()
        self.up_mid2 = nn.ModuleList()
        self.up_mid3 = nn.ModuleList()
        # self.latent_mid_fusions = nn.ModuleList()
        # self.latent_mid_BFs = nn.ModuleList()
        # self.latent_max_BFs = nn.ModuleList()
        self.down_mid2max = nn.ModuleList()
        self.down_small2mid = nn.ModuleList()
        # self.BF_smalls = nn.ModuleList()
        # self.BF_mids = nn.ModuleList()
        self.down_latent_max = nn.ModuleList()
        self.down_latent_mid = nn.ModuleList()
        self.down_latent_small = nn.ModuleList()
        self.max_down_l3 = nn.ModuleList()
        # self.mid_down_l1 = nn.ModuleList()
        # self.mid_down_l2 = nn.ModuleList()
        self.mid_down_l3 = nn.ModuleList()
        self.output_smalls = nn.ModuleList()
        self.output_mids = nn.ModuleList()
        self.output_maxs = nn.ModuleList()
        self.output_smalls2 = nn.ModuleList()
        self.output_mids2 = nn.ModuleList()
        # self.output_maxs2 = nn.ModuleList()
        self.down_max_kerfeatures = nn.ModuleList()
        self.down_mid_kerfeatures = nn.ModuleList()
        self.patch_embed_smalls = nn.ModuleList()
        self.patch_embed_mids = nn.ModuleList()
        self.patch_embed_maxs = nn.ModuleList()
        self.patch_embed_smalls2 = nn.ModuleList()
        self.patch_embed_mids2 = nn.ModuleList()
        self.patch_embed_maxs2 = nn.ModuleList()
        self.aff_mid = nn.ModuleList()
        self.aff_max = nn.ModuleList()

        self.fuse_method_encoder = fuse_method_encoder
        self.fuse_method_decoder = fuse_method_decoder
        self.fusd_mid_2 = nn.ModuleList()
        self.fusd_max_2 = nn.ModuleList()

        channels_enc_max = [dims[0], dims[0] * 2, dims[0] * 4]
        channels_dec_max = [dims[0] * 4, dims[0] * 2, dims[0]]

        channels_enc_mid = [dims[1], dims[1] * 2, dims[1] * 4]
        channels_dec_mid = [dims[1] * 4, dims[1] * 2, dims[1]]
        channels_enc_small = [dims[2], dims[2] * 2, dims[2] * 4]
        channels_dec_small = [dims[2] * 4, dims[2] * 2, dims[2]]
        dp_rate_e = [x.item() for x in torch.linspace(0, drop_path, len(enc_max_layers))]
        dp_rate_d = [x.item() for x in torch.linspace(0, drop_path, len(dec_max_layers))]
        dp_rate_d.reverse()

        self.fuse_in_encoder = ['Fuse_in_Encoder', 'Fuse_in_Encoder2shallow']
        # self.fuse_after_encoder = ['DownSample1Add012', 'DownSample2Add012', 'DownSample3Add012']
        self.fuse_before_encoder = ['DownSample2shallow', 'Fuse_in_Encoder2shallow', 'DownSample3shallow']
        self.fuse_no_encoder = ['NoFuse']
        self.fuse_in_decoder = ['Fuse_in_Decoder', 'Fuse_in_Decoder2latent']
        # self.fuse_after_decoder = ['UpSampleAdd012']
        self.fuse_before_decoder = ['DownSample2latent', 'DownSample3latent', 'DownSample4latent',
                                    'Fuse_in_Decoder2latent']
        self.fuse_no_decoder = ['NoFuse']
        self.multiscale_enc = multiscale_enc
        self.deblur_kernel_mode = deblur_kernel_mode
        save_memory_ue = False if self.frozen_encoder else save_memory
        if self.num_global_unet > 0:

            for i in range(self.num_global_unet):
                first_col = first_col_ if i == 0 else False
                save_memory_encoder = save_memory_global_encoder[i]
                save_memory_decoder = save_memory_global_decoder[i]

                first_col_enc_max = first_col if prior_model is None else False
                if self.deblur_kernel:
                    # save_memory_kernel = True
                    first_col_ker = True if i == 0 else False
                    # print(first_col_ker)
                    if self.deblur_kernel_mode == 'ker_max':
                        if i > 0:
                            if self.mid_scale:
                                self.down_max_kerfeatures.append(DownSample(dims[0], dims[1], fuse_downsample_mode))
                            if self.small_scale:
                                self.down_mid_kerfeatures.append(DownSample(dims[1], dims[2], fuse_downsample_mode))
                        else:
                            self.down_max_kerfeatures.append(nn.Identity())
                        self.kernel_estimator.append( # Patched Grided
                            KernelEstimator(img_channel=dims[0] * 2, num_ker=num_ker,
                                                                                 width=dims_ker,
                                                                                 width_d=dims, downsample=downsample_ker,
                                                                                 baseblock_enc_ker=baseblock_enc_ker,
                                                                                 enc_ker_layers=enc_ker_layers,
                                                                                 dec_ker_layers=dec_ker_layers,
                                                                                 num_heads_enc=num_heads_enc,
                                                                                 num_heads_dec=num_heads_dec,
                                                                                 baseblock_dec_ker=baseblock_dec_ker,
                                                                                 train_size=sub_image_size, flip_transpose=flip_transpose,
                                                                                 patch_size=sub_image_size, inp_feature=inp_feature, deconv_feature=deconv_feature,
                                                                                 save_memory=save_memory_kernel[i],
                                                                                 kernel_size=[255, 127, 63], k_len=61,
                                                                                 upx=False, ks=ks, scale=0, pad_size=pad_size,
                                                                                 first_col=first_col_ker, mask=deblur_mask, norm_type=norm_type, spatial_add=spatial_add,
                                                                                 frozen_kernel=False, spatial_decoder=spatial_decoder, get_ker=get_ker))  # , mask_type=None, Sigmoid, Softmax, GumbelSoftmax , norm='forward' [255, 127, 63]

                    elif self.deblur_kernel_mode == 'ker_max_mid':
                        if i > 0:
                            if self.small_scale:
                                self.down_mid_kerfeatures.append(DownSample(dims[1], dims[2], fuse_downsample_mode))
                        else:
                            self.down_mid_kerfeatures.append(nn.Identity())
                        self.kernel_estimator.append(
                            KernelEstimator(img_channel=dims[0] * 2, num_ker=num_ker,
                                                                                 width=dims_ker,
                                                                                 width_d=dims, downsample=downsample_ker,
                                                                                 baseblock_enc_ker=baseblock_enc_ker,
                                                                                 enc_ker_layers=enc_ker_layers,
                                                                                 dec_ker_layers=dec_ker_layers,
                                                                                 num_heads_enc=num_heads_enc,
                                                                                 num_heads_dec=num_heads_dec,
                                                                                 baseblock_dec_ker=baseblock_dec_ker,
                                                                                 train_size=sub_image_size,
                                                                                 patch_size=sub_image_size,
                                            overlap_size=overlap_size, pad_size=pad_size, flip_transpose=flip_transpose,
                                            inp_feature=inp_feature, deconv_feature=deconv_feature,
                                                                                 save_memory=save_memory_kernel[i],
                                                                                 kernel_size=[255, 127, 63], k_len=61,
                                                                                 upx=False, ks=ks, scale=0,
                                                                                 first_col=first_col_ker,mask=deblur_mask, norm_type=norm_type, spatial_add=spatial_add,
                                                                                 frozen_kernel=False, spatial_decoder=spatial_decoder, get_ker=get_ker))
                        self.mid_kernel_estimator.append(
                            KernelEstimator(img_channel=dims[1] * 2, flip_transpose=flip_transpose,
                                                                                 num_ker=num_ker,
                                                                                 width=dims_ker[1:],
                                                                                 width_d=dims[1:],
                                                                                 downsample=downsample_ker,
                                                                                 baseblock_enc_ker=baseblock_enc_ker,
                                                                                 enc_ker_layers=enc_ker_layers,
                                                                                 dec_ker_layers=dec_ker_layers,
                                                                                 num_heads_enc=num_heads_enc,
                                                                                 num_heads_dec=num_heads_dec,
                                                                                 baseblock_dec_ker=baseblock_dec_ker,
                                            train_size=[sub_image_size[0] // 2, sub_image_size[1] // 2],
                                            patch_size=[sub_image_size[0] // 2, sub_image_size[1] // 2],
                                            overlap_size=[overlap_size[0] // 2, overlap_size[1] // 2],
                                            inp_feature=inp_feature, deconv_feature=deconv_feature,
                                            save_memory=save_memory_kernel[i], pad_size=pad_size//2,
                                                                                 kernel_size=[255, 127, 63], k_len=61,
                                                                                 upx=False, ks=ks, scale=0,
                                                                                 first_col=first_col_ker,mask=deblur_mask, norm_type=norm_type, spatial_add=spatial_add,
                                                                                 frozen_kernel=False, spatial_decoder=spatial_decoder, get_ker=get_ker))
                    elif self.deblur_kernel_mode == 'ker_max_mid_small':

                        self.kernel_estimator.append(
                            KernelEstimator(img_channel=dims[0] * 2, num_ker=num_ker,
                                            width=dims_ker, flip_transpose=flip_transpose,
                                            width_d=dims, downsample=downsample_ker,
                                            baseblock_enc_ker=baseblock_enc_ker,
                                            enc_ker_layers=enc_ker_layers,
                                            dec_ker_layers=dec_ker_layers,
                                            num_heads_enc=num_heads_enc,
                                            num_heads_dec=num_heads_dec,
                                            baseblock_dec_ker=baseblock_dec_ker,
                                            train_size=sub_image_size,
                                            patch_size=sub_image_size,
                                            overlap_size=overlap_size, inp_feature=inp_feature,
                                            deconv_feature=deconv_feature,
                                            save_memory=save_memory_kernel[i],
                                            kernel_size=[255, 127, 63], k_len=61,
                                            upx=False, ks=ks, scale=0, pad_size=pad_size,
                                            first_col=first_col_ker, mask=deblur_mask, norm_type=norm_type,
                                            spatial_add=spatial_add,
                                            frozen_kernel=False, spatial_decoder=spatial_decoder, get_ker=get_ker))
                        self.mid_kernel_estimator.append(
                            KernelEstimator(img_channel=dims[1] * 2,
                                            num_ker=num_ker, flip_transpose=flip_transpose,
                                            width=dims_ker,
                                            width_d=dims,
                                            downsample=downsample_ker,
                                            baseblock_enc_ker=baseblock_enc_ker,
                                            enc_ker_layers=enc_ker_layers,
                                            dec_ker_layers=dec_ker_layers,
                                            num_heads_enc=num_heads_enc,
                                            num_heads_dec=num_heads_dec,
                                            baseblock_dec_ker=baseblock_dec_ker,
                                            train_size=[sub_image_size[0]//2, sub_image_size[1]//2],
                                            patch_size=[sub_image_size[0]//2, sub_image_size[1]//2],
                                            overlap_size=[overlap_size[0] // 2, overlap_size[1] // 2],
                                            inp_feature=inp_feature, deconv_feature=deconv_feature,
                                            save_memory=save_memory_kernel[i],
                                            kernel_size=[255, 127, 63], k_len=61,
                                            upx=False, ks=ks, scale=0, pad_size=pad_size//2,
                                            first_col=first_col_ker, mask=deblur_mask, norm_type=norm_type,
                                            spatial_add=spatial_add,
                                            frozen_kernel=False, spatial_decoder=spatial_decoder, get_ker=get_ker))
                        self.small_kernel_estimator.append(
                            KernelEstimator(img_channel=dims[1] * 2,
                                            num_ker=num_ker, flip_transpose=flip_transpose,
                                            width=dims_ker,
                                            width_d=dims,
                                            downsample=downsample_ker,
                                            baseblock_enc_ker=baseblock_enc_ker,
                                            enc_ker_layers=enc_ker_layers,
                                            dec_ker_layers=dec_ker_layers,
                                            num_heads_enc=num_heads_enc,
                                            num_heads_dec=num_heads_dec,
                                            baseblock_dec_ker=baseblock_dec_ker,
                                            train_size=[sub_image_size[0] // 4, sub_image_size[1] // 4],
                                            patch_size=[sub_image_size[0] // 4, sub_image_size[1] // 4],
                                            overlap_size=[overlap_size[0] // 4, overlap_size[1] // 4],
                                            inp_feature=inp_feature, deconv_feature=deconv_feature,
                                            save_memory=save_memory_kernel[i],
                                            kernel_size=[255, 127, 63], k_len=61,
                                            upx=False, ks=ks, scale=0, pad_size=pad_size//4,
                                            first_col=first_col_ker, mask=deblur_mask, norm_type=norm_type,
                                            spatial_add=spatial_add,
                                            frozen_kernel=False, spatial_decoder=spatial_decoder, get_ker=get_ker))
                self.patch_embed_maxs.append(OverlapPatchEmbed(inp_channels, dims[0]))
                if self.multiscale_enc:
                    self.patch_embed_maxs2.append(OverlapPatchEmbed(inp_channels, dims[0]))
                    self.patch_embed_mids2.append(OverlapPatchEmbed(inp_channels, dims[0]*2))
                    self.patch_embed_smalls2.append(OverlapPatchEmbed(inp_channels, dims[0]*4))
                    self.output_mids2.append(
                        nn.Conv2d(int(dims[0] * 2), 3, kernel_size=3, stride=1, padding=1,
                                  bias=bias)
                    )
                    self.output_smalls2.append(
                        nn.Conv2d(int(dims[0] * 4), 3, kernel_size=3, stride=1, padding=1,
                                  bias=bias)
                    )
                if self.small_scale:
                    self.patch_embed_smalls.append(OverlapPatchEmbed(inp_channels, dims[2]))
                if self.mid_scale:
                    self.patch_embed_mids.append(OverlapPatchEmbed(inp_channels, dims[1]))
                if i > 0:
                    self.patch_embed_maxs2.append(nn.Conv2d(dims[0] * 2, dims[0], 1))
                    if self.small_scale:
                        self.patch_embed_smalls2.append(nn.Conv2d(dims[1] * 2, dims[1], 1))
                    if self.mid_scale:
                        self.patch_embed_mids2.append(nn.Conv2d(dims[2] * 2, dims[2], 1))
                else:
                    self.patch_embed_maxs2.append(nn.Identity())
                    if self.small_scale:
                        self.patch_embed_smalls2.append(nn.Identity())
                    if self.mid_scale:
                        self.patch_embed_mids2.append(nn.Identity())
                first_col_dec = first_col_ if i == 0 else False
                # print(enc_max_layers)
                if self.multiscale_enc:
                    self.subnets_max.append(UniSimpleMEncoder3Level(
                        channels=channels_enc_max, layers=enc_max_layers[i], num_heads=num_heads_enc,
                        kernel_size=kernel_size,
                        first_col=first_col_enc_max, dp_rates=dp_rate_e, save_memory=save_memory_encoder,
                        baseblock=baseblock_enc_max[i], train_size=[train_size[0], train_size[1]],
                        patch_size=[patch_size[0], patch_size[1]], sub_idx=(i + 1), downsample=downsample, downsample_mode=downsample_mode))
                else:
                    self.subnets_max.append(UniSimpleEncoder3Level(
                        channels=channels_enc_max, layers=enc_max_layers[i], num_heads=num_heads_enc, kernel_size=kernel_size,
                        first_col=first_col_enc_max, dp_rates=dp_rate_e, save_memory=save_memory_encoder,
                        baseblock=baseblock_enc_max[i], train_size=[train_size[0], train_size[1]],
                        patch_size=[patch_size[0], patch_size[1]], sub_idx=(i + 1), downsample_mode=downsample_mode))
                # self.down_latent_max.append(nn.Sequential(Downsample2(dims[0] * 4, dims[0] * 8)))
                if self.mid_scale:
                    # self.down_latent_mid.append(nn.Sequential(Downsample2(dims[1] * 4, dims[1] * 8)))
                    if self.small_scale:
                        # self.down_latent_small.append(nn.Sequential(Downsample2(dims[2] * 4, dims[2] * 8)))
                        self.subdenets_small.append(UniSimpleDecoder3Level(
                            channels=channels_dec_small, layers=dec_small_layers[i], num_heads=num_heads_dec,
                            kernel_size=kernel_size,
                            first_col=first_col_dec, dp_rates=dp_rate_d, save_memory=save_memory_decoder,
                            baseblock=baseblock_dec_small[i], train_size=[train_size[0] // 4, train_size[1] // 4],
                            patch_size=[patch_size[0] // 4, patch_size[1] // 4], sub_idx=(i + 1), upsample_mode=upsample_mode))

                    if fuse_method_decoder in self.fuse_in_decoder and self.small_scale:
                        # self.up_mid0.append(nn.Conv2d(dims[2] * 4, dims[1] * 8, 1))
                        self.up_mid1.append(nn.Conv2d(dims[2] * 2, dims[1] * 4, 1))
                        self.up_mid2.append(nn.Conv2d(dims[2] * 1, dims[1] * 2, 1))
                        self.up_mid3.append(UpSample(dims[2] * 1, dims[1] * 1, fuse_upsample_mode))
                        self.subdenets_mid.append(UniSimpleMDecoder3Level(
                            channels=channels_dec_mid, layers=dec_mid_layers[i], num_heads=num_heads_dec,
                            kernel_size=kernel_size,
                            first_col=first_col_dec, dp_rates=dp_rate_d, save_memory=save_memory_decoder,
                            baseblock=baseblock_dec_mid[i], train_size=[train_size[0] // 2, train_size[1] // 2],
                            patch_size=[patch_size[0] // 2, patch_size[1] // 2], sub_idx=(i + 1), upsample_mode=upsample_mode))
                    else:
                        self.subdenets_mid.append(UniSimpleDecoder3Level(
                            channels=channels_dec_mid, layers=dec_mid_layers[i], num_heads=num_heads_dec,
                            kernel_size=kernel_size,
                            first_col=first_col_dec, dp_rates=dp_rate_d, save_memory=save_memory_decoder,
                            baseblock=baseblock_dec_mid[i], train_size=[train_size[0] // 2, train_size[1] // 2],
                            patch_size=[patch_size[0] // 2, patch_size[1] // 2], sub_idx=(i + 1), upsample_mode=upsample_mode))
                if fuse_method_decoder in self.fuse_in_decoder and self.mid_scale:
                    # self.up_max0.append(nn.Conv2d(dims[1] * 4, dims[0] * 8, 1))
                    self.up_max1.append(nn.Conv2d(dims[1] * 2, dims[0] * 4, 1))
                    self.up_max2.append(nn.Conv2d(dims[1] * 1, dims[0] * 2, 1))
                    self.up_max3.append(UpSample(dims[1] * 1, dims[0] * 1, fuse_upsample_mode))
                    self.subdenets_max.append(UniSimpleMDecoder3Level(
                        channels=channels_dec_max, layers=dec_max_layers[i], num_heads=num_heads_dec,
                        kernel_size=kernel_size,
                        first_col=first_col_dec, dp_rates=dp_rate_d, save_memory=save_memory_decoder,
                        baseblock=baseblock_dec_max[i], train_size=[train_size[0], train_size[1]],
                        patch_size=[patch_size[0], patch_size[1]], sub_idx=(i + 1), upsample_mode=upsample_mode))
                else:
                    self.subdenets_max.append(UniSimpleDecoder3Level(
                        channels=channels_dec_max, layers=dec_max_layers[i], num_heads=num_heads_dec,
                        kernel_size=kernel_size,
                        first_col=first_col_dec, dp_rates=dp_rate_d, save_memory=save_memory_decoder,
                        baseblock=baseblock_dec_max[i], train_size=[train_size[0], train_size[1]],
                        patch_size=[patch_size[0], patch_size[1]], sub_idx=(i + 1), upsample_mode=upsample_mode))

                if fuse_method_encoder in self.fuse_no_encoder or fuse_method_encoder in self.fuse_in_encoder:
                    first_col = first_col_ if i == 0 else False
                else:
                    first_col = False
                if self.mid_scale:
                    if fuse_method_encoder in self.fuse_in_encoder:
                        if self.small_scale:
                            self.down_mid1.append(DownSample(dims[1], dims[2], fuse_downsample_mode))
                            self.down_mid2.append(DownSample(dims[1] * 2, dims[2] * 2, fuse_downsample_mode))
                            self.down_mid3.append(DownSample(dims[1] * 4, dims[2] * 4, fuse_downsample_mode))
                            self.subnets_small.append(UniSimpleMEncoder3Level(
                                channels=channels_enc_small, layers=enc_small_layers[i], num_heads=num_heads_enc,
                                kernel_size=kernel_size,
                                first_col=first_col, dp_rates=dp_rate_e, save_memory=save_memory_encoder,
                                baseblock=baseblock_enc_small[i], train_size=[train_size[0] // 4, train_size[1] // 4],
                                patch_size=[patch_size[0] // 4, patch_size[1] // 4], sub_idx=(i + 1), downsample=downsample, downsample_mode=downsample_mode))
                        # print(fuse_downsample_mode)
                        self.down_max1.append(DownSample(dims[0], dims[1], fuse_downsample_mode))
                        self.down_max2.append(DownSample(dims[0]*2, dims[1]*2, fuse_downsample_mode))
                        self.down_max3.append(DownSample(dims[0]*4, dims[1]*4, fuse_downsample_mode))
                        # self.down_max4.append(Downsample2(dims[0] * 8, dims[1] * 8))
                        self.subnets_mid.append(UniSimpleMEncoder3Level(
                            channels=channels_enc_mid, layers=enc_mid_layers[i], num_heads=num_heads_enc,
                            kernel_size=kernel_size,
                            first_col=first_col, dp_rates=dp_rate_e, save_memory=save_memory_encoder,
                            baseblock=baseblock_enc_mid[i], train_size=[train_size[0] // 2, train_size[1] // 2],
                            patch_size=[patch_size[0] // 2, patch_size[1] // 2], sub_idx=(i + 1), downsample=downsample, downsample_mode=downsample_mode))
                    else:
                        if self.small_scale:
                            self.subnets_small.append(UniSimpleEncoder3Level(
                                channels=channels_enc_small, layers=enc_small_layers[i], num_heads=num_heads_enc,
                                kernel_size=kernel_size,
                                first_col=first_col, dp_rates=dp_rate_e, save_memory=save_memory_encoder,
                                baseblock=baseblock_enc_small[i], train_size=[train_size[0] // 4, train_size[1] // 4],
                                patch_size=[patch_size[0] // 4, patch_size[1] // 4], sub_idx=(i + 1), downsample_mode=downsample_mode))

                        self.subnets_mid.append(UniSimpleEncoder3Level(
                            channels=channels_enc_mid, layers=enc_mid_layers[i], num_heads=num_heads_enc,
                            kernel_size=kernel_size,
                            first_col=first_col, dp_rates=dp_rate_e, save_memory=save_memory_encoder,
                            baseblock=baseblock_enc_mid[i], train_size=[train_size[0] // 2, train_size[1] // 2],
                            patch_size=[patch_size[0] // 2, patch_size[1] // 2], sub_idx=(i + 1), downsample_mode=downsample_mode))

                    if self.fuse_method_encoder in self.fuse_before_encoder:
                        if self.small_scale:
                            self.aff_mid.append(nn.Conv2d(dims[2] * 2, dims[2], 1))
                        self.aff_max.append(nn.Conv2d(dims[1] * 2, dims[1], 1))
                        if self.fuse_method_encoder in ['DownSample2shallow', 'Fuse_in_Encoder2shallow']:
                            # self.max_down_l1.append(Downsample2(dims[0], dims[1]))
                            self.max_down_l3.append(UpSample(dims[0] * 4, dims[1], fuse_upsample_mode))
                            # self.mid_down_l1.append(Downsample2(dims[1], dims[2]))
                            if self.small_scale:
                                self.mid_down_l3.append(UpSample(dims[1] * 4, dims[2], fuse_upsample_mode))
                        # elif self.fuse_method_encoder == 'DownSample3shallow':
                        #     # self.max_down_l1.append(Downsample3(dims[0], dims[1]))
                        #     self.max_down_l3.append(Upsample3(dims[0] * 4, dims[1]))
                        #     # self.mid_down_l1.append(Downsample3(dims[1], dims[2]))
                        #     if self.small_scale:
                        #         self.mid_down_l3.append(Upsample3(dims[1] * 4, dims[2]))


                    if self.fuse_method_decoder in self.fuse_before_decoder:
                        if self.fuse_method_decoder in ['DownSample2latent', 'Fuse_in_Decoder2latent']:
                            if self.mid_scale:
                                if self.small_scale:
                                    self.down_small2mid.append(DownSample(dims[2], dims[1] * 4, fuse_downsample_mode))
                                self.down_mid2max.append(DownSample(dims[1], dims[0] * 4, fuse_downsample_mode))
                                # self.down_mid2max.append(DownSample(dims[1] * 1, dims[0] * 4, fuse_downsample_mode))
                        # 
                        # elif self.fuse_method_decoder == 'DownSample3latent':
                        #     if self.mid_scale:
                        #         if self.small_scale:
                        #             self.down_small2mid.append(Downsample3(dims[2] * 2, dims[1] * 4))
                        #         self.down_mid2max.append(Downsample3(dims[1] * 2, dims[0] * 4))
                        # 
                        # elif self.fuse_method_decoder == 'DownSample4latent':
                        #     if self.mid_scale:
                        #         if self.small_scale:
                        #             self.down_small2mid.append(Downsample4(dims[2] * 2, dims[1] * 4))
                        # 
                        #         self.down_mid2max.append(Downsample4(dims[1] * 2, dims[0] * 4))
                        if self.small_scale:
                            self.fusd_mid_2.append(nn.Conv2d(dims[1]*8, dims[1]*4, 1))
                        if self.mid_scale:
                            self.fusd_max_2.append(nn.Conv2d(dims[0]*8, dims[0]*4, 1))

                out_c = out_channels  # num_ker*2 # + out_channels


                if self.small_scale:
                    self.output_smalls.append(
                        nn.Conv2d(int(dims[2] * 1 ** 1), 3, kernel_size=3, stride=1, padding=1, bias=bias)
                    )
                    # self.output_smalls2.append(
                    #     nn.Conv2d(int(dims[2]), out_c, kernel_size=3, stride=1, padding=1, bias=bias)
                    # )
                if self.mid_scale:
                    self.output_mids.append(
                        nn.Conv2d(int(dims[1] * 1 ** 1), 3, kernel_size=3, stride=1, padding=1, bias=bias)
                    )
                    # self.output_mids2.append(
                    #     nn.Conv2d(int(dims[1]), out_c, kernel_size=3, stride=1, padding=1, bias=bias)
                    # )
                if self.deblur_kernel and not self.latent_sharp:
                    self.output_maxs.append(
                        nn.Identity()
                    )
                else:
                    self.output_maxs.append(
                        nn.Conv2d(int(dims[0] * 1 ** 1), 3, kernel_size=3, stride=1, padding=1,
                                  bias=bias)
                    )
                # self.output_maxs2.append(
                #     nn.Conv2d(int(dims[0]), out_c, kernel_size=3, stride=1, padding=1,
                #               bias=bias)
                # )
        if self.up2scale:
            deblur_loss_weight = deblur_loss_weight * 0.5
        self.loss_L1 = L1Loss(loss_weight=1.)
        self.loss_FFT = FourierLoss(loss_weight=deblur_loss_weight)
        self.i = 1
        self.dims = dims
    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign

    def _fuse_encoder_max2mid_feature(self, mid_features, latent_max2mid, i=0):
        inp_enc_level1_mid, out_enc_level1_mid, out_enc_level2_mid, latent_mid = mid_features
        if self.fuse_method_encoder in self.fuse_before_encoder:
            latent_max2mid = self.max_down_l3[i](latent_max2mid)
            if self.mid_flip:
                latent_max2mid = torch.flip(latent_max2mid, [-2, -1])
            inp_enc_level1_mid = self.aff_max[i](torch.cat([latent_max2mid, inp_enc_level1_mid], dim=1)) # inp_enc_level1_max2mid1 * inp_enc_level1_mid + inp_enc_level1_max2mid2
        return inp_enc_level1_mid, out_enc_level1_mid, out_enc_level2_mid, latent_mid

    def _fuse_encoder_mid2small_feature(self, small_features, latent_mid2small, i=0):
        inp_enc_level1_small, out_enc_level1_small, out_enc_level2_small, latent_small = small_features
        if self.fuse_method_encoder in self.fuse_before_encoder:
            latent_mid2small = self.mid_down_l3[i](latent_mid2small)
            if self.mid_flip:
                latent_mid2small = torch.flip(latent_mid2small, [-2, -1])
            inp_enc_level1_small = self.aff_mid[i](torch.cat([latent_mid2small, inp_enc_level1_small], dim=1))
        return inp_enc_level1_small, out_enc_level1_small, out_enc_level2_small, latent_small
    def _fuse_decoder_mid2max_feature(self, mid_features, max_features, i=0):
        out_enc_level1_mid2max, out_dec_level1_mid2max = mid_features
        out_enc_level1_max, out_enc_level2_max, out_enc_level3_max = max_features
        if self.fuse_method_decoder in self.fuse_before_decoder:
            # latent_mid2max = torch.cat([out_enc_level1_mid2max, out_dec_level1_mid2max], dim=1)
            latent_mid2max = out_dec_level1_mid2max
            latent_mid2max = self.down_mid2max[i](latent_mid2max)
            if self.mid_flip:
                latent_mid2max = torch.flip(latent_mid2max, [-2, -1])
            out_enc_level3_max = self.fusd_max_2[i](torch.cat([latent_mid2max, out_enc_level3_max], dim=1))
        return out_enc_level1_max, out_enc_level2_max, out_enc_level3_max

    def _fuse_decoder_small2mid_feature(self, small_features, mid_features, i=0):
        out_enc_level1_mid, out_enc_level2_mid, latent_mid = mid_features
        out_enc_level1_small2mid, out_dec_level1_small2mid = small_features
        if self.fuse_method_decoder in self.fuse_before_decoder:
            # latent_small2mid = torch.cat([out_enc_level1_small2mid, out_dec_level1_small2mid], dim=1)
            latent_small2mid = out_dec_level1_small2mid
            latent_small2mid = self.down_small2mid[i](latent_small2mid)
            if self.mid_flip:
                latent_small2mid = torch.flip(latent_small2mid, [-2, -1])
            # self._clamp_abs(self.fusd_mid_2[i].data, 1e-3)
            # latent_mid = latent_small2mid + latent_mid * self.fusd_mid_2[i]
            latent_mid = self.fusd_mid_2[i](torch.cat([latent_small2mid, latent_mid], dim=1))
            # latent_mid = latent_small2mid + latent_mid
        return out_enc_level1_mid, out_enc_level2_mid, latent_mid

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x

    def forward_kernel_max(self, feature_max, inp_blur, img_deblur, ker_feature, spatial_feature, k_idx):

        # if self.deblur_kernel_mode == 'ker_max':
        # x_feature = [feature_max[-1], feature_max[-3], feature_max[-2]]
        x_feature = [feature_max[-1], feature_max[0], feature_max[1]]
        outdict = self.kernel_estimator[k_idx](x_feature, inp_blur,  deblur_img=img_deblur,
                                                                           ker_feature=ker_feature, spatial_feature=spatial_feature)

            # ker_feature = outdict_max['ker_feature']
            #
            # kernel_feature = [outdict_max['feature_deblur'], outdict_mid['feature_deblur'],
            #                   outdict_small['feature_deblur']]
        return outdict
    def forward_kernel_mid(self, feature_mid, inp_blur, img_deblur, ker_feature, spatial_feature, k_idx):

        # if self.deblur_kernel_mode == 'ker_max_mid':
        x_feature = [feature_mid[-1], feature_mid[-3], feature_mid[-2]]
        outdict = self.mid_kernel_estimator[k_idx](x_feature, inp_blur,  deblur_img=img_deblur, ker_feature=ker_feature, spatial_feature=spatial_feature)

            # ker_feature = outdict_max['ker_feature']
            #
            # kernel_feature = [outdict_max['feature_deblur'], outdict_mid['feature_deblur'],
            #                   outdict_small['feature_deblur']]
        return outdict
    def forward_kernel_small(self, feature_small, inp_blur, img_deblur, ker_feature, spatial_feature, k_idx):

        # if self.deblur_kernel_mode == 'ker_max_mid':
        x_feature = [feature_small[-1], feature_small[-3], feature_small[-2]]
        outdict = self.small_kernel_estimator[k_idx](x_feature, inp_blur,  deblur_img=img_deblur, ker_feature=ker_feature, spatial_feature=spatial_feature)

            # ker_feature = outdict_max['ker_feature']
            #
            # kernel_feature = [outdict_max['feature_deblur'], outdict_mid['feature_deblur'],
            #                   outdict_small['feature_deblur']]
        return outdict
    # def forward_(self, inp_img, gt=None):
    def forward(self, inps_pry, gts=None, feature_prior=None):
        outputs = list()
        # if self.deblur_kernel:
        ker_feature_max, ker_feature_mid, ker_feature_small = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
        spatial_feature_max, spatial_feature_mid, spatial_feature_small = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]

        kernel_feature_max, kernel_feature_mid, kernel_feature_small = None, None, None
        out_img_level1_small = None
        out_img_level1_mid = None
        outputs_mid = list()
        outputs_small = list()
        feature_visuals = list()
        max_spatial_feature_list = list()
        max_fourier_feature_list = list()
        mid_spatial_feature_list = list()
        mid_fourier_feature_list = list()
        # if self.i:
        #     os.environ['MASTER_ADDR'] = 'localhost'
        #     os.environ['MASTER_PORT'] = '12345'
        #     rank, world_size = get_dist_info()
        #     dist.init_process_group('nccl', rank=rank, world_size=world_size)
        #     self.kernel_estimator = DDP(self.kernel_estimator)
        #     self.i = 0
        # self.kernel_estimator.to('cuda:1')
        if gts is not None:
            gt = gts[0]
            if self.up2scale:
                gt_mid = gt
                gt_small = gt
            else:
                # gts = kornia.geometry.transform.build_pyramid(gt, 3)
                gt_mid = gts[1]  # [..., self.ks[1]:-self.ks[1], self.ks[1]:-self.ks[1]]
                gt_small = gts[2]  # [..., self.ks[2]:-self.ks[2], self.ks[2]:-self.ks[2]]
        else:
            gt = None
            gt_mid = None
            gt_small = None

        deblur_loss = 0.

        out_dict = {}
        kernels = []
        # inp_img = inps_pry[0]
        inp_img_max = inps_pry[0]
        # print(inp_img_max.shape)
        # inps_pry = kornia.geometry.transform.build_pyramid(inp_img, 3)

        inp_img_mid = inps_pry[1]  # [..., self.ks[1]:-self.ks[1], self.ks[1]:-self.ks[1]]
        inp_img_small = inps_pry[2]  # [..., self.ks[2]:-self.ks[2], self.ks[2]:-self.ks[2]]
        if self.mid_flip:
            inp_img_mid = torch.flip(inp_img_mid, [-2, -1])

        out_enc_level1_small, out_enc_level2_small, latent_small, out_dec_level1_small = 0, 0, 0, 0 # feature_small
        out_enc_level1_mid, out_enc_level2_mid, latent_mid, out_dec_level1_mid = 0, 0, 0, 0
        out_enc_level1_max, out_enc_level2_max, latent_max, out_dec_level1_max = 0, 0, 0, 0
        if self.prior_model is not None:
            out_enc_level1_max, out_enc_level2_max, latent_max = feature_prior
            # print(feature_prior)
        if self.up2scale:
            inp_img_mid_ = inp_img_max
            inp_img_small_ = inp_img_max
        else:
            inp_img_mid_ = inp_img_mid
            inp_img_small_ = inp_img_small
        for i in range(self.num_global_unet):

            inp_enc_level1_max = self.patch_embed_maxs[i](inp_img_max)
            if i > 0:
                if self.deblur_kernel:
                    # print(kernel_feature[0].shape)
                    inp_enc_level1_max = self.patch_embed_maxs2[i](
                        torch.cat([inp_enc_level1_max, kernel_feature_max], dim=1))
                else:
                    inp_enc_level1_max = self.patch_embed_maxs2[i](
                        torch.cat([inp_enc_level1_max, out_dec_level1_max], dim=1))
            if self.multiscale_enc:
                u0 = self.patch_embed_maxs2[i](inp_img_max)
                u1 = self.patch_embed_mids2[i](inp_img_mid)
                u2 = self.patch_embed_smalls2[i](inp_img_small)
                out_enc_level1_max, out_enc_level2_max, latent_max = self.subnets_max[i](inp_enc_level1_max,
                                                                                         out_enc_level1_max,
                                                                                         out_enc_level2_max,
                                                                                         latent_max,
                                                                                      u0, u1, u2)
            else:
                out_enc_level1_max, out_enc_level2_max, latent_max = self.subnets_max[i](inp_enc_level1_max,
                                                                                     out_enc_level1_max,
                                                                                     out_enc_level2_max,
                                                                                     latent_max)
            if self.get_ker:
                max_spatial_feature_list.append([inp_enc_level1_max, out_enc_level1_max, out_enc_level2_max, latent_max])
            if self.mid_scale:
                inp_enc_level1_mid = self.patch_embed_mids[i](inp_img_mid)
                if i > 0:
                    if self.deblur_kernel:
                        if self.deblur_kernel_mode == 'ker_max':
                            kernel_feature_mid = self.down_max_kerfeatures[i](kernel_feature_max)
                            # print(kernel_feature_mid.shape)
                            inp_enc_level1_mid = self.patch_embed_mids2[i](
                                torch.cat([inp_enc_level1_mid, kernel_feature_mid], dim=1))
                        elif self.deblur_kernel_mode == 'ker_max_mid':
                            # kernel_feature_mid = kernel_feature[1]
                            # print(kernel_feature_mid.shape)
                            inp_enc_level1_mid = self.patch_embed_mids2[i](
                                torch.cat([inp_enc_level1_mid, kernel_feature_mid], dim=1))

                    else:
                        inp_enc_level1_mid = self.patch_embed_mids2[i](
                            torch.cat([inp_enc_level1_mid, out_dec_level1_mid], dim=1))
                inp_enc_level1_mid, out_enc_level1_mid, out_enc_level2_mid, latent_mid = self._fuse_encoder_max2mid_feature(
                    [inp_enc_level1_mid, out_enc_level1_mid, out_enc_level2_mid, latent_mid],
                    latent_max, i)
                if self.fuse_method_encoder in self.fuse_in_encoder:
                    # print(self.fuse_method_encoder)
                    # b
                    feature_1 = self.down_max1[i](out_enc_level1_max)
                    feature_2 = self.down_max2[i](out_enc_level2_max)
                    feature_3 = self.down_max3[i](latent_max)
                    # feature_4 = self.down_max4[i](latent_max)
                    out_enc_level1_mid, out_enc_level2_mid, latent_mid = self.subnets_mid[i](inp_enc_level1_mid,
                                                                                             out_enc_level1_mid,
                                                                                             out_enc_level2_mid,
                                                                                             latent_mid,
                                                                                             feature_1,
                                                                                             feature_2,
                                                                                             feature_3
                                                                                             )
                else:
                    out_enc_level1_mid, out_enc_level2_mid, latent_mid = self.subnets_mid[i](inp_enc_level1_mid,
                                                                                             out_enc_level1_mid,
                                                                                             out_enc_level2_mid,
                                                                                             latent_mid)
                if self.get_ker:
                    mid_spatial_feature_list.append([inp_enc_level1_mid, out_enc_level1_mid, out_enc_level2_mid, latent_mid])
                if self.small_scale:

                    inp_enc_level1_small = self.patch_embed_smalls[i](inp_img_small)
                    if i > 0:
                        if self.deblur_kernel_mode == 'ker_max' or self.deblur_kernel_mode == 'ker_max_mid':
                            kernel_feature_small = self.down_mid_kerfeatures[i](kernel_feature_mid)
                            # print(kernel_feature_mid.shape)
                            inp_enc_level1_small = self.patch_embed_smalls2[i](
                                torch.cat([inp_enc_level1_small, kernel_feature_small], dim=1))
                        elif self.deblur_kernel_mode == 'ker_max_mid_small':
                            # kernel_feature_mid = kernel_feature[1]
                            # print(kernel_feature_mid.shape)
                            inp_enc_level1_small = self.patch_embed_smalls2[i](
                                torch.cat([inp_enc_level1_mid, kernel_feature_small], dim=1))

                        else:
                            inp_enc_level1_small = self.patch_embed_mids2[i](
                                torch.cat([inp_enc_level1_small, out_dec_level1_small], dim=1))
                    # if self.deblur_kernel:
                    #     inp_enc_level1_small = self.patch_embed_smalls2[i](
                    #         torch.cat([inp_enc_level1_small, kernel_feature[2]], dim=1))
                    inp_enc_level1_small, out_enc_level1_small, out_enc_level2_small, latent_small = self._fuse_encoder_mid2small_feature(
                        [inp_enc_level1_small, out_enc_level1_small, out_enc_level2_small, latent_small],
                        latent_mid, i)

                    if self.fuse_method_encoder in self.fuse_in_encoder:
                        feature_1 = self.down_mid1[i](out_enc_level1_mid)
                        feature_2 = self.down_mid2[i](out_enc_level2_mid)
                        feature_3 = self.down_mid3[i](latent_mid)
                        out_enc_level1_small, out_enc_level2_small, latent_small = self.subnets_small[i](
                                                                                            inp_enc_level1_small,
                                                                                            out_enc_level1_small,
                                                                                            out_enc_level2_small,
                                                                                            latent_small,
                                                                                             feature_1,
                                                                                             feature_2,
                                                                                             feature_3
                                                                                                )
                    else:
                        out_enc_level1_small, out_enc_level2_small, latent_small = self.subnets_small[i](
                            inp_enc_level1_small,
                            out_enc_level1_small,
                            out_enc_level2_small,
                            latent_small)

                    out_enc_level2_small, out_enc_level1_small, out_dec_level1_small = self.subdenets_small[i](
                        latent_small,
                        out_enc_level2_small,
                        out_enc_level1_small,
                        out_dec_level1_small)

                    # out_img_level1_small = self.output_smalls[i](out_dec_level1_small) + outdict_small['img_deblur'] # inp_img_small_
                    out_img_level1_small = self.output_smalls[i](out_dec_level1_small) + inp_img_small_
                    outputs_small.append(out_img_level1_small)
                    # if gt is not None:
                    #     deblur_loss += self.loss_L1(out_img_level1_small, gt_small) + self.loss_FFT(out_img_level1_small, gt_small)
                    if self.deblur_kernel and self.deblur_kernel_mode == 'ker_max_mid_small':
                        feature_small = [out_enc_level1_small, out_enc_level2_small, latent_small, out_dec_level1_small]
                        outdict_small = self.forward_kernel_small(feature_small, inp_img_small, out_img_level1_small,
                                                              ker_feature_small, spatial_feature_small, k_idx=i)
                        # out_img_level1_mid = self.output_mids[i](out_dec_level1_mid) + outdict_mid['img_deblur']
                        outputs_small.append(outdict_small['img_deblur'])
                        ker_feature_small = outdict_small['ker_feature']
                        kernel_feature_small = outdict_small['feature_deblur']
                        spatial_feature_small = outdict_small['spatial_feature']

                        out_enc_level1_mid, out_enc_level2_mid, latent_mid = self._fuse_decoder_small2mid_feature(
                            [out_enc_level1_small, kernel_feature_small],
                            [out_enc_level1_mid, out_enc_level2_mid, latent_mid], i)
                    else:
                        out_enc_level1_mid, out_enc_level2_mid, latent_mid = self._fuse_decoder_small2mid_feature(
                            [out_enc_level1_small, out_dec_level1_small],
                            [out_enc_level1_mid, out_enc_level2_mid, latent_mid], i)
                # if H >= self.scale_size[1]:

                # out_enc_level1_latent2mid = self.up_latent_mid[i](latent_mid)
                # print(latent_mid.shape)
                if self.fuse_method_decoder in self.fuse_in_decoder and self.small_scale:
                    # e
                    # feature_0 = self.up_mid0[i](latent_small)
                    feature_1 = self.up_mid1[i](out_enc_level2_small)
                    feature_2 = self.up_mid2[i](out_enc_level1_small)
                    feature_3 = self.up_mid3[i](out_dec_level1_small)
                    out_enc_level2_mid, out_enc_level1_mid, out_dec_level1_mid = self.subdenets_mid[i](latent_mid,
                                                                                                       out_enc_level2_mid,
                                                                                                       out_enc_level1_mid,
                                                                                                       out_dec_level1_mid,
                                                                                                       # feature_0,
                                                                                                       feature_1,
                                                                                                       feature_2,
                                                                                                       feature_3
                                                                                                       )
                else:
                    out_enc_level2_mid, out_enc_level1_mid, out_dec_level1_mid = self.subdenets_mid[i](latent_mid,
                                                                                                       out_enc_level2_mid,
                                                                                                       out_enc_level1_mid,
                                                                                                       out_dec_level1_mid)
                out_img_level1_mid = self.output_mids[i](out_dec_level1_mid) + inp_img_mid_
                outputs_mid.append(out_img_level1_mid)

                if self.deblur_kernel and (self.deblur_kernel_mode == 'ker_max_mid' or self.deblur_kernel_mode == 'ker_max_mid_small'):
                    feature_mid = [out_enc_level1_mid, out_enc_level2_mid, latent_mid, out_dec_level1_mid]
                    outdict_mid = self.forward_kernel_mid(feature_mid, inp_img_mid, out_img_level1_mid, ker_feature_mid, spatial_feature_mid, k_idx=i)
                    # out_img_level1_mid = self.output_mids[i](out_dec_level1_mid) + outdict_mid['img_deblur']
                    outputs_mid.append(outdict_mid['img_deblur'])
                    ker_feature_mid = outdict_mid['ker_feature']
                    kernel_feature_mid = outdict_mid['feature_deblur']
                    spatial_feature_mid = outdict_mid['spatial_feature']
                    if self.get_ker:
                        mid_spatial_feature_list[-1].extend([latent_mid, out_enc_level2_mid, out_enc_level1_mid, out_dec_level1_mid, kernel_feature_mid])
                        mid_fourier_feature_list.append(outdict_mid['visual_feature'])
                    out_enc_level1_max, out_enc_level2_max, latent_max = self._fuse_decoder_mid2max_feature(
                        [out_dec_level1_mid, kernel_feature_mid],
                        [out_enc_level1_max, out_enc_level2_max, latent_max], i)
                else:
                    out_enc_level1_max, out_enc_level2_max, latent_max = self._fuse_decoder_mid2max_feature(
                        [out_enc_level1_mid, out_dec_level1_mid],
                        [out_enc_level1_max, out_enc_level2_max, latent_max], i)

            # out_enc_level1_max_z = out_enc_level1_max
            if self.fuse_method_decoder in self.fuse_in_decoder and self.mid_scale:
                # feature_0 = self.up_max0[i](latent_mid)
                feature_1 = self.up_max1[i](out_enc_level2_mid)
                feature_2 = self.up_max2[i](out_enc_level1_mid)
                feature_3 = self.up_max3[i](out_dec_level1_mid)
                out_enc_level2_max, out_enc_level1_max, out_dec_level1_max = self.subdenets_max[i](latent_max,
                                                                                                   out_enc_level2_max,
                                                                                                   out_enc_level1_max,
                                                                                                   out_dec_level1_max,
                                                                                                       feature_1,
                                                                                                       feature_2,
                                                                                                       feature_3)
            else:
                out_enc_level2_max, out_enc_level1_max, out_dec_level1_max = self.subdenets_max[i](latent_max,
                                                                                                   out_enc_level2_max,
                                                                                                   out_enc_level1_max,
                                                                                                   out_dec_level1_max)
            if self.deblur_kernel and not self.latent_sharp:
                out_img_level1_max = None

            else:
                out_img_level1_max = self.output_maxs[i](out_dec_level1_max) + inp_img_max
                outputs.append(out_img_level1_max)


            if self.multiscale_enc:
                out_img_level1_mid = self.output_mids2[i](out_enc_level2_max) + inp_img_mid
                out_img_level1_small = self.output_smalls2[i](latent_max) + inp_img_small
                if gt is not None:
                    deblur_loss += self.loss_L1(out_img_level1_mid, gt_mid) + self.loss_FFT(out_img_level1_mid, gt_mid)
                    deblur_loss += self.loss_L1(out_img_level1_small, gt_small) + self.loss_FFT(out_img_level1_small, gt_small)
            if self.deblur_kernel:

                # img_deblur = out_img_level1_max # [out_img_level1_max, out_img_level1_mid, out_img_level1_small]
                # print(self.num_kernel_est, self.num_local_unet)
                # out_dict['feature_small'] = [out_enc_level1_small, out_enc_level2_small, latent_small, out_dec_level1_small]
                # feature_mid = [out_enc_level1_mid, out_enc_level2_mid, latent_mid, out_dec_level1_mid]
                feature_max = [out_enc_level1_max, out_enc_level2_max, latent_max, out_dec_level1_max]
                outdict_max = self.forward_kernel_max(feature_max, inp_img_max, out_img_level1_max, ker_feature_max, spatial_feature_max, k_idx=i)
                if self.get_ker:
                    feature_visuals.append(outdict_max['feature_visual'])
                outputs.append(outdict_max['img_deblur'])
                # print(outdict_max.keys())
                # out_dict['kernel-max'] = outdict_max['kernel_deblur']
                # if self.training:
                #     deblur_loss += outdict_max['loss_deblur'] + outdict_mid['loss_deblur'] + outdict_small[
                #         'loss_deblur']
                    # if outdict_max['loss_reblur'] is not None:
                    #     loss_reblur += outdict_max['loss_reblur'] + outdict_mid['loss_reblur'] + outdict_small[
                    #         'loss_reblur']
                    #     out_dict['loss_reblur'] = loss_reblur
                ker_feature_max = outdict_max['ker_feature']
                spatial_feature_max = outdict_max['spatial_feature']
                kernel_feature_max = outdict_max['feature_deblur']
                # kernel_feature = [outdict_max['feature_deblur']]
            else:
                kernel_feature_max, kernel_feature_mid = None, None
            if self.get_ker:
                max_spatial_feature_list[-1].extend([latent_max, out_enc_level2_max, out_enc_level1_max, out_dec_level1_max, kernel_feature_max])
                max_fourier_feature_list.append(outdict_max['visual_feature'])
        # out_dict['feature_small'] = [out_enc_level1_small, out_enc_level2_small, latent_small, out_dec_level1_small]
        # print(len(outputs_mid))
        if self.mid_scale and gt is not None:
            deblur_loss += self.loss_L1(outputs_mid, gt_mid) + self.loss_FFT(outputs_mid, gt_mid)
            if self.small_scale:
                deblur_loss += self.loss_L1(outputs_small, gt_small) + self.loss_FFT(outputs_small, gt_small)
        out_dict['feature_mid'] = [out_enc_level1_mid, out_enc_level2_mid, latent_mid, out_dec_level1_mid, kernel_feature_mid]
        # out_dict['feature_max'] = [ker_feature_max, out_enc_level2_max, latent_max, out_dec_level1_max]
        out_dict['feature_max'] = [out_enc_level1_max, out_enc_level2_max, latent_max, out_dec_level1_max, kernel_feature_max]
        # out_dict['deblur_small'] = out_img_level1_small
        out_dict['deblur_mid'] = out_img_level1_mid
        out_dict['deblur_max'] = outputs[-1]
        if self.get_ker:
            out_dict['max_spatial_feature_list'] = max_spatial_feature_list
            out_dict['mid_spatial_feature_list'] = mid_spatial_feature_list
            out_dict['max_fourier_feature_list'] = max_fourier_feature_list
            out_dict['mid_fourier_feature_list'] = mid_fourier_feature_list
            out_dict['feature_visual'] = feature_visuals
        out_dict['ker_feature'] = [ker_feature_max, ker_feature_mid]
        out_dict['spatial_feature'] = [spatial_feature_max, spatial_feature_mid]
        # print(len(out_dict['ker_feature']))
        # out_dict['loss_param'] = param_loss
        # if self.training and (self.small_scale or self.mid_scale):
        #     out_dict['img'] = outputs
        #
        #     out_dict['loss_deblur'] = deblur_loss # / self.num_global_unet
        #
        # else:
        out_dict['img'] = outputs
        out_dict['loss_deblur'] = deblur_loss
        return out_dict # [::-1]

class DeepRFTv2(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dims=[48, 48, 48],
                 sub_dec_width =64,
                 fuse_method_encoder='DownSampleAdd',
                 fuse_method_decoder='UpSampleAdd',
                 downsample='downsample',
                 downsample_ker='noconcat_nodownsmaple',
                 upsample_mode='conv', downsample_mode='conv',
                 fuse_upsample_mode='conv', fuse_downsample_mode='conv',
                 downsample_local='None',
                 deblur_kernel_mode='ker_max', inp_feature='inp_feature', deconv_feature='feature',
                 kernel_size=65, latent_sharp=True, spatial_add=False, spatial_decoder=False, get_ker=False,
                 num_heads_enc=[1, 2, 4],
                 num_heads_dec=[8, 4, 2, 1],
                 baseblock_enc_small_global=['naf', 'naf', 'naf'],
                 baseblock_dec_small_global=['naf', 'naf', 'naf', 'naf'],
                 baseblock_enc_mid_global=['naf', 'naf', 'naf', 'naf'],
                 baseblock_dec_mid_global=['naf', 'naf', 'naf', 'naf'],
                 baseblock_enc_max_global=['naf', 'naf', 'naf', 'naf'],
                 baseblock_dec_max_global=['naf', 'naf', 'naf', 'naf'],
                 baseblock_enc_small_local=['naf', 'naf', 'naf'],
                 baseblock_dec_small_local=['naf', 'naf', 'naf'],
                 baseblock_enc_mid_local=['naf', 'naf', 'naf'],
                 baseblock_dec_mid_local=['naf', 'naf', 'naf'],
                 baseblock_enc_max_local=['naf', 'naf', 'naf'],
                 baseblock_dec_max_local=['naf', 'naf', 'naf'],
                 baseblock_enc_ker=['naf', 'naf', 'naf'],
                 baseblock_dec_ker=['naf', 'naf', 'naf'],
                 baseblock_enc_local=['fcnaf', 'fcnaf', 'fcnaf', 'fcnaf'],
                 baseblock_dec_local=['fcnaf', 'fcnaf', 'fcnaf', 'fcnaf'],
                 enc_small_layers=[3, 6, 8], # 64 32 16
                 dec_small_layers=[8, 8, 6, 3],
                 enc_mid_layers=[2, 3, 4], # 128 64 32
                 dec_mid_layers=[4, 4, 3, 2],
                 enc_max_layers=[1, 2, 3], # 256 128 64
                 dec_max_layers=[3, 3, 2, 1],
                 enc_ker_layers=[1, 2, 3],  # 256 128 64
                 dec_ker_layers=[3, 2, 1],
                 enc_light_layers=[1, 1, 1],
                 enc_local_layers=[1, 1, 1],
                 dec_local_layers=[1, 1, 1, 1],
                 num_global_unet=1,
                 num_kernel_est=0,
                 num_local_unet=0,
                 select_range=None,
                 num_local_decoder=0,
                 # decoder_layers=[1, 1, 1, 1, 1],
                 sub_image_size=[256, 256],
                 train_size=[256, 256],
                 patch_size=[256, 256],
                 overlap_size=[48, 48],
                 drop_path=0.,
                 save_memory=True,
                 save_memory_global_encoder=[False, False, False],
                 save_memory_global_decoder=[False, False, False],
                 save_memory_local_encoder=[False, False, False],
                 save_memory_local_decoder=[False, False, False],
                 save_memory_ker=[False, False, False],
                 save_memory_local_ker=[False, False, False],
                 dynamic_dec=False,
                 mid_flip=False,
                 deblur_kernel=False,
                 num_ker=4,
                 pad_size=32,
                 dims_ker=[48, 48, 48],
                 inter_out=False,
                 inter_ker=False,
                 small_enc=False,
                 ratio_loss_weight=0.5,
                 ratio=0.75,
                 frozen_encoder=False,
                 up2scale=False,
                 exit_thr=0.1,
                 small_scale=True,
                 mid_scale=True,
                 sub_image=False,
                 deblur_mask=False,
                 ks=[0, 0, 0],
                 scale_size=[256, 128, 64],
                 frozen_global=False,
                 frozen_kernel=False,
                 sub_image_pad=False,
                 multiscale_enc=False,
                 prior_model=None,
                 flip_transpose=False, norm_type='None', deblur_loss_weight=0.01
                 ):
        super(DeepRFTv2, self).__init__()
        shortcut_scale_init_value = 0.5
        self.sub_image = sub_image
        self.h = -1
        self.w = -1
        self.flip_transpose = flip_transpose
        self.deblur_kernel = deblur_kernel
        self.pad_size = pad_size
        self.prior_model = prior_model
        self.num_ker = num_ker
        self.ks = ks
        self.multiscale_enc = multiscale_enc
        self.frozen_global = frozen_global
        self.frozen_kernel = frozen_kernel
        self.num_kernel_est = num_kernel_est
        self.overlap_size = overlap_size
        # self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
        # self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size[1]).view(1, 1, self.overlap_size[1])
        # self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
        # self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size[0]).view(1, self.overlap_size[0], 1)
        self.ker_size = [int(sub_image_size[0] * 1.), int(sub_image_size[1] * 1.)]
        self.up_scale = 1
        self.overlap_size_up = (self.overlap_size[0] * self.up_scale, self.overlap_size[1] * self.up_scale)
        # self.kernel_size = [train_size, train_size]
        self.kernel_size_up = [self.ker_size[0] * self.up_scale, self.ker_size[1] * self.up_scale]
        self.out_channels = out_channels
        # print()
        self.dynic_overlap = False
        self.scale_size = scale_size
        self.mid_scale = mid_scale
        self.small_scale = small_scale
        self.small_enc = small_enc
        self.exit_thr = exit_thr
        self.up2scale = up2scale
        self.frozen_encoder = frozen_encoder
        self.ratio = ratio
        self.mid_flip = mid_flip
        self.ratio_loss_weight = ratio_loss_weight
        self.dynamic_dec = dynamic_dec
        self.num_global_unet = num_global_unet
        self.num_local_unet = num_local_unet
        self.inter_out = inter_out
        self.kernel_size = kernel_size
        self.inter_ker = inter_ker
        self.sub_image_pad = sub_image_pad
        self.get_ker = get_ker
        self.kernel_estimator = nn.ModuleList()
        self.mid_kernel_estimator = nn.ModuleList()
        # self.local_model = nn.ModuleList()

        if self.num_global_unet > 0:
            # print(spatial_add, spatial_decoder)
            save_memory_global = save_memory if not self.frozen_global else False
            self.global_model=BaseGoKModel(inp_channels=inp_channels, out_channels=out_channels, dims=dims, num_ker=num_ker, inp_feature=inp_feature, deconv_feature=deconv_feature,
                                         fuse_method_encoder=fuse_method_encoder, fuse_method_decoder=fuse_method_decoder,
                                         downsample=downsample, upsample_mode=upsample_mode, downsample_mode=downsample_mode,
                                         fuse_upsample_mode=fuse_upsample_mode, fuse_downsample_mode=fuse_downsample_mode, multiscale_enc=multiscale_enc,
                                         num_heads_enc=num_heads_enc,  num_heads_dec=num_heads_dec, latent_sharp=latent_sharp,
                                         baseblock_enc_small=baseblock_enc_small_global, save_memory_global_encoder=save_memory_global_encoder, save_memory_global_decoder=save_memory_global_decoder,
                                         baseblock_dec_small=baseblock_dec_small_global,select_range=select_range,
                                         baseblock_enc_mid=baseblock_enc_mid_global, baseblock_dec_mid=baseblock_dec_mid_global,
                                         baseblock_enc_max=baseblock_enc_max_global, baseblock_dec_max=baseblock_dec_max_global,
                                         enc_small_layers=enc_small_layers, dec_small_layers=dec_small_layers,
                                         enc_mid_layers=enc_mid_layers,  dec_mid_layers=dec_mid_layers,
                                         enc_max_layers=enc_max_layers,  dec_max_layers=dec_max_layers, num_global_unet=num_global_unet,
                                         train_size=train_size, patch_size=patch_size, sub_image_size=sub_image_size, drop_path=drop_path, save_memory=save_memory_global, deblur_kernel=deblur_kernel,
                                         deblur_kernel_mode=deblur_kernel_mode, save_memory_kernel=save_memory_ker,
                                         downsample_ker=downsample_ker, dims_ker=dims_ker, pad_size=ks[0],
                                         baseblock_enc_ker=baseblock_enc_ker,
                                         baseblock_dec_ker=baseblock_dec_ker,
                                         enc_ker_layers=enc_ker_layers,  # 256 128 64
                                         dec_ker_layers=dec_ker_layers, get_ker=get_ker, overlap_size=overlap_size, prior_model=prior_model, flip_transpose=flip_transpose,
                                         mid_flip=False, bias=True, deblur_loss_weight=deblur_loss_weight, deblur_mask=deblur_mask,
                                           norm_type=norm_type, spatial_add=spatial_add, spatial_decoder=spatial_decoder, up2scale=up2scale,
                                           small_scale=small_scale, mid_scale=mid_scale, first_col_=True)
        self.padder_size = 8
    def forward_kernel_max(self, feature_max, inp_blur, img_deblur, ker_feature, spatial_feature, k_idx):

        # if self.deblur_kernel_mode == 'ker_max':
        x_feature = [feature_max[-1], feature_max[1], feature_max[2]]
        outdict = self.kernel_estimator[k_idx](x_feature, inp_blur,  deblur_img=img_deblur,
                                                                           ker_feature=ker_feature, spatial_feature=spatial_feature)

        return outdict
    def _clamp_abs(self, data, value):
        with torch.no_grad():
            sign = data.sign()
            data.abs_().clamp_(value)
            data *= sign

    def grids(self, x, level=0):
        b, c, h, w = x.shape
        n_level = 2 ** level

        assert b == 1
        k1, k2 = self.ker_size[0] // n_level, self.ker_size[1] // n_level
        # print(h, w, k1, k2)
        k1 = min(h, k1)
        k2 = min(w, k2)
        # print(k1, k2)
        if self.dynic_overlap and n_level == 1:
            num_row = h // self.ker_size[0] + 1
            num_col = w // self.ker_size[1] + 1
            self.overlap_size[0] = (num_row * k1 - h) // (num_row - 1)
            self.overlap_size[1] = (num_col * k2 - w) // (num_col - 1)

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
        # print(idxes)
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

    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    def forward(self, inp_img, gt=None, batch_size=None):
        loss_deblur = 0.


        k_prior = [0, 0, 0]
        # inps_pry = kornia.geometry.transform.build_pyramid(inp_img, 3)

        # print(len(feature_max))
        if self.sub_image and not self.training:
            inps_pry = kornia.geometry.transform.build_pyramid(inp_img, 3)
            # inp_img = self.grids(inp_img, 0)
            # print(inp_img.shape)
            inps_pry_ = []
            for k in range(len(inps_pry)):
                # print(feature_max[k].shape)
                inps_pry_.append(self.grids(inps_pry[k], k))
            inps_pry = inps_pry_
            # inps_pry = kornia.geometry.transform.build_pyramid(inp_img, 3)
        else:
            inps_pry = kornia.geometry.transform.build_pyramid(inp_img, 3)
        # print(inps_pry[0].shape)
        if gt is not None:
            # gts_pry = kornia.geometry.transform.build_pyramid(gt, 3)
            # gts_pry_ = []
            # print(len(feature_max))
            if self.sub_image and not self.training:
                gts_pry = kornia.geometry.transform.build_pyramid(gt, 3)
                # gt = self.grids(gt, 0)
                gts_pry_ = []

                for k in range(len(gts_pry)):
                    # print(feature_max[k].shape)
                    gts_pry_.append(self.grids(gts_pry[k], k))
                gts_pry = gts_pry_
                # gts_pry = kornia.geometry.transform.build_pyramid(gt, 3)
            else:
                gts_pry = kornia.geometry.transform.build_pyramid(gt, 3)
        else:
            gts_pry = None
        out_dict = {}
        outputs = list()
        if self.num_global_unet > 0:
            if self.frozen_global:
                with torch.no_grad():
                    out_dict_global = self.global_model(inps_pry, gts_pry, k_prior)
            else:
                out_dict_global = self.global_model(inps_pry, gts_pry, k_prior)
                outputs.extend(out_dict_global['img'])
                if self.training:
                    loss_deblur += out_dict_global['loss_deblur']
            if self.get_ker:
                out_dict['max_spatial_feature_list'] = out_dict_global['max_spatial_feature_list']
                out_dict['mid_spatial_feature_list'] = out_dict_global['mid_spatial_feature_list']
                out_dict['max_fourier_feature_list'] = out_dict_global['max_fourier_feature_list']
                out_dict['mid_fourier_feature_list'] = out_dict_global['mid_fourier_feature_list']
                feature_visual = out_dict_global['feature_visual']
                # kernels.extend(out_dict_global['kernel'])
        # print(self.num_kernel_est)

        # print(len(outputs))
        # out_dict['img'] = [x_i[:, :, :h, :w] for x_i in outputs]
        if self.sub_image and not self.training:
            out_dict['img'] = self.grids_inverse(outputs[-1])
        else:
            out_dict['img'] = outputs
        if self.get_ker:
            out_dict['feature_visual'] = feature_visual
            if gt is not None:
                blur_fft = torch.fft.rfft2(inp_img)
                sharp_fft = torch.fft.rfft2(gt)
                otf_gt_deblur = sharp_fft / otf_clamp(blur_fft)
                otf_gt_reblur = blur_fft / otf_clamp(sharp_fft)
                out_dict['otf_gt_deblur_max'] = torch.cat([otf_gt_deblur.real, otf_gt_deblur.imag], dim=1)
                out_dict['otf_gt_reblur_max'] = torch.cat([otf_gt_reblur.real, otf_gt_reblur.imag], dim=1)
                blur_fft2 = torch.fft.fft2(inp_img)
                sharp_fft2 = torch.fft.fft2(gt)
                otf_gt_deblur_ = sharp_fft2 / otf_clamp(blur_fft2)
                otf_gt_reblur_ = blur_fft2 / otf_clamp(sharp_fft2)
                out_dict['ker_deblur_fft'] = torch.log(
                    torch.fft.fftshift(torch.abs(otf_gt_deblur_) + 1e-7, dim=[-2, -1]))
                out_dict['ker_reblur_fft'] = torch.log(
                    torch.fft.fftshift(torch.abs(otf_gt_reblur_) + 1e-7, dim=[-2, -1]))
                blur2_fft = torch.fft.rfft2(inps_pry[1])
                sharp2_fft = torch.fft.rfft2(gts_pry[1])
                otf2_gt_deblur = sharp2_fft / otf_clamp(blur2_fft)
                otf2_gt_reblur = blur2_fft / otf_clamp(sharp2_fft)

                out_dict['otf_gt_deblur_mid'] = torch.cat([otf2_gt_deblur.real, otf2_gt_deblur.imag], dim=1)
                out_dict['otf_gt_reblur_mid'] = torch.cat([otf2_gt_reblur.real, otf2_gt_reblur.imag], dim=1)
                out_dict['blur_pattern_max'] = gts_pry[0] - inps_pry[0]
                out_dict['blur_pattern_mid'] = gts_pry[1] - inps_pry[1]
                out_dict['blur_max'] = inps_pry[0]
                out_dict['blur_mid'] = inps_pry[1]
                out_dict['sharp_max'] = gts_pry[0]
                out_dict['sharp_mid'] = gts_pry[1]
                ker_deblur = convert_otf2psf(otf_gt_deblur, [65, 65])
                ker_reblur = convert_otf2psf(otf_gt_reblur, [65, 65])
                ker_deblur = ker_topk(ker_deblur, 61, 65)
                ker_reblur = ker_topk(ker_reblur, 61, 65)
                # DPDD
                # ker_deblur = convert_otf2psf(otf_gt_deblur, [65, 65])
                # ker_reblur = convert_otf2psf(otf_gt_reblur, [65, 65])
                ker_deblur = torch.mean(ker_deblur, 1, keepdim=True)
                ker_reblur = torch.mean(ker_reblur, 1, keepdim=True)
                out_dict['ker_deblur'] = ker_deblur
                out_dict['ker_reblur'] = ker_reblur
                # out_dict['ker_reblur'] = get_ker_(inp_img, gt, [65, 65])
                # out_dict['ker_deblur'] = get_ker_(gt, inp_img, [65, 65])
        if self.training and not self.frozen_global:
            if self.mid_scale or self.multiscale_enc:
                # if self.frozen_global and self.frozen_kernel:
                #     out_dict['loss_deblur'] = loss_deblur # / (1.5*(self.num_local_unet+self.num_global_unet))
                # elif self.frozen_global:
                #     out_dict['loss_deblur'] = loss_deblur / (self.num_local_unet+self.num_kernel_est)
                # else:
                out_dict['loss_deblur'] = loss_deblur # / (self.num_global_unet)
        # print(out_dict.keys())
        return out_dict


if __name__ == '__main__':
    import yaml

    try:
        from yaml import CLoader as Loader
    except ImportError:
        from yaml import Loader

    # yaml_file = '../../../Motion_Deblurring/Options/Abalation-DeepRFTv2-200k-4gpu.yml'
    yaml_file = '../../../Motion_Deblurring/Options/Abalation-DeepRFTv2-test.yml'
    # yaml_file = '../../../Motion_Deblurring/Options/Abalation-DeepRFTv2-postrain-200k-4gpu.yml'
    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    s = x['network_g'].pop('type')
    ##########################
    print(x['network_g'])

    inp = torch.randn((1, 3, 256, 256)).cuda()
    model = DeepRFTv2(**x['network_g']).cuda()
    model.eval()
    y = model(inp, inp)
    # print(y[-1].shape)
    # print(y['img'][-1].shape)
    inp_shape = (3, 256, 256)

    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, inp_shape, verbose=False, print_per_layer_stat=False)

    # params = float(params[:-3])
    # macs = float(macs[:-4])

    print(macs, params)
    # from fvcore.nn import FlopCountAnalysis, parameter_count_table
    # from thop import clever_format
    # from thop import profile
    #
    # flops, params = profile(model, inputs=(inp,))
    # print(flops / 1e9, params / 1e6)
    # # 创建resnet50网络
    #
    # # 分析FLOPs
    # flops = FlopCountAnalysis(model, inp)
    # print("FLOPs: ", flops.total())

    # 分析parameters
    # print(parameter_count_table(model))
