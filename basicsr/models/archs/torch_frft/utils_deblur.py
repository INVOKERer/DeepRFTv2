#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:23:34 2024

@author: dl
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from basicsr.models.archs.torch_frft.frft import FRFT2D, IFRFT2D
# from frft import FRFT2D, IFRFT2D
import numpy as np
from functools import reduce
import math

class MedianPool2d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.
    
    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """
    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool2d, self).__init__()
        self.k = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _quadruple(padding)  # convert to l, r, t, b
        self.same = same

    def _padding(self, x):
        if self.same:
            ih, iw = x.size()[2:]
            if ih % self.stride[0] == 0:
                ph = max(self.k[0] - self.stride[0], 0)
            else:
                ph = max(self.k[0] - (ih % self.stride[0]), 0)
            if iw % self.stride[1] == 0:
                pw = max(self.k[1] - self.stride[1], 0)
            else:
                pw = max(self.k[1] - (iw % self.stride[1]), 0)
            pl = pw // 2
            pr = pw - pl
            pt = ph // 2
            pb = ph - pt
            padding = (pl, pr, pt, pb)
        else:
            padding = self.padding
        return padding
    
    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd, 
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k[0], self.stride[0]).unfold(3, self.k[1], self.stride[1])
        x = x.contiguous().view(x.size()[:4] + (-1,)).median(dim=-1)[0]
        return x
    
def get_uperleft_denominator(img, kernel):
    ker_f = convert_psf2otf(kernel, img) # discrete fourier transform of kernel
    nsr = wiener_filter_para(img)
    denominator = inv_fft_kernel_est(ker_f, nsr )#
    # img1 = img.cuda()
    # numerator = torch.fft.fft2(img)
    numerator = FRFT2D(img, 0.5)
    deblur = deconv(denominator, numerator)
    return deblur

# --------------------------------
# --------------------------------
def wiener_filter_para(_input_blur):
    median_filter = MedianPool2d(kernel_size=3, padding=1)(_input_blur)
    diff = median_filter - _input_blur
    num = (diff.shape[2]*diff.shape[2])
    mean_n = torch.sum(diff, (2,3)).view(-1,diff.shape[1],1,1)/num
    var_n = torch.sum((diff - mean_n) * (diff - mean_n), (2,3))/(num-1)
    mean_input = torch.sum(_input_blur, (2,3)).view(-1,diff.shape[1],1,1)/num
    var_s2 = (torch.sum((_input_blur-mean_input)*(_input_blur-mean_input), (2,3))/(num-1))**(0.5)
    NSR = var_n / var_s2 * 8.0 / 3.0 / 10.0
    NSR = NSR.view(-1,diff.shape[1],1,1)
    return NSR

# --------------------------------
# --------------------------------
def inv_fft_kernel_est(ker_f, NSR):
    # ker_conju = torch.conj(ker_f).transpose(2,3)
    inv_denominator = ker_f.real * ker_f.real \
                      + ker_f.imag * ker_f.imag + NSR#ker_f*ker_conju + NSR
    # pseudo inverse kernel in flourier domain.
    inv_ker_f = torch.zeros_like(ker_f)
    # inv_ker_f = ker_conju/inv_denominator
    inv_ker_f.real = ker_f.real / inv_denominator
    inv_ker_f.imag = -ker_f.imag / inv_denominator
    return inv_ker_f

# --------------------------------
# --------------------------------
def deconv(inv_ker_f, fft_input_blur):
    # delement-wise multiplication.

    deblur_f = torch.zeros_like(inv_ker_f).cuda()
    # deblur_f = inv_ker_f * fft_input_blur
    # deblur_f[:, :, :, :, 0] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 0] \
    #                         - inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 1]
    # deblur_f[:, :, :, :, 1] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 1] \
    #                         + inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 0]
    deblur_f.real = inv_ker_f.real * fft_input_blur.real \
                    - inv_ker_f.imag * fft_input_blur.imag
    deblur_f.imag = inv_ker_f.real * fft_input_blur.imag \
                            + inv_ker_f.imag * fft_input_blur.real
    # deblur = torch.fft.ifft2(deblur_f)
    deblur = IFRFT2D(deblur_f, 0.5)
    deblur = torch.abs(deblur)
    # batch, channel, height, width = deblur.size()
    # deblur = deblur/(height*width)
    return deblur

# --------------------------------
# --------------------------------
def convert_psf2otf(ker, img):
    psf = torch.zeros_like(img)#.cuda()
    # circularly shift
    centre = ker.shape[2]//2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    # compute the otf
    # otf = torch.fft.fft2(psf)
    otf = FRFT2D(psf, 0.5)
    return otf

# def convert_psf2otf2(ker, img):
#     psf = torch.zeros_like(img)#.cuda()
#     # circularly shift
#     psf[:,:,0:ker.shape[2], 0:ker.shape[3]] = ker
    
#     bs = tuple(int(i) for i in -(np.asarray(ker.shape[-2:])//2))
#     bs = (0,)*(len(img.shape)-2) + bs
#     psf = shift(psf, bs, bc='circular')    
#     return psf


def postprocess(*images, rgb_range):
    def _postprocess(img):
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    return [_postprocess(img) for img in images]

# ker = torch.rand(1,1,13,13)
# img = torch.rand(1,10, 256,256)
# x =get_uperleft_denominator(img, ker)