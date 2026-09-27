import math

import torch
from basicsr.models.archs import median_pool
import kornia
import cv2
from einops import rearrange
'''
This is based on the implementation by Kai Zhang (github: https://github.com/cszn)
'''

"""## Modified Wiener"""

def estimate_noise(I):
    H = I.shape[-2]
    W = I.shape[-1]
    M = torch.tensor([[1., -2., 1.], [-2., 4., -2.], [1., -2., 1.]], device=I.device)
    S = torch.sum(torch.abs(kornia.filters.filter2d(I, M.unsqueeze(0).expand(I.shape[0], -1, -1))))
    S = S*torch.sqrt(torch.tensor(0.5*torch.pi, device=I.device))/(6.0*torch.tensor(W-2.0, device=I.device)*torch.tensor(H-2.0, device=I.device))
    return S
def ModifiedWiener(xs, ker, maxiter=100):
    noise_mean = 0.
    # noise_var = 0.00001
    noise_std = 0.01
    # F =
    Hf = convert_psf2otf(ker, xs.shape)
    f = lambda x: torch.fft.irfftn(torch.fft.rfftn(x[:, :, :], dim=[1,2,3]) * Hf, dim=[1,2,3])
    # F = lambda x: skimage.util.random_noise(f(x), mode='gaussian', mean=noise_mean, var=noise_var)
    y = torch.normal(mean=noise_mean, std=noise_std, size=xs.shape, device=xs.device)+f(xs)
    nsr = estimate_noise(y)
    # a = 100.0*nsr
    a = nsr

    W = y.clone()

    FW = torch.normal(mean=noise_mean, std=noise_std, size=xs.shape, device=xs.device)+W
    H = torch.fft.rfftn(FW, dim=[1,2,3]) / (torch.fft.rfftn(W, dim=[1,2,3]) + 1e-16)

    for i in range(1, maxiter + 1):
        WF = torch.fft.rfftn(W, dim=[1, 2, 3])
        WF_mag = torch.abs(WF)
        WF_phase = torch.angle(WF)
        H = H * (i - 1) / i + torch.fft.rfftn(FW, dim=[1, 2, 3]) / (torch.exp(WF_phase*(1j)) * (WF_mag + 1e-6)) / i
        # H = H * (i - 1) / i + torch.fft.rfftn(FW, dim=[1,2,3]) / (torch.fft.rfftn(W, dim=[1,2,3]) + 1e-16) / i
        Hconj = torch.conj(H)
        W = torch.fft.irfftn(Hconj / (Hconj * H + a) * torch.fft.rfftn(y, dim=[1,2,3]), dim=[1,2,3])
        FW = W + torch.normal(mean=noise_mean, std=noise_std, size=xs.shape, device=xs.device)

    return W
def ModifiedWiener_NSR(xs, ker, noise, maxiter=100):
    H = xs.shape[-2]
    W = xs.shape[-1]
    # F =
    Hf = convert_psf2otf(ker, xs.shape)
    f = lambda x: torch.fft.irfftn(torch.fft.rfftn(x[:, :, :], dim=[1,2,3]) * Hf, dim=[1,2,3])
    # F = lambda x: skimage.util.random_noise(f(x), mode='gaussian', mean=noise_mean, var=noise_var)
    y = f(xs)
    # nsr = estimate_noise(y)
    # a = 100.0*nsr
    S = torch.sum(torch.abs(noise))
    nsr = S * torch.sqrt(torch.tensor(0.5 * torch.pi, device=xs.device)) / (
                6.0 * torch.tensor(W - 2.0, device=xs.device) * torch.tensor(H - 2.0, device=xs.device))
    a = nsr

    W = y.clone()

    # FW = torch.normal(mean=noise_mean, std=noise_std, size=xs.shape, device=xs.device)+W
    H = torch.fft.rfftn(W, dim=[1,2,3]) / (torch.fft.rfftn(W, dim=[1,2,3]) + 1e-16)

    for i in range(1, maxiter + 1):
        WF = torch.fft.rfftn(W, dim=[1, 2, 3])
        WF_mag = torch.abs(WF)
        WF_phase = torch.angle(WF)
        H = H * (i - 1) / i + torch.fft.rfftn(W, dim=[1, 2, 3]) / (torch.exp(WF_phase*(1j)) * (WF_mag + 1e-6)) / i
        # H = H * (i - 1) / i + torch.fft.rfftn(FW, dim=[1,2,3]) / (torch.fft.rfftn(W, dim=[1,2,3]) + 1e-16) / i
        Hconj = torch.conj(H)
        W = torch.fft.irfftn(Hconj / (Hconj * H + a) * torch.fft.rfftn(y, dim=[1,2,3]), dim=[1,2,3])


    return W
# --------------------------------
# --------------------------------
def get_uperleft_denominator(img, kernel, noise=None):
    ker_f = convert_psf2otf(kernel, img.size()) # discrete fourier transform of kernel
    if noise is None:
        nsr = wiener_filter_para(img)
    else:
        nsr = get_NSR_Noise(img, noise)
        # S = torch.sum(torch.abs(noise))
        # nsr = S * torch.sqrt(torch.tensor(0.5 * torch.pi, device=img.device)) / (
        #         6.0 * torch.tensor(W - 2.0, device=img.device) * torch.tensor(H - 2.0, device=img.device))
    denominator = inv_fft_kernel_est(ker_f, nsr)
    # img1 = img.cuda()
    # numerator = torch.fft.fftn(img1, dim=[-3, -2, -1])
    numerator = torch.fft.rfft2(img)
    deblur = deconv(denominator, numerator)
    return deblur

# --------------------------------
# --------------------------------
def wiener_filter_para(_input_blur):
    median_filter = median_pool.MedianPool2d(kernel_size=3, padding=1)(_input_blur)
    diff = median_filter - _input_blur
    # print(diff.shape)
    num = (diff.shape[2]**2)
    mean_n = torch.sum(diff, (2, 3), keepdim=True)/num # .view(-1,1,1,1)
    # print(mean_n.shape)
    var_n = torch.sum((diff - mean_n) ** 2, (2,3), keepdim=True)/(num-1)
    mean_input = torch.sum(_input_blur, (2,3), keepdim=True)/num
    var_s2 = (torch.sum((_input_blur-mean_input)**2, (2, 3), keepdim=True)/(num-1))**(0.5)
    NSR = var_n / var_s2 * 8.0 / 3.0 / 10.0
    # NSR = NSR.view(-1,1,1,1)
    return NSR

def get_NSR_Noise(_input_blur, diff):

    num = (diff.shape[2]**2)
    mean_n = torch.sum(diff, (2, 3), keepdim=True)/num # .view(-1,1,1,1)
    # print(mean_n.shape)
    var_n = torch.sum((diff - mean_n) ** 2, (2,3), keepdim=True)/(num-1)
    mean_input = torch.sum(_input_blur, (2,3), keepdim=True)/num
    var_s2 = (torch.sum((_input_blur-mean_input)**2, (2, 3), keepdim=True)/(num-1))**(0.5) + 1e-6
    NSR = var_n / var_s2 * 8.0 / 3.0 / 10.0
    # NSR = NSR.view(-1,1,1,1)
    return NSR

# --------------------------------
# --------------------------------
def inv_fft_kernel_est(ker_f, NSR):
    ker_f_real = ker_f.real
    ker_f_imag = ker_f.imag
    inv_denominator = ker_f_real ** 2 + ker_f_imag ** 2 + NSR
    # pseudo inverse kernel in flourier domain.
    inv_ker_f = torch.complex(ker_f_real, -ker_f_imag) * (inv_denominator + 1e-6) / (inv_denominator)
    # inv_ker_f = torch.zeros_like(ker_f)
    # print(inv_denominator.shape, NSR.shape)
    # inv_ker_f.real = ker_f_real / inv_denominator
    # inv_ker_f.imag = -ker_f_imag / inv_denominator
    return inv_ker_f

# --------------------------------
# --------------------------------
def deconv(inv_ker_f, fft_input_blur):
    # delement-wise multiplication.
    # deblur_f = torch.zeros_like(inv_ker_f).cuda()
    deblur_f = inv_ker_f.real * fft_input_blur.imag + inv_ker_f.imag * fft_input_blur.real
    # deblur_f[:, :, :, :, 1] = inv_ker_f[:, :, :, :, 0] * fft_input_blur[:, :, :, :, 1] \
    #                         + inv_ker_f[:, :, :, :, 1] * fft_input_blur[:, :, :, :, 0]
    deblur = torch.fft.ifftn(deblur_f, dim=[-3, -2, -1]).real
    # deblur_f = fft_input_blur / inv_ker_f
    # deblur = torch.fft.irfft2(deblur_f)
    return deblur

def convert_psf2otf_roll(psf, size):
    # psf = torch.zeros(size, device=ker.device)
    # # circularly shift
    # # psf = torch.fft.fftshift(ker, dim=[-1, -2])
    # centre = ker.shape[2]//2 + 1
    # psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    # psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    # psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    # psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    otf = torch.zeros(size).type_as(psf)
    otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
    for axis, axis_size in enumerate(psf.shape[2:]):
        otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
    # compute the otf
    otf = torch.fft.rfft2(otf)
    # otf = torch.fft.fftn(psf, dim=[-3, -2, -1])
    return otf
# --------------------------------
# --------------------------------
def convert_psf2otf(ker, size, mode='rfft'):
    # psf = torch.zeros(size, device=ker.device)
    # centre = ker.shape[2] // 2 + 1
    # psf[:, :, :centre, :centre] = ker[:, :, (centre - 1):, (centre - 1):]
    # psf[:, :, :centre, -(centre - 1):] = ker[:, :, (centre - 1):, :(centre - 1)]
    # psf[:, :, -(centre - 1):, :centre] = ker[:, :, : (centre - 1), (centre - 1):]
    # psf[:, :, -(centre - 1):, -(centre - 1):] = ker[:, :, :(centre - 1), :(centre - 1)]
    # circularly shift
    # psf = torch.fft.fftshift(ker, dim=[-1, -2])
    # print(ker.shape, size)
    centre = [size[-2] // 2, size[-1] // 2]
    ker_centre  = [ker.shape[-2] // 2, ker.shape[-1] // 2]
    psf = torch.zeros(size).type_as(ker)
    psf[..., centre[0]-ker_centre[0]:centre[0]+ker_centre[0]+1, centre[1]-ker_centre[1]:centre[1]+ker_centre[1]+1].copy_(ker)
    psf = torch.fft.fftshift(psf, dim=[-1, -2])
    # psf[..., :ker.shape[-2], :ker.shape[-1]].copy_(ker)
    # ker_size0, ker_size1 = ker.shape[-2:]
    # psf = torch.roll(psf, -int(ker_size0 / 2), dims=-2)
    # psf = torch.roll(psf, -int(ker_size1 / 2), dims=-1)
    # for axis, axis_size in enumerate(ker.shape[-2:]):
    #     psf = torch.roll(psf, -int(axis_size / 2), dims=axis + 2)

    # compute the otf
    if mode == 'rfft':
        otf = torch.fft.rfft2(psf)
    else:
        otf = torch.fft.fft2(psf)
    # otf = torch.fft.fftn(psf, dim=[-3, -2, -1])
    return otf

def convert_otf2psf(otf, kernel_size=[31, 31], mode='rfft'):
    if mode == 'rfft':
        psf = torch.fft.irfft2(otf)
    else:
        psf = torch.fft.ifft2(otf).real
    psf = torch.fft.ifftshift(psf, dim=[-2, -1])
    psf = torch.nn.functional.pad(psf, (0, 1, 0, 1))
    ker = kornia.geometry.center_crop(psf, kernel_size)
    # ker = kornia.geometry.center_crop(psf[..., 1:, 1:], kernel_size)
    # for axis, axis_size in enumerate(psf.shape[-2:]):
    # psf = torch.roll(psf, int(kernel_size[0] / 2), dims=-2)
    # psf = torch.roll(psf, int(kernel_size[1] / 2), dims=-1)
    # ker = psf[..., :kernel_size[0], :kernel_size[1]]
    return ker
# def convert_otf2psf(otf, mode='rfft'):
#     # psf = torch.zeros(size, device=ker.device)
#     # # circularly shift
#     # # psf = torch.fft.fftshift(ker, dim=[-1, -2])
#
#     # psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
#     # psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
#     # psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
#     # psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
#     if mode == 'rfft':
#         psf = torch.fft.irfft2(otf)
#     else:
#         psf = torch.fft.ifft2(otf).real
#     for axis, axis_size in enumerate(psf.shape[-2:]):
#         psf = torch.roll(psf, int(axis_size / 2), dims=axis + 2)
#     ker = psf
#     # ker = torch.fft.fftshift(psf)
#     # centre = psf.shape[2] // 2+1
#     # ker = torch.zeros_like(psf)
#     # ker[:, :, -centre:, -centre:] = psf[:, :, :centre, :centre]
#     # ker[:, :, -centre:, :(centre-1)] = psf[:, :, :centre, -(centre-1):]
#     # ker[:, :, : (centre-1), -centre:] = psf[:, :, -(centre-1):, :centre]
#     # ker[:, :, :(centre-1), :(centre-1)] = psf[:, :, -(centre-1):, -(centre-1):]
#     # compute the otf
#
#     # otf = torch.fft.fftn(psf, dim=[-3, -2, -1])
#     return ker
def convert_psf2otf_dim3(ker, size):
    psf = torch.zeros(size, device=ker.device)
    # circularly shift
    # psf = torch.fft.fftshift(ker, dim=[-1, -2])
    centre = ker.shape[2]//2 + 1
    psf[:, :centre, :centre] = ker[:, (centre-1):, (centre-1):]
    psf[:, :centre, -(centre-1):] = ker[:, (centre-1):, :(centre-1)]
    psf[:, -(centre-1):, :centre] = ker[:, : (centre-1), (centre-1):]
    psf[:, -(centre-1):, -(centre-1):] = ker[:, :(centre-1), :(centre-1)]
    # compute the otf
    otf = torch.fft.rfft2(psf)
    # otf = torch.fft.fftn(psf, dim=[-3, -2, -1])
    return otf
def direct_deconv(blur_img, ker):
    ker = ker.expand(-1, blur_img.shape[1], -1, -1)
    blur_fft = torch.fft.rfft2(blur_img)
    otf = convert_psf2otf(ker, blur_img.shape)
    # out = blur_fft / (otf / torch.abs(otf))
    otf_mag = torch.abs(otf)
    otf_phase = torch.angle(otf)
    out = blur_fft / (torch.exp(otf_phase * 1j) * (otf_mag + 1e-7))
    return torch.fft.irfft2(out)

def otf_clamp(otf, eps=1e-7):

    otf_mag = torch.abs(otf)
    otf_phase = torch.angle(otf)
    out = (torch.exp(otf_phase * 1j) * torch.clamp(otf_mag, eps))
    return out
def ker_topk(ker, k_len=61, kernel_size=65):
    ker = torch.mean(ker, dim=1, keepdim=True)

    ker = ker.view(-1, kernel_size * kernel_size)
    _, idx = torch.topk(torch.abs(ker), k=k_len, dim=-1)
    z = torch.zeros_like(ker)
    # print(idx)
    # print(kernels_reblur[:, idx])
    z[:, idx] = 1.
    ker = ker * z
    ker = ker.view(-1, 1, kernel_size, kernel_size)
    # ker = ker.view(-1, kernel_size, kernel_size)
    return ker

def get_ker_clamp(otf1, otf2, eps=1e-7):

    otf_mag = torch.abs(otf1)
    otf_phase = torch.angle(otf1)
    out = (torch.exp(otf_phase * 1j) * torch.clamp(otf_mag, eps))
    return out
def postprocess(*images, rgb_range):
    def _postprocess(img):
        pixel_range = 255 / rgb_range
        return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

    return [_postprocess(img) for img in images]
def deblurfeaturefilter_fft(image, kernel, NSR=None):
    # num_k = kernel.shape[1]
    # ch = image.shape[1]
    ks = max(kernel.shape[-1], kernel.shape[-2])
    dim = (ks, ks, ks, ks)
    image = torch.nn.functional.pad(image, dim, "replicate")
    # image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    # image = rearrange(image, 'b (g c) h w -> b g c h w', g=groups)

    otf = convert_psf2otf(kernel, image.size())
    otf = torch.conj(otf) / (torch.abs(otf) + 1e-7)
    # otf = torch.conj(otf) / (torch.abs(otf) ** 2 + 1e-7)
    # otf = otf.unsqueeze(1).expand(-1, groups, -1, -1, -1)
    # otf = rearrange(otf, 'b k c h w -> b (k c) h w')
    Image_blur = torch.fft.rfft2(image) * otf

    image_blur = torch.fft.irfft2(Image_blur)[:, :, ks:-ks, ks:-ks].contiguous()
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    # image_blur = rearrange(image_blur, 'b g c h w -> b (g c) h w')

    return image_blur

def featurefilter_fft(image, kernel, NSR=None, pad_method='replicate'):
    # num_k = kernel.shape[1]
    # ch = image.shape[1]
    ks = max(kernel.shape[-1] // 2, kernel.shape[-2] // 2)
    if pad_method is not None:
        dim = (ks, ks, ks, ks)
        image = torch.nn.functional.pad(image, dim, pad_method)
    # image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    # image = rearrange(image, 'b (g c) h w -> b g c h w', g=groups)
    # print(image.shape, kernel.shape)
    otf = convert_psf2otf(kernel, image.size())
    # otf = torch.conj(otf) / (torch.abs(otf) + 1e-7)
    # otf = torch.conj(otf) / (torch.abs(otf) ** 2 + 1e-7)
    # otf = otf.unsqueeze(1).expand(-1, groups, -1, -1, -1)
    # otf = rearrange(otf, 'b k c h w -> b (k c) h w')
    Image_blur = torch.fft.rfft2(image) * otf

    if pad_method is not None:
        image_blur = torch.fft.irfft2(Image_blur)[:, :, ks:-ks, ks:-ks].contiguous()
    else:
        ks1 = kernel.shape[-2] // 2
        ks2 = kernel.shape[-1] // 2
        image_blur = torch.fft.irfft2(Image_blur)[:, :, ks1:-ks1, ks2:-ks2].contiguous()
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    # image_blur = rearrange(image_blur, 'b g c h w -> b (g c) h w')

    return image_blur

def featurefilter_deconv_fft(image, kernel, NSR=None, pad_method='replicate'):
    # num_k = kernel.shape[1]
    # ch = image.shape[1]
    ks = max(kernel.shape[-1] // 2, kernel.shape[-2] // 2)
    if pad_method is not None:
        dim = (ks, ks, ks, ks)
        image = torch.nn.functional.pad(image, dim, pad_method)
    # image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    # image = rearrange(image, 'b (g c) h w -> b g c h w', g=groups)

    otf = convert_psf2otf(kernel, image.size())
    # otf = torch.conj(otf) / (torch.abs(otf) + 1e-7)
    if NSR is None:
        otf = otf / (torch.abs(otf) ** 2 + 1e-5)
        # otf = torch.conj(otf) / (torch.abs(otf) + 1e-5)
    else:
        otf = otf / (torch.abs(otf) ** 2 + NSR)
    # otf = torch.conj(otf) / (torch.abs(otf) + 1e-5)
    # otf_real = torch.clamp(otf.real, -100., 100.)
    # otf_imag = torch.clamp(otf.imag, -100., 100.)
    # otf = torch.complex(otf_real, otf_imag)
    # otf = otf.unsqueeze(1).expand(-1, groups, -1, -1, -1)
    # otf = rearrange(otf, 'b k c h w -> b (k c) h w')
    Image_blur = torch.fft.rfft2(image) * otf
    if pad_method is not None:
        image_blur = torch.fft.irfft2(Image_blur)[:, :, ks:-ks, ks:-ks].contiguous()
    else:
        ks1 = kernel.shape[-2] // 2
        ks2 = kernel.shape[-1] // 2
        image_blur = torch.fft.irfft2(Image_blur)[:, :, ks1:-ks1, ks2:-ks2].contiguous()
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    # image_blur = rearrange(image_blur, 'b g c h w -> b (g c) h w')

    return image_blur
def _compute_padding(kernel_size: list[int]) -> list[int]:
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

def fourierfilter_otf(image, otf, mask=None):
    num_k = otf.shape[1]
    # b, c, h, w = image.shape

    Image = torch.fft.fft2(image)
    Image = Image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    if mask is not None:
        mask = mask.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)

    otf = otf.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)

    otf = torch.conj(otf)
    image_conv = torch.fft.ifft2(Image * otf).real
    if mask is not None:
        image_conv = image_conv * mask
    return torch.sum(image_conv, dim=1)
def fourierfilter_otf_2(image, otf):

    Image = torch.fft.rfft2(image)

    # otf = torch.conj(otf)
    image_conv = torch.fft.irfft2(Image * otf)

    return image_conv
def fourierfilter(image, kernel, padding_mode="replicate", mask=None, mask_first=False, method='reblur', NSR=1., ks=None):
    num_k = kernel.shape[1]
    b, c, h, w = image.shape
    # num_k = kernel.shape[1]
    # ch = image.shape[1]
    if ks is None:
        ks = max(kernel.shape[-1], kernel.shape[-2])
        ks = ks // 2
        # print(image.shape, ks, mask.shape)
        padding_shape: list[int] = _compute_padding([kernel.shape[-1], kernel.shape[-2]])
    else:
        padding_shape: list[int] = (ks, ks, ks, ks)
        # print(padding_shape, ks)

    # dim = (ks, ks, ks, ks)
    # print(image.shape)
    if mask is not None:
        mask = mask.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)
    if mask is not None and mask_first:
        image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1) * mask
        # print(image.shape, b, num_k*c, h, w)
        image = image.contiguous().view(b, num_k*c, h, w)
        if ks > 0:
            image = torch.nn.functional.pad(image, padding_shape, padding_mode)
        H, W = image.shape[-2:]
        image = image.view(b, num_k, c, H, W)
        Image = torch.fft.rfft2(image)
    else:
        if ks > 0:
            image = torch.nn.functional.pad(image, padding_shape, padding_mode)
        Image = torch.fft.rfft2(image)
        Image = Image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    otf = convert_psf2otf(kernel, (b, num_k, h+2*ks, w+2*ks))
    otf = otf.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)
    if method == 'deblur':
        otf = otf / (torch.abs(otf) ** 2 + NSR)
    else:
        otf = torch.conj(otf)
    # print(padding_shape, Image.shape, otf.shape)

    image_conv = torch.fft.irfft2(Image * otf) # [..., ks:-ks, ks:-ks].contiguous()
    if ks > 0:
        image_conv = image_conv[..., ks:-ks, ks:-ks].contiguous()
    if mask is not None and not mask_first:
        image_conv = image_conv * mask
    if mask is not None:
        image_conv = torch.sum(image_conv, dim=1)
    else:
        image_conv = torch.mean(image_conv, dim=1)
    return image_conv

def fourierfilter_nopad(image, kernel, padding_mode="replicate", mask=None, mask_first=False, method='reblur', NSR=1., ks=None, return_before=False):
    num_k = kernel.shape[1]
    b, c, h, w = image.shape
    # num_k = kernel.shape[1]
    # ch = image.shape[1]

    # dim = (ks, ks, ks, ks)
    # print(image.shape)
    if mask is not None:
        mask = mask.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)
    if mask is not None and mask_first:
        image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1) * mask
        # print(image.shape, b, num_k*c, h, w)
        image = image.contiguous().view(b, num_k*c, h, w)

        H, W = image.shape[-2:]
        image = image.view(b, num_k, c, H, W)
        Image = torch.fft.rfft2(image)
    else:

        Image = torch.fft.rfft2(image)
        Image = Image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    otf = convert_psf2otf(kernel, (b, num_k, h, w))
    otf = otf.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)
    if method == 'deblur':
        otf = otf / (torch.abs(otf) ** 2 + NSR)
    else:
        otf = torch.conj(otf)
    # print(padding_shape, Image.shape, otf.shape)

    image_conv = torch.fft.irfft2(Image * otf) # [..., ks:-ks, ks:-ks].contiguous()
    if ks > 0:
        image_conv = image_conv[..., ks:-ks, ks:-ks].contiguous()
    if return_before:
        H, W = image_conv.shape[-2:]
        image_convs = image_conv.contiguous()
    if mask is not None and not mask_first:
        image_conv = image_conv * mask
    if mask is not None:
        image_conv = torch.sum(image_conv, dim=1)
    else:
        image_conv = torch.mean(image_conv, dim=1)
    if return_before:
        return image_conv, image_convs
    else:
        return image_conv
def fourierfilter_otf(image, otf, padding_mode="replicate", mask=None, mask_first=False, method='reblur', NSR=1., ks=None):
    num_k = otf.shape[1]

    if mask is not None:
        mask = mask.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)
    if mask is not None and mask_first:
        image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1) * mask
        # print(image.shape, b, num_k*c, h, w)
        # image = image.contiguous().view(b, num_k*c, h, w)
        # image = torch.nn.functional.pad(image, padding_shape, padding_mode)
        # H, W = image.shape[-2:]
        # image = image.view(b, num_k, c, H, W)
        Image = torch.fft.fft2(image)
    else:
        # image = torch.nn.functional.pad(image, padding_shape, padding_mode)
        Image = torch.fft.fft2(image)
        Image = Image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    # otf = convert_psf2otf(kernel, (b, num_k, h+2*ks, w+2*ks))
    otf = otf.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)
    if method == 'deblur':
        otf = otf / (torch.abs(otf) ** 2 + NSR)
    else:
        otf = torch.conj(otf)
    # print(padding_shape, Image.shape, otf.shape)
    image_conv = torch.fft.ifft2(Image * otf).real.contiguous()
    if ks is not None:
        image_conv = image_conv[..., ks:-ks, ks:-ks]
    if mask is not None and not mask_first:
        image_conv = image_conv * mask
    if mask is not None:
        image_conv = torch.sum(image_conv, dim=1)
    else:
        image_conv = torch.mean(image_conv, dim=1)
    return image_conv
def groupFourierFilter(image, kernel, padding_mode="replicate", mask=None, mask_first=False, method='reblur', NSR=1., ks=None):
    num_k = kernel.shape[1]
    b, c, h, w = image.shape
    # num_k = kernel.shape[1]
    # ch = image.shape[1]
    if ks is None:
        ks = max(kernel.shape[-1], kernel.shape[-2])
        ks = ks // 2
        # print(image.shape, ks, mask.shape)
        padding_shape: list[int] = _compute_padding([kernel.shape[-1], kernel.shape[-2]])
    else:
        padding_shape: list[int] = (ks, ks, ks, ks)
        # print(padding_shape, ks)
    # dim = (ks, ks, ks, ks)

    if mask is not None:
        mask = mask.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)
    if mask is not None and mask_first:
        # image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1) * mask
        # # print(image.shape, b, num_k*c, h, w)
        # image = image.contiguous().view(b, num_k*c, h, w)
        image = torch.nn.functional.pad(image, padding_shape, padding_mode)
        H, W = image.shape[-2:]
        image = image.view(b, num_k, -1, H, W)
        Image = torch.fft.rfft2(image)
    else:
        image = torch.nn.functional.pad(image, padding_shape, padding_mode)
        Image = torch.fft.rfft2(image)
        H, W = Image.shape[-2:]
        Image = Image.view(b, num_k, -1, H, W)
        # Image = Image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    otf = convert_psf2otf(kernel, (b, num_k, h+2*ks, w+2*ks))
    otf = otf.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)
    if method == 'deblur':
        otf = otf / (torch.abs(otf) ** 2 + NSR)
    else:
        otf = torch.conj(otf)
    # print(padding_shape, Image.shape, otf.shape)
    image_conv = torch.fft.irfft2(Image * otf)[..., ks:-ks, ks:-ks].contiguous()
    if mask is not None and not mask_first:
        image_conv = image_conv * mask
    image_conv = image_conv.view(b, -1, h, w)
    # if mask is not None:
    #     image_conv = torch.sum(image_conv, dim=1)
    # else:
    #     image_conv = torch.mean(image_conv, dim=1)
    return image_conv
def groupFourierFilter_nopad(image, kernel, padding_mode="replicate", mask=None, mask_first=False, method='reblur', NSR=1., ks=None):
    num_k = kernel.shape[1]
    b, c, h, w = image.shape
    # num_k = kernel.shape[1]
    # ch = image.shape[1]
    # if ks is None:
    #     ks = max(kernel.shape[-1], kernel.shape[-2])
    #     ks = ks // 2
    #     # print(image.shape, ks, mask.shape)
    #     padding_shape: list[int] = _compute_padding([kernel.shape[-1], kernel.shape[-2]])
    # else:
    #     padding_shape: list[int] = (ks, ks, ks, ks)
        # print(padding_shape, ks)
    # dim = (ks, ks, ks, ks)

    if mask is not None:
        mask = mask.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)

    Image = torch.fft.rfft2(image)
    H, W = Image.shape[-2:]
    Image = Image.view(b, num_k, -1, H, W)
        # Image = Image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    otf = convert_psf2otf(kernel, (b, num_k, h, w))
    otf = otf.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)
    if method == 'deblur':
        otf = otf / (torch.abs(otf) ** 2 + NSR)
    else:
        otf = torch.conj(otf)
    # print(padding_shape, Image.shape, otf.shape)
    image_conv = torch.fft.irfft2(Image * otf)
    if ks is not None and ks > 0:
        image_conv = image_conv[..., ks:-ks, ks:-ks].contiguous()
        h, w = h - 2 * ks, w - 2 * ks
    if mask is not None and not mask_first:
        image_conv = image_conv * mask

    image_conv = image_conv.view(b, -1, h, w)
    # if mask is not None:
    #     image_conv = torch.sum(image_conv, dim=1)
    # else:
    #     image_conv = torch.mean(image_conv, dim=1)
    return image_conv

def groupFourierFilter_otf(image, otf, padding_mode="replicate", mask=None, mask_first=False, method='conj', NSR=1., ks=None):
    num_k = otf.shape[1]
    b, c, h, w = image.shape

    if mask is not None:
        mask = mask.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)
    if mask is not None and mask_first:
        # image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1) * mask
        # # print(image.shape, b, num_k*c, h, w)
        # image = image.contiguous().view(b, num_k*c, h, w)
        # image = torch.nn.functional.pad(image, padding_shape, padding_mode)
        H, W = image.shape[-2:]
        image = image.view(b, num_k, -1, H, W) * mask
        Image = torch.fft.fft2(image)
    else:
        # image = torch.nn.functional.pad(image, padding_shape, padding_mode)
        Image = torch.fft.fft2(image)
        H, W = Image.shape[-2:]
        Image = Image.view(b, num_k, -1, H, W)
        # Image = Image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

    # otf = convert_psf2otf(kernel, (b, num_k, h+2*ks, w+2*ks))
    otf = otf.unsqueeze(2) # .expand(-1, -1, ch, -1, -1)
    if method == 'deblur':
        otf = otf / (torch.abs(otf) ** 2 + NSR)
    elif method == 'conj':
        otf = torch.conj(otf)
    # print(padding_shape, Image.shape, otf.shape)
    image_conv = torch.fft.ifft2(Image * otf).real.contiguous()
    if ks is not None:
        image_conv = image_conv[..., ks:-ks, ks:-ks]
    else:
        ks = 0
    if mask is not None and not mask_first:
        image_conv = image_conv * mask
    image_conv = image_conv.view(b, -1, h-2*ks, w-2*ks)
    # if mask is not None:
    #     image_conv = torch.sum(image_conv, dim=1)
    # else:
    #     image_conv = torch.mean(image_conv, dim=1)
    return image_conv
def reblurfilter_fftx(image, kernel, mask=None, method='reblur', NSR=1.):
    num_k = kernel.shape[1]
    ch = image.shape[1]
    ks = max(kernel.shape[-1], kernel.shape[-2])
    # ks = ks // 2
    # print(image.shape, ks, mask.shape)
    dim = (ks, ks, ks, ks)
    if mask is not None:
        mask = mask.unsqueeze(2).expand(-1, -1, ch, -1, -1)

    b, nk = kernel.shape[:2]
    h, w = image.shape[-2:]
    otf = convert_psf2otf(kernel, (b, nk, h+2*ks, w+2*ks))
    otf = otf.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    # otf = rearrange(otf, 'b k c h w -> b (k c) h w')
    if method == 'deblur':
        # otf = otf / (torch.abs(otf)+1e-7)

        # if mask is not None:
        #     image = image * mask
        # image = rearrange(image, 'b k c h w -> b (k c) h w')
        image = torch.nn.functional.pad(image, dim, "replicate")

        otf = otf / (torch.abs(otf) ** 2 + NSR)
        # otf = torch.exp(1j*torch.angle(otf))
        Image = torch.fft.rfft2(image)
        Image = Image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
        # Image_blur = torch.fft.rfft2(image)
        image_blur = torch.fft.irfft2(Image * otf)[..., ks:-ks, ks:-ks].contiguous()
        # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
        # image_blur = rearrange(image_blur, 'b (k c) h w -> b k c h w', k=num_k, c=ch)
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

        # if mask is not None:
        #     image = image * mask
        # image = rearrange(image, 'b k c h w -> b (k c) h w')
        image = torch.nn.functional.pad(image, dim, "replicate")
        Image = torch.fft.rfft2(image)
        Image = Image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
        # Image = rearrange(Image, 'b k c h w -> b (k c) h w')
        # Image_blur = torch.fft.rfft2(image)
        otf = torch.conj(otf)
        image_blur = torch.fft.irfft2(Image * otf)[..., ks:-ks, ks:-ks].contiguous()
        # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
        # image_blur = rearrange(image_blur, 'b (k c) h w -> b k c h w', k=num_k, c=ch)
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

def conv_fft(estimated_image, otf, num_k, dim, ks, ch):
    blurred_image = estimated_image # .unsqueeze(1).expand(-1, num_k, -1, -1, -1)
    # if mask is not None:
    #     image = image * mask
    # blurred_image = rearrange(blurred_image, 'b k c h w -> b (k c) h w')
    blurred_image = torch.nn.functional.pad(blurred_image, dim, "replicate")
    blurred_image = torch.fft.rfft2(blurred_image) * torch.conj(otf)
    # Image_blur = torch.fft.rfft2(image)
    blurred_image = torch.fft.irfft2(blurred_image)[:, :, ks:-ks, ks:-ks].contiguous()
    # image_blur = kornia.filters.filter2d(image.unsqueeze(1), kernel)
    # blurred_image = rearrange(blurred_image, 'b (k c) h w -> b k c h w', k=num_k, c=ch)
    # if mask is not None:
    #     blurred_image = blurred_image * mask
    return blurred_image # torch.sum(blurred_image, dim=1)


def richardson_lucy(image, kernel, mask=None, estimated_image=None, iterations=100):

    num_k = kernel.shape[1]
    ch = image.shape[1]
    ks = max(kernel.shape[-1], kernel.shape[-2])
    # ks = ks // 2
    dim = (ks, ks, ks, ks)
    if mask is not None:
        mask = mask.unsqueeze(2).expand(-1, -1, ch, -1, -1)

    b, nk = kernel.shape[:2]
    h, w = image.shape[-2:]
    otf = convert_psf2otf(kernel, (b, nk, h + 2 * ks, w + 2 * ks))
    otf = otf.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    otf = rearrange(otf, 'b k c h w -> (b k) c h w')
    # print(otf.shape, ks, h, w)
    otf_conj = torch.conj(otf)
    image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
    # if mask is not None:
    #     estimated_image = estimated_image * mask
    image = rearrange(image, 'b k c h w -> (b k) c h w')
    if estimated_image is None:
        estimated_image = torch.ones_like(image)
        estimated_image = torch.nn.functional.pad(estimated_image, dim, 'replicate')
    else:
        estimated_image = torch.nn.functional.pad(estimated_image, dim, 'replicate')
        estimated_image = estimated_image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
        estimated_image = rearrange(estimated_image, 'b k c h w -> (b k) c h w')
    image = torch.nn.functional.pad(image, dim, 'replicate')
    for _ in range(iterations):
        blurred_image = fourierfilter_otf_2(estimated_image, otf_conj)
        error = image / torch.clamp(blurred_image, 1e-3, 1.) # torch.clamp(blurred_image, 1e-9) # (blurred_image + 1e-9)
        error = torch.clamp(error, 0., 2.)
        estimated_image = estimated_image * fourierfilter_otf_2(error, otf)
        # print(estimated_image.shape)
    estimated_image = rearrange(estimated_image, '(b k) c h w -> b k c h w', k=num_k, c=ch)
    estimated_image = estimated_image[..., ks:-ks, ks:-ks]
    if mask is not None:
        estimated_image = estimated_image * mask
    # print(estimated_image.shape)
    estimated_image = torch.sum(estimated_image, dim=1)
    return torch.clamp(estimated_image, 0., 1.)
    # return estimated_image

def group_richardson_lucy(image, kernel, mask=None, estimated_image=None, iterations=100):

    num_k = kernel.shape[1]
    ch = image.shape[1] // num_k
    ks = max(kernel.shape[-1], kernel.shape[-2])
    ks = ks // 2
    dim = (ks, ks, ks, ks)
    if mask is not None:
        mask = mask.unsqueeze(2).expand(-1, -1, ch, -1, -1)

    b, nk = kernel.shape[:2]
    h, w = image.shape[-2:]
    otf = convert_psf2otf(kernel, (b, nk, h + 2 * ks, w + 2 * ks))
    otf = otf.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    otf = rearrange(otf, 'b k c h w -> (b k) c h w')
    # print(otf.shape, ks, h, w)
    otf_conj = torch.conj(otf)
    # image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
    # if mask is not None:
    #     estimated_image = estimated_image * mask
    image = rearrange(image, 'b (k c) h w -> (b k) c h w', k=num_k)
    if estimated_image is None:
        estimated_image = torch.ones_like(image)
        estimated_image = torch.nn.functional.pad(estimated_image, dim, 'replicate')
    else:
        estimated_image = torch.nn.functional.pad(estimated_image, dim, 'replicate')
        estimated_image = estimated_image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
        estimated_image = rearrange(estimated_image, 'b k c h w -> (b k) c h w')
    image = torch.nn.functional.pad(image, dim, 'replicate')
    for _ in range(iterations):
        blurred_image = fourierfilter_otf_2(estimated_image, otf_conj)
        error = image / torch.clamp(blurred_image, 1e-3, 1.) # torch.clamp(blurred_image, 1e-9) # (blurred_image + 1e-9)
        error = torch.clamp(error, 0., 2.)
        estimated_image = estimated_image * fourierfilter_otf_2(error, otf)
        # print(estimated_image.shape)
    estimated_image = rearrange(estimated_image, '(b k) c h w -> b k c h w', k=num_k, c=ch)
    estimated_image = estimated_image[..., ks:-ks, ks:-ks]
    if mask is not None:
        estimated_image = estimated_image * mask
    # print(estimated_image.shape)
    # estimated_image = torch.sum(estimated_image, dim=1)
    estimated_image = rearrange(estimated_image, 'b k c h w -> b (k c) h w')
    return torch.clamp(estimated_image, 0., 1.)
    # return estimated_image
def richardson_lucy_kernel(image, kernel, mask=None, estimated_kernel=None, iterations=100):

    num_k = kernel.shape[1]
    ch = image.shape[1]
    ks = max(kernel.shape[-1], kernel.shape[-2])
    ks = ks // 2
    dim = (ks, ks, ks, ks)
    if mask is not None:
        mask = mask.unsqueeze(2).expand(-1, -1, ch, -1, -1)

    b, nk = kernel.shape[:2]
    h, w = image.shape[-2:]
    otf = convert_psf2otf(image, (b, nk, h + 2 * ks, w + 2 * ks))
    otf = otf.unsqueeze(2).expand(-1, -1, ch, -1, -1)
    otf = rearrange(otf, 'b k c h w -> (b k) c h w')
    otf_conj = torch.conj(otf)
    image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
    # if mask is not None:
    #     estimated_image = estimated_image * mask
    image = rearrange(image, 'b k c h w -> (b k) c h w')
    if estimated_kernel is None:
        estimated_kernel = torch.ones_like(image)
    else:
        estimated_kernel = estimated_kernel.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
        estimated_kernel = rearrange(estimated_kernel, 'b k c h w -> (b k) c h w')
    for _ in range(iterations):
        blurred_image = conv_fft(estimated_kernel, otf, num_k, dim, ks, ch)
        error = kernel / torch.clamp(blurred_image, 1e-9)
        estimated_kernel = estimated_kernel * conv_fft(error, otf_conj, num_k, dim, ks, ch)
        # print(estimated_image.shape)
    estimated_kernel = rearrange(estimated_kernel, '(b k) c h w -> b k c h w', k=num_k, c=ch)
    if mask is not None:
        estimated_kernel = estimated_kernel * mask
    # print(estimated_image.shape)
    return torch.sum(estimated_kernel, dim=1)


if __name__=='__main__':
    # model = torch.nn.MultiheadAttention(16, 1, dropout=0., batch_first=True)
    # model = model.cuda()
    # q = torch.randn((1, 8, 16)).cuda()
    # k = torch.randn((1, 16)).cuda()
    # from einops import repeat
    # k = repeat(k, 'b c -> b k c', k=8)
    # v = torch.randn((1, 8, 16)).cuda()
    # x, _ = model(q, k, v)
    # y1 = torch.bmm(q, k.transpose(-2, -1)) @ v
    # y2 = torch.softmax(q @ k.transpose(-2, -1), dim=-1) @ v
    # z = (k @ q.transpose(-2, -1)) @ v
    # # print(y.shape)
    # print(x)
    # print(y1)
    # print(y2)
    # print(z)
    # y, _ = model(q, k, v)

    # print(torch.sum(torch.abs(x-y)))
    import numpy as np
    x = cv2.imread('/data/mxt_data/GoPro/val/target_crops_384/0.png')
    x_tensor = kornia.image_to_tensor(x / 255., keepdim=False)
    kernel = kornia.filters.get_motion_kernel2d(31, angle=21, direction=-0.5)
    print(kernel.shape)
    kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
    x_tensor_blur = kornia.filters.filter2d(x_tensor, kernel, border_type='replicate')
    x_blur = fourierfilter(x_tensor, kernel.unsqueeze(1))
    print('x_blur: ', torch.sum(torch.abs(x_blur - x_tensor_blur)))
    ks = 31//2
    x_tensor_pad = torch.nn.functional.pad(x_tensor, [ks, ks, ks, ks], 'replicate')
    otf_reblur = convert_psf2otf(kernel.unsqueeze(1), [1, 1, 384+ks*2, 384+ks*2], 'fft')
    y_blur = fourierfilter_otf(x_tensor_pad, otf_reblur)[..., ks:-ks, ks:-ks]
    print('x_blur: ', torch.sum(torch.abs(x_blur - y_blur)))
    # b, c, h, w = x_tensor.shape
    # kernel = kernel.unsqueeze(0)
    # ks = max(kernel.shape[-1], kernel.shape[-2])
    # ks = ks // 2
    # otf = convert_psf2otf(kernel, (b, 1, h + 2 * ks, w + 2 * ks))
    # otf2 = convert_psf2otf_roll(kernel, (b, 1, h + 2 * ks, w + 2 * ks))
    ker = convert_otf2psf(otf_reblur, [31, 31], 'fft')
    # ker = kornia.geometry.center_crop(ker, [31, 31])
    # for axis, axis_size in enumerate(x_blur.shape[-2:]):
    #     x_blur = torch.roll(x_blur, -int(axis_size / 2), dims=axis + 2)
    # for axis, axis_size in enumerate(x_blur.shape[-2:]):
    #     x_blur = torch.roll(x_blur, int(axis_size / 2), dims=axis + 2)
    # print(ker.squeeze(), kernel.squeeze())
    print('ker: ', torch.mean(torch.abs(ker - kernel)))
    temp = x_tensor_blur
    x_tensor_deblur = x_tensor_blur
    for i in range(5):
        temp = temp - kornia.filters.filter2d(temp, kernel, border_type='replicate')
        x_tensor_deblur = x_tensor_blur + temp
    img_ = kornia.tensor_to_image(x_tensor_deblur)
    img_blur = kornia.tensor_to_image(x_tensor_blur)
    cv2.imwrite('/data/mxt_code/DeepMXT/Motion_Deblurring/tmp/NRKNet3.jpg', np.uint8(img_ * 255.))
    cv2.imwrite('/data/mxt_code/DeepMXT/Motion_Deblurring/tmp/blur.png', np.uint8(img_blur * 255.))
    # print('otf: ', torch.sum(torch.abs(otf2 - otf)))
    # otf_ = convert_psf2otf(kernel, kernel.size(), mode='fft')
    # ker = convert_otf2psf(otf_, mode='fft')
    # print('ker: ', torch.sum(torch.abs(ker - kernel)))

    # otf_deblur = torch.conj(otf) # / (torch.abs(otf) + 1e-8) ** 2
    # kernel_deblur = torch.fft.irfft2(otf_deblur)
    # kernel_deblur = kornia.geometry.center_crop(kernel_deblur, [31, 31])
    # x_tensor_deblur = richardson_lucy(x_tensor_blur, kernel, iterations=50)
    # # print(x_tensor_deblur.shape)
    # print(x_tensor_deblur.max(), x_tensor_deblur.min())
    # print(kornia.metrics.psnr(x_tensor, x_tensor_blur, 1.))
    # print(kornia.metrics.psnr(x_tensor, x_tensor_deblur, 1.))
    # img_ = kornia.tensor_to_image(x_tensor_deblur)
    # cv2.imwrite('/home/ubuntu/90t/personal_data/mxt/MXT/exp_results/richlucy_50iter.jpg', np.uint8(img_ * 255.))
    # img_ = kornia.tensor_to_image(x_tensor_blur)
    # cv2.imwrite('/home/ubuntu/wy-3090-1/mxt_code/DeepMXT/Motion_Deblurring/results_DeepReFT/blurv1.png',
    #             np.uint8(img_ * 255.))
    # x_fft = torch.fft.rfft2(x_tensor)
    # x_mag = torch.abs(x_fft)
    # x_phase = torch.angle(x_fft)
    # print(torch.sum(torch.abs(x_fft-x_mag*torch.exp(1j*x_phase))))
    # y = cv2.imread('/home/mxt/106-48t/personal_data/mxt/Datasets/Deblur/GoPro/val/t0/100.png')
    # y_tensor = kornia.image_to_tensor(y / 255., keepdim=False)
    # xy = torch.cat([x_tensor, y_tensor], dim=0)
    # h, w = xy.shape[-2:]
    # xy = xy.view(-1, 1, h, w)
    # z = xy[:3, :, :, :]
    # z = z.view(1,3,h,w)
    # img_ = kornia.tensor_to_image(z)
    # cv2.imwrite('/home/mxt/106-48t/personal_data/mxt/Datasets/Deblur/GoPro/val/t0/0_z.png', np.uint8(img_ * 255.))
    # kernel = kornia.filters.get_motion_kernel2d(31, angle=21)
    # print(kernel.shape)
    # kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
    # print(kernel.shape)
    # H, W = x_tensor.shape[-2:]
    # h, w = kernel.shape[-2:]
    # pad_1 = (H-h)//2
    # pad_2 = (W-w)//2
    # pad_1_ = H - (H - h) // 2 - h
    # pad_2_ = W - (W - w) // 2 - w
    # pad = torch.nn.ZeroPad2d((pad_1, pad_1_, pad_2, pad_2_))
    # x_blur = kornia.filters.filter2d(x_tensor, kernel)
    # kernel_x = kernel.unsqueeze(0)
    # ks = 31
    # dim = (ks, ks, ks, ks)
    # x_blur = torch.nn.functional.pad(x_blur, dim, "replicate")
    # # print(kernel_x.shape, pad_1, pad_1_, pad_2)
    # # print(kernel_x)
    # # kernel_x = pad(kernel_x)
    # # print(kernel_x.shape)
    # out_weiner = get_uperleft_denominator(x_blur.cuda(), kernel_x.cuda())
    # # out_weiner = kornia.enhance.normalize_min_max(out_weiner, 0., 1.)
    # # out_weiner = torch.clamp(out_weiner, 0., 1.)
    # # print(out_weiner.max(), out_weiner.min())
    # # print(torch.mean(out_weiner - x_blur))
    # # print('mean_x', torch.mean(out_weiner - x_tensor))
    # img_ = kornia.tensor_to_image(out_weiner)
    # # cv2.imwrite('/home/ubuntu/106-48t/personal_data/mxt/Datasets/Deblur/GoPro/val/t0/0_deconv.png', np.uint8(img_*255.))