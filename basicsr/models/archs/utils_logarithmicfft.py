# import numpy as np
import torch
import matplotlib.pyplot as plt
import math
import kornia
def logarithmic_rfft(image, add=math.exp(-6)):
    image_fft = torch.fft.rfft2(image)
    # image_fft_shift = torch.fft.fftshift(image_fft, dim=[-2, -1])
    # image_fft_shift = torch.cat([image_fft_shift.real, image_fft_shift.imag], dim=1)
    mag = torch.clamp(torch.abs(image_fft), add)
    phase = torch.angle(image_fft)
    # image_fft_shift_s = image_fft_shift / (image_fft_shift_abs+1e-7)
    mag_log = torch.log(mag+add)
    return mag_log, phase
def inverse_logarithmic_rfft(x_mag, x_phase, sub=0.):

    I_mag = torch.exp(x_mag) - sub

    I_phase = torch.exp(1j*x_phase)
    Image = I_mag * I_phase
    image = torch.fft.irfft2(Image)
    # Image = torch.fft.ifftshift(Image, dim=[-2, -1])
    # image = torch.fft.irfft2(Image[..., :Image.shape[-1]//2+1])
    return image
def logarithmic_fft(image, add=math.exp(-2)):
    image_fft = torch.fft.fft2(image)
    image_fft_shift = torch.fft.fftshift(image_fft, dim=[-2, -1])
    # image_fft_shift = torch.cat([image_fft_shift.real, image_fft_shift.imag], dim=1)
    mag = torch.abs(image_fft_shift)

    phase = torch.angle(image_fft_shift)
    # image_fft_shift_s = image_fft_shift / (image_fft_shift_abs+1e-7)
    mag_log = torch.log(mag+add) # * image_fft_shift_s
    return mag, mag_log, phase

def logarithmic_fft_clamp(image, clamp_min=math.exp(-10)):
    image_fft = torch.fft.fft2(image)
    image_fft_shift = torch.fft.fftshift(image_fft, dim=[-2, -1])
    # image_fft_shift = torch.cat([image_fft_shift.real, image_fft_shift.imag], dim=1)
    mag = torch.clamp(torch.abs(image_fft_shift), clamp_min)
    phase = torch.angle(image_fft_shift)
    # image_fft_shift_s = image_fft_shift / (image_fft_shift_abs+1e-7)
    mag_log = torch.log(mag) # * image_fft_shift_s
    return mag, mag_log, phase
def fft_clamp(image, clamp_min=math.exp(-10)):
    image_fft = torch.fft.fft2(image)
    image_fft_shift = torch.fft.fftshift(image_fft, dim=[-2, -1])
    # image_fft_shift = torch.cat([image_fft_shift.real, image_fft_shift.imag], dim=1)
    mag = torch.clamp(torch.abs(image_fft_shift), clamp_min)
    phase = torch.angle(image_fft_shift)
    # image_fft_shift_s = image_fft_shift / (image_fft_shift_abs+1e-7)
    # mag_log = torch.log(mag) # * image_fft_shift_s
    return mag, phase
def fft_ri(image, clamp_min=math.exp(-10), norm='backward'):
    image_fft = torch.fft.fft2(image, norm=norm)
    image_fft_shift = torch.fft.fftshift(image_fft, dim=[-2, -1])
    # image_fft_shift = torch.cat([image_fft_shift.real, image_fft_shift.imag], dim=1)
    x_real, x_imag = image_fft_shift.real, image_fft_shift.imag
    # image_fft_shift_s = image_fft_shift / (image_fft_shift_abs+1e-7)
    # mag_log = torch.log(mag) # * image_fft_shift_s
    return x_real, x_imag
def fft_ri_(image, clamp_min=math.exp(-10), norm='backward'):
    image_fft = torch.fft.fft2(image, norm=norm)
    image_fft_shift = torch.fft.fftshift(image_fft, dim=[-2, -1])
    # image_fft_shift = torch.cat([image_fft_shift.real, image_fft_shift.imag], dim=1)
    # x_real, x_imag = image_fft_shift.real, image_fft_shift.imag
    # image_fft_shift_s = image_fft_shift / (image_fft_shift_abs+1e-7)
    # mag_log = torch.log(mag) # * image_fft_shift_s
    return image_fft_shift
def logarithmic_fft_clamp_rot180(image, clamp_min=math.exp(-10)):
    h, w = image.shape[-2:]
    image_fft = torch.fft.fft2(image)
    image_fft[..., 1:, 1+w//2:] = torch.conj(image_fft[..., 1:, 1+w//2:])
    image_fft_shift = torch.fft.fftshift(image_fft, dim=[-2, -1])
    # image_fft_shift = torch.cat([image_fft_shift.real, image_fft_shift.imag], dim=1)
    mag = torch.clamp(torch.abs(image_fft_shift), clamp_min)
    phase = torch.angle(image_fft_shift)
    # image_fft_shift_s = image_fft_shift / (image_fft_shift_abs+1e-7)
    mag_log = torch.log(mag) # * image_fft_shift_s
    return mag, mag_log, phase
def inverse_logarithmic_fft(x_mag, x_phase, sub=0):

    I_mag = torch.exp(x_mag) - sub

    I_phase = torch.exp(1j*x_phase)
    Image = I_mag * I_phase
    Image = torch.fft.ifftshift(Image, dim=[-2, -1])
    image = torch.fft.ifft2(Image).real
    return image
def inverse_logarithmic_irfft(x_mag, x_phase, sub=math.exp(-2)):

    I_mag = torch.exp(x_mag) - sub

    I_phase = torch.exp(1j*x_phase)
    Image = I_mag * I_phase
    # image = torch.fft.irfft2(Image)
    Image = torch.fft.ifftshift(Image, dim=[-2, -1])
    image = torch.fft.irfft2(Image[..., :Image.shape[-1]//2+1])
    return image

def logarithmic_fft_ker(image, add=math.exp(-10)):
    image_fft = torch.fft.fft2(image)
    # image_fft_shift = torch.fft.fftshift(image_fft, dim=[-2, -1])
    # image_fft_shift = torch.cat([image_fft_shift.real, image_fft_shift.imag], dim=1)
    mag = torch.clamp(torch.abs(image_fft), add)
    phase = torch.angle(image_fft)
    # image_fft_shift_s = image_fft_shift / (image_fft_shift_abs+1e-7)
    mag_log = torch.log(mag + add)  # * image_fft_shift_s
    return mag, mag_log, phase
def inverse_logarithmic_fft_ker(x_mag, x_phase, sub=0):

    I_mag = torch.exp(x_mag) - sub

    I_phase = torch.exp(1j*x_phase)
    Image = I_mag * I_phase
    # Image = torch.fft.ifftshift(Image, dim=[-2, -1])
    image = torch.fft.ifft2(Image).real
    return image

def logarithmic_fft_kernel(kernel, size, clamp_min=math.exp(-10)):

    otf = convert_psf2otf(kernel, size, 'fft')
    otf = torch.fft.fftshift(otf, dim=[-2, -1])
    mag = torch.clamp(torch.abs(otf), clamp_min)
    phase = torch.angle(otf)
    # image_fft_shift_s = image_fft_shift / (image_fft_shift_abs+1e-7)
    mag_log = torch.log(mag)  # * image_fft_shift_s
    return mag, mag_log, phase
def logarithmic_fft_kernel_rot180(kernel, size, clamp_min=math.exp(-10)):

    otf = convert_psf2otf(kernel, size, 'fft')
    w = otf.shape[-1]
    otf[..., 1:, 1 + w // 2:] = torch.conj(otf[..., 1:, 1 + w // 2:])
    otf = torch.fft.fftshift(otf, dim=[-2, -1])
    mag = torch.clamp(torch.abs(otf), clamp_min)
    phase = torch.angle(otf)
    # image_fft_shift_s = image_fft_shift / (image_fft_shift_abs+1e-7)
    mag_log = torch.log(mag)  # * image_fft_shift_s
    return mag, mag_log, phase
def inverse_logarithmic_ifft_kernel(x_mag, x_phase, kernel_size, sub=0.):

    I_mag = torch.exp(x_mag) - sub

    I_phase = torch.exp(1j*x_phase)
    otf = I_mag * I_phase
    # image = torch.fft.irfft2(Image)
    otf = torch.fft.ifftshift(otf, dim=[-2, -1])
    kernel = convert_otf2psf(otf, kernel_size, mode='fft')
    # psf = torch.fft.ifft2(otf).real
    # psf = torch.fft.ifftshift(psf, dim=[-2, -1])
    # for axis, axis_size in enumerate(psf.shape[2:]):
    #     psf = torch.roll(psf, int(axis_size / 2), dims=axis+2)
    return kernel, otf
def inverse_ifft_kernel(x_mag, x_phase, kernel_size, sub=0.):

    I_mag = x_mag # torch.exp(x_mag) - sub

    I_phase = torch.exp(1j*x_phase)
    otf = I_mag * I_phase
    # image = torch.fft.irfft2(Image)
    otf = torch.fft.ifftshift(otf, dim=[-2, -1])
    kernel = convert_otf2psf(otf, kernel_size, mode='fft')
    # psf = torch.fft.ifft2(otf).real
    # psf = torch.fft.ifftshift(psf, dim=[-2, -1])
    # for axis, axis_size in enumerate(psf.shape[2:]):
    #     psf = torch.roll(psf, int(axis_size / 2), dims=axis+2)
    return kernel, otf
def inverse_ifft_ri_kernel(x_real, x_imag, kernel_size, sub=0., norm='backward'):

    # I_mag = x_mag # torch.exp(x_mag) - sub
    #
    # I_phase = torch.exp(1j*x_phase)
    otf = torch.complex(x_real, x_imag)
    # image = torch.fft.irfft2(Image)
    otf = torch.fft.ifftshift(otf, dim=[-2, -1])
    kernel = convert_otf2psf(otf, kernel_size, mode='fft', norm=norm)
    # psf = torch.fft.ifft2(otf).real
    # psf = torch.fft.ifftshift(psf, dim=[-2, -1])
    # for axis, axis_size in enumerate(psf.shape[2:]):
    #     psf = torch.roll(psf, int(axis_size / 2), dims=axis+2)
    return kernel, otf
def inverse_ifft_ri_otf(x_real, x_imag, kernel_size=19, sub=0., norm='backward'):

    # I_mag = x_mag # torch.exp(x_mag) - sub
    #
    # I_phase = torch.exp(1j*x_phase)
    otf = torch.complex(x_real, x_imag)
    # image = torch.fft.irfft2(Image)
    otf = torch.fft.ifftshift(otf, dim=[-2, -1])
    # kernel = convert_otf2psf(otf, kernel_size, mode='fft', norm=norm)
    # psf = torch.fft.ifft2(otf).real
    # psf = torch.fft.ifftshift(psf, dim=[-2, -1])
    # for axis, axis_size in enumerate(psf.shape[2:]):
    #     psf = torch.roll(psf, int(axis_size / 2), dims=axis+2)
    return otf
def inverse_irfft_ri_kernel(x_real, x_imag, kernel_size, sub=0.):

    # I_mag = x_mag # torch.exp(x_mag) - sub
    #
    # I_phase = torch.exp(1j*x_phase)
    otf = torch.complex(x_real, x_imag)
    h, w = x_real.shape[-2:]
    # image = torch.fft.irfft2(Image)
    otf = torch.fft.ifftshift(otf, dim=[-2, -1])
    kernel = convert_otf2psf(otf[..., :w//2+1], kernel_size, mode='rfft')
    # psf = torch.fft.ifft2(otf).real
    # psf = torch.fft.ifftshift(psf, dim=[-2, -1])
    # for axis, axis_size in enumerate(psf.shape[2:]):
    #     psf = torch.roll(psf, int(axis_size / 2), dims=axis+2)
    return kernel, otf
def inverse_logarithmic_irfft_kernel(x_mag, x_phase, kernel_size, sub=0.):

    I_mag = torch.exp(x_mag) - sub

    I_phase = torch.exp(1j*x_phase)
    otf = I_mag * I_phase
    # image = torch.fft.irfft2(Image)
    otf = torch.fft.ifftshift(otf, dim=[-2, -1])

    # y = otf.clone()
    # print(otf[..., 1, 1], otf[..., -1, -1])
    # y[..., 1:, 1:] = torch.rot90(otf[..., 1:, 1:], -2, dims=[-2, -1])
    # print(y[..., 10, 0], otf[..., 10, 0])
    # otf[..., 1:, 1:] = (otf[..., 1:, 1:] + torch.conj(y[..., 1:, 1:])) / 2.

    kernel = convert_otf2psf(otf[..., :otf.shape[-1]//2+1], kernel_size, mode='rfft')
    # psf = torch.fft.irfft2(otf[..., :otf.shape[-1]//2+1])
    # # psf = torch.fft.ifftshift(psf, dim=[-2, -1])
    # for axis, axis_size in enumerate(psf.shape[2:]):
    #     psf = torch.roll(psf, int(axis_size / 2), dims=axis+2)
    return kernel, otf
def log_fft(image, add=1.):
    image_fft = torch.fft.fft2(image)
    image_fft_shift = torch.fft.fftshift(image_fft, dim=[-2, -1])
    # image_fft_shift = torch.cat([image_fft_shift.real, image_fft_shift.imag], dim=1)
    mag = torch.abs(image_fft_shift)
    phase = torch.angle(image_fft_shift)
    # image_fft_shift_s = image_fft_shift / (image_fft_shift_abs+1e-7)
    mag_log = torch.log(mag+add) # * image_fft_shift_s
    res = mag_log * torch.exp(1j*phase) / 10.
    return res.real, res.imag

def inverse_log_rfft(x_real, x_imag, sub=1.):
    x = torch.complex(x_real, x_imag) * 10.
    x_mag = torch.abs(x)
    x_phase = torch.angle(x)
    I_mag = torch.exp(x_mag) - sub

    I_phase = torch.exp(1j*x_phase)
    Image = I_mag * I_phase
    Image = torch.fft.ifftshift(Image, dim=[-2, -1])
    image = torch.fft.irfft2(Image[..., :Image.shape[-1]//2+1])
    return image

def inverse_log_deconv_rfft(x_real, x_imag, y_real, y_imag, sub=0.):
    x = torch.complex(x_real, x_imag) * 10.
    y = torch.complex(y_real, y_imag) * 10.
    x_mag = torch.abs(x)
    x_phase = torch.angle(x)

    y_mag = torch.abs(y)
    y_phase = torch.angle(y)
    # I_mag = (torch.exp(x_mag) - sub) / (torch.abs(torch.exp(y_mag) - sub) + 1.)
    I_mag = torch.exp(x_mag-y_mag) - sub

    I_phase = torch.exp(1j*(x_phase-y_phase))
    Image = I_mag * I_phase
    Image = torch.fft.ifftshift(Image, dim=[-2, -1])
    image = torch.fft.irfft2(Image[..., :Image.shape[-1]//2+1])
    return image


def convert_psf2otf_logarithmic_rfft(ker, size, add=math.exp(-2)):
    psf = torch.zeros(size, device=ker.device)
    # circularly shift

    # psf = torch.fft.fftshift(ker, dim=[-1, -2])
    centre = ker.shape[2]//2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    # compute the otf
    otf_mag_log, otf_phase = logarithmic_rfft(psf, add=add)
    # otf = torch.fft.fftn(psf, dim=[-3, -2, -1])
    return otf_mag_log, otf_phase
def convert_psf2otf_logarithmic(ker, size, add=math.exp(-10)):
    psf = torch.zeros(size, device=ker.device)
    # circularly shift
    # psf = torch.fft.fftshift(ker, dim=[-1, -2])
    centre = ker.shape[2]//2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    # compute the otf
    otf_mag, otf_mag_log, otf_phase = logarithmic_fft_clamp(psf, clamp_min=add)
    # otf = torch.fft.fftn(psf, dim=[-3, -2, -1])
    return otf_mag, otf_mag_log, otf_phase

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

def convert_otf2psf(otf, kernel_size=[31, 31], mode='rfft', norm='backward'):
    if mode == 'rfft':
        psf = torch.fft.irfft2(otf, norm=norm)
    else:
        psf = torch.fft.ifft2(otf, norm=norm).real
    psf = torch.fft.ifftshift(psf, dim=[-2, -1])
    psf = torch.nn.functional.pad(psf, (0, 1, 0, 1))
    ker = kornia.geometry.center_crop(psf, kernel_size)
    # ker = kornia.geometry.center_crop(psf[..., 1:, 1:], kernel_size)
    # for axis, axis_size in enumerate(psf.shape[-2:]):
    # psf = torch.roll(psf, int(kernel_size[0] / 2), dims=-2)
    # psf = torch.roll(psf, int(kernel_size[1] / 2), dims=-1)
    # ker = psf[..., :kernel_size[0], :kernel_size[1]]
    return ker
def convert_psf2otv2f(ker, size, mode='rfft'):
    # psf = torch.zeros(size, device=ker.device)
    # centre = ker.shape[2] // 2 + 1
    # psf[:, :, :centre, :centre] = ker[:, :, (centre - 1):, (centre - 1):]
    # psf[:, :, :centre, -(centre - 1):] = ker[:, :, (centre - 1):, :(centre - 1)]
    # psf[:, :, -(centre - 1):, :centre] = ker[:, :, : (centre - 1), (centre - 1):]
    # psf[:, :, -(centre - 1):, -(centre - 1):] = ker[:, :, :(centre - 1), :(centre - 1)]
    # circularly shift
    # psf = torch.fft.fftshift(ker, dim=[-1, -2])
    # print(ker.shape, size)
    psf = torch.zeros(size).type_as(ker)
    psf[..., :ker.shape[-2], :ker.shape[-1]].copy_(ker)
    for axis, axis_size in enumerate(ker.shape[-2:]):
        psf = torch.roll(psf, -int(axis_size / 2), dims=axis + 2)

    # compute the otf
    if mode == 'rfft':
        otf = torch.fft.rfft2(psf)
    else:
        otf = torch.fft.fft2(psf)
    # otf = torch.fft.fftn(psf, dim=[-3, -2, -1])
    return otf
def convert_otf2psfv2(otf, kernel_size=[31, 31], mode='rfft'):
    if mode == 'rfft':
        psf = torch.fft.irfft2(otf)
    else:
        psf = torch.fft.ifft2(otf).real

    # for axis, axis_size in enumerate(psf.shape[-2:]):
    psf = torch.roll(psf, int(kernel_size[0] / 2), dims=-2)
    psf = torch.roll(psf, int(kernel_size[1] / 2), dims=-1)
    ker = psf[..., :kernel_size[0], :kernel_size[1]]
    return ker
def convert_otf2psfv1(otf, mode='rfft'):
    # psf = torch.zeros(size, device=ker.device)
    # # circularly shift
    # # psf = torch.fft.fftshift(ker, dim=[-1, -2])

    # psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    # psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    # psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    # psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    if mode == 'rfft':
        psf = torch.fft.irfft2(otf)
    else:
        psf = torch.fft.ifft2(otf).real
    for axis, axis_size in enumerate(psf.shape[-2:]):
        psf = torch.roll(psf, int(axis_size / 2), dims=axis + 2)
    ker = psf
    # ker = torch.fft.fftshift(psf)
    # centre = psf.shape[2] // 2+1
    # ker = torch.zeros_like(psf)
    # ker[:, :, -centre:, -centre:] = psf[:, :, :centre, :centre]
    # ker[:, :, -centre:, :(centre-1)] = psf[:, :, :centre, -(centre-1):]
    # ker[:, :, : (centre-1), -centre:] = psf[:, :, -(centre-1):, :centre]
    # ker[:, :, :(centre-1), :(centre-1)] = psf[:, :, -(centre-1):, -(centre-1):]
    # compute the otf

    # otf = torch.fft.fftn(psf, dim=[-3, -2, -1])
    return ker
# def convert_psf2otf_log(ker, size, add=1.):
#     psf = torch.zeros(size, device=ker.device)
#     # circularly shift
#     # psf = torch.fft.fftshift(ker, dim=[-1, -2])
#     centre = ker.shape[2]//2 + 1
#     psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
#     psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
#     psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
#     psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
#     # compute the otf
#     otf_real_log, otf_imag_log = log_fft(psf, add=add)
#     # otf = torch.fft.fftn(psf, dim=[-3, -2, -1])
#     return otf_real_log, otf_imag_log
# def convert_psf2otf(psf, size):
#     # psf = torch.zeros(size, device=ker.device)
#     # # circularly shift
#     # # psf = torch.fft.fftshift(ker, dim=[-1, -2])
#     # centre = ker.shape[2]//2 + 1
#     # psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
#     # psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
#     # psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
#     # psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
#     otf = torch.zeros(size).type_as(psf)
#     otf[..., :psf.shape[2], :psf.shape[3]].copy_(psf)
#     for axis, axis_size in enumerate(psf.shape[2:]):
#         otf = torch.roll(otf, -int(axis_size / 2), dims=axis + 2)
#     # compute the otf
#     otf = torch.fft.fft2(otf)
#     # otf = torch.fft.fftn(psf, dim=[-3, -2, -1])
#     return otf

def get_ker(blur, sharp, ks=65, add=math.exp(-3), sub=0.):
    _, mag_log_blur, phase_blur = logarithmic_fft(blur, add=add)
    _, mag_log_sharp, phase_sharp = logarithmic_fft(sharp, add=add)
    mag_log_ker = mag_log_blur - mag_log_sharp
    phase_ker = phase_blur - phase_sharp
    # ker_blur, _ = inverse_logarithmic_irfft_kernel(mag_log_ker, phase_ker, sub=sub)
    ker_blur, _ = inverse_logarithmic_irfft_kernel(mag_log_ker, phase_ker, [ks, ks], sub=sub)
    # for axis, axis_size in enumerate(ker_blur.shape[2:]):
    #     ker_blur = torch.roll(ker_blur, int(axis_size / 2), dims=axis+2)
    # ker_blur = torch.fft.fftshift(ker_blur, dim=[-2, -1])
    # ker_blur = kornia.geometry.center_crop(ker_blur, [ks, ks])
    return ker_blur
def get_ker_(blur, sharp, ks=[65, 65], add=math.exp(-2), sub=0.):
    _, mag_log_blur, phase_blur = logarithmic_fft_clamp(blur, clamp_min=add)
    _, mag_log_sharp, phase_sharp = logarithmic_fft_clamp(sharp, clamp_min=add)
    mag_log_ker = mag_log_blur - mag_log_sharp
    phase_ker = phase_blur - phase_sharp
    # ker_blur, _ = inverse_logarithmic_ifft_kernel(mag_log_ker, phase_ker, ks, sub=sub)
    ker_blur, _ = inverse_logarithmic_ifft_kernel(mag_log_ker, phase_ker, ks, sub=sub)
    # for axis, axis_size in enumerate(ker_blur.shape[2:]):
    #     ker_blur = torch.roll(ker_blur, int(axis_size / 2), dims=axis+2)
    # ker_blur = torch.fft.fftshift(ker_blur, dim=[-2, -1])
    # ker_blur = kornia.geometry.center_crop(ker_blur, [ks, ks])
    # ker_blur = torch.clamp(ker_blur, 0., 1.)
    return ker_blur

def get_ker_with_mask(blur, sharp, mask=None, ks=65, add=math.exp(-5), sub=0.):
    if mask is not None:
        blur = torch.mean(blur, dim=1, keepdim=True)
        sharp = torch.mean(sharp, dim=1, keepdim=True)
        blur = blur * mask
        sharp = sharp * mask
    _, mag_log_blur, phase_blur = logarithmic_fft_clamp(blur, clamp_min=add)
    _, mag_log_sharp, phase_sharp = logarithmic_fft_clamp(sharp, clamp_min=add)
    mag_log_ker = mag_log_blur - mag_log_sharp
    phase_ker = phase_blur - phase_sharp
    # ker_blur, _ = inverse_logarithmic_irfft_kernel(mag_log_ker, phase_ker, sub=sub)
    ker_blur, _ = inverse_logarithmic_ifft_kernel(mag_log_ker, -phase_ker, sub=sub)
    # for axis, axis_size in enumerate(ker_blur.shape[2:]):
    #     ker_blur = torch.roll(ker_blur, int(axis_size / 2), dims=axis+2)
    # ker_blur = torch.fft.fftshift(ker_blur, dim=[-2, -1])
    # ker_blur = ker_blur.view()
    ker_blur = kornia.geometry.center_crop(ker_blur, [ks, ks])
    return ker_blur
def get_kerV2(dXh, dXv, dXeh, dXev, ks=65, add=1.):
    FdXh = torch.fft.rfft2(dXh)
    FdXv = torch.fft.rfft2(dXv)
    FdXeh = torch.fft.rfft2(dXeh)
    FdXev = torch.fft.rfft2(dXev)
    ker_blur = (torch.conj(FdXeh)*FdXh + torch.conj(FdXev)*FdXv) / (FdXev**2 + FdXev**2 + add)
    ker_blur = torch.fft.irfft2(ker_blur)
    ker_blur = torch.fft.fftshift(ker_blur, dim=[-2, -1])
    ker_blur = kornia.geometry.center_crop(ker_blur, [ks, ks])
    return ker_blur
if __name__=='__main__':
    import numpy as np
    import cv2
    import math
    import kornia
    # x = cv2.imread('/home/ubuntu/wy-3090-1/mxt_data/GoPro/val/target_crops_384/0.png')
    # x_tensor = kornia.image_to_tensor(x / 255., keepdim=False)
    # # a = torch.log(x_tensor+1.)
    # # b = torch.exp(torch.abs(a)) - 1
    # z = torch.fft.fft2(x_tensor)
    # mag = torch.abs(z)
    # phase = torch.angle(z)
    # y = mag * torch.exp(1j*phase)
    # y = torch.fft.ifft2(y).real
    # print(torch.sum(torch.abs(y-x_tensor)))
    # # z = torch.fft.fftshift(z, dim=[-2,-1])
    # z_ = z
    # z_[..., 1:, 1:] = torch.rot90(torch.conj(z[..., 1:, 1:]), -2, dims=[-2, -1])
    # i, j = 1, 1
    # # print(i, j)
    # # print(z[..., i, j])
    # # print(z[..., -i, -j])
    # # z_ = torch.flip(z_, dims=[-1])
    #
    # print(torch.sum(torch.abs(torch.abs(z_) - torch.abs(z))))
    # # print(z.shape)
    #
    # a, b = log_fft(x_tensor, add=1)
    # print(a.max(), a.min(), b.max(), b.min())
    # print(a.mean(), b.mean())
    # d2 = inverse_log_rfft(a, b, sub=1)
    # print(torch.sum(torch.abs(d2 - x_tensor)))
    #
    # a, b, c = logarithmic_fft(x_tensor, add=math.exp(-2))
    # d1 = inverse_logarithmic_fft(b, c, sub=math.exp(-2))
    # d2 = inverse_logarithmic_irfft(b, c, sub=math.exp(-2))
    # print(torch.sum(torch.abs(d1 - x_tensor)))
    # print(torch.sum(torch.abs(d2 - x_tensor)))
    # print(torch.sum(torch.abs(d2 - d1)))
    # print(d.max(), d.min())
    # x_img = kornia.tensor_to_image(d, keepdim=False)
    kernel = kornia.filters.get_motion_kernel2d(65, 215, -1)
    print(kernel.shape)
    # kernel = torch.rand_like(kernel) # torch.ones_like(kernel)
    # kernel[:, 32, 32] = 1.
    kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
    kernel = kernel.unsqueeze(1)
    _, x1, x2 = logarithmic_fft_kernel(kernel,[1, 1, 256, 256])
    print(x1.min(), x1.max(), x2.min(), x2.max())
    print(x2[:, :, :1, :], x2[:, :, :, :1])
    ker, _ = inverse_logarithmic_irfft_kernel(x1, x2, [65, 65])
    print(torch.sum(torch.abs(ker - kernel)))
    # x_tensor_blur = kornia.filters.filter2d(x_tensor, kernel)
    # b, c, h, w = x_tensor.shape
    # kernel = kernel.unsqueeze(0)
    # ks = max(kernel.shape[-1], kernel.shape[-2])
    # ks = ks // 2
    # print(ks)
    # # kernel = torch.nn.functional.pad(kernel)
    # # otf = torch.fft.fft2(kernel)
    # otf = convert_psf2otf(kernel, (b, 1, h + 2 * ks, w + 2 * ks))
    # dim = (ks, ks, ks, ks)
    # x_blur_ = torch.nn.functional.pad(x_tensor, dim, "reflect")
    # x_blur = torch.fft.fft2(x_blur_) * otf
    # x_blur_ = torch.fft.ifft2(x_blur).real
    # x_blur = x_blur_[:, :, ks:-ks, ks:-ks]
    # print(torch.mean(torch.abs(x_blur - x_tensor_blur)))
    #
    # ker_1 = get_ker(x_tensor_blur, x_tensor, ks=31, add=math.exp(-5))
    # ker_2 = get_ker_(x_tensor_blur, x_tensor, ks=31, add=math.exp(-5))
    # ker_1 = torch.mean(ker_1, dim=1, keepdim=True)
    # ker_2 = torch.mean(ker_2, dim=1, keepdim=True)
    # kernel_size = 31
    # k_len = 31
    # ker_1 = ker_1.view(-1, kernel_size * kernel_size)
    # _, idx = torch.topk(ker_1, k=k_len, dim=-1)
    # z = torch.zeros_like(ker_1)
    # z[:, idx] = 1.
    # ker_1 = ker_1 * z
    # ker_1 = ker_1.view(-1, 1, kernel_size, kernel_size)
    #
    # ker_2 = ker_2.view(-1, kernel_size * kernel_size)
    # _, idx = torch.topk(ker_2, k=k_len, dim=-1)
    # z = torch.zeros_like(ker_2)
    # z[:, idx] = 1.
    # ker_2 = ker_2 * z
    # ker_2 = ker_2.view(-1, 1, kernel_size, kernel_size)
    # print('kernel1: ', torch.sum(torch.abs(ker_1 - kernel)))
    # print('kernel2: ', torch.sum(torch.abs(ker_2 - kernel)))
    # add = math.exp(-2)
    # otf_mag, otf_mag_log, otf_phase = convert_psf2otf_logarithmic(kernel, (b, 1, h + 2 * ks, w + 2 * ks), add=add)
    # x_mag, x_mag_log, x_phase = logarithmic_fft(x_blur_, add=add)
    # print(otf_mag_log.max(), otf_mag_log.min(), x_mag_log.max(), x_mag_log.min())
    # mag = x_mag_log + otf_mag_log
    # phase = x_phase + otf_phase
    # y_blur = inverse_logarithmic_fft(mag, phase, sub=0)[:, :, ks:-ks, ks:-ks]
    # print('blur: ', torch.mean(torch.abs(y_blur - x_tensor_blur)))
    # print(y_blur.max(), y_blur.min(), x_tensor_blur.max(), x_tensor_blur.min())

    # add = math.exp(-2)
    # otf_mag, otf_mag_log, otf_phase = convert_psf2otf_logarithmic(kernel, (b, 1, h + 2 * ks, w + 2 * ks), add=add)
    # x_blur_ = torch.nn.functional.pad(x_tensor_blur, dim, "reflect")
    # # otf_mag, otf_mag_log, otf_phase = convert_psf2otf_logarithmic(kernel, (b, 1, h, w), add=add)
    # # x_blur_ = x_tensor_blur
    # x_mag, x_mag_log, x_phase = logarithmic_fft(x_blur_, add=add)
    # print(otf_mag_log.max(), otf_mag_log.min(), x_mag_log.max(), x_mag_log.min())
    # mag = x_mag_log - otf_mag_log
    # # mag = torch.abs(mag) # + math.exp(-3)
    # phase = x_phase - otf_phase
    # y_deblur = inverse_logarithmic_fft(mag, phase, sub=0)[:, :, ks:-ks, ks:-ks]
    # print(torch.mean(torch.abs(y_deblur - x_tensor)))
    # print(y_deblur.max(), y_deblur.min(), x_tensor.max(), x_tensor.min())
    # add = 1.
    # otf_real, otf_imag = convert_psf2otf_log(kernel, (b, 1, h + 2 * ks, w + 2 * ks), add=add)
    # x_blur_ = torch.nn.functional.pad(x_tensor_blur, dim, "reflect")
    # # otf_mag, otf_mag_log, otf_phase = convert_psf2otf_logarithmic(kernel, (b, 1, h, w), add=add)
    # # x_blur_ = x_tensor_blur
    # x_real, x_imag = log_fft(x_blur_, add=add)
    # print(otf_mag_log.max(), otf_mag_log.min(), x_mag_log.max(), x_mag_log.min())
    # # mag = x_mag_log - otf_mag_log
    # # # mag = torch.abs(mag) # + math.exp(-3)
    # # phase = x_phase - otf_phase
    # y_deblur = inverse_log_deconv_rfft(x_real, x_imag, otf_real, otf_imag, sub=0.)[:, :, ks:-ks, ks:-ks]
    # print(torch.mean(torch.abs(y_deblur - x_tensor)))
    # print(y_deblur.max(), y_deblur.min(), x_tensor.max(), x_tensor.min())
    # # 显示结果
    # plt.figure(figsize=(12, 4))
    # # plt.subplot(131); plt.imshow(x, cmap='gray'); plt.title('Original Image')
    # # plt.subplot(132); plt.imshow(kornia.tensor_to_image(x_tensor_blur), cmap='gray'); plt.title('x_tensor_blur')
    # # plt.subplot(133); plt.imshow(kornia.tensor_to_image(y_deblur), cmap='gray'); plt.title('Reconstructed Image')
    # plt.subplot(131); plt.imshow(kornia.tensor_to_image(kernel), cmap='gray'); plt.title('Original Image')
    # plt.subplot(132); plt.imshow(kornia.tensor_to_image(ker_1), cmap='gray'); plt.title('x_tensor_blur')
    # plt.subplot(133); plt.imshow(kornia.tensor_to_image(ker_2), cmap='gray'); plt.title('Reconstructed Image')
    # plt.show()