# import numpy as np
import torch
import matplotlib.pyplot as plt
import math

def logarithmic_fft(image, add=math.exp(-2)):
    image_fft = torch.fft.fft2(image)
    image_fft_shift = torch.fft.fftshift(image_fft)
    # image_fft_shift = torch.cat([image_fft_shift.real, image_fft_shift.imag], dim=1)
    mag = torch.abs(image_fft_shift)
    phase = torch.angle(image_fft_shift) / torch.pi
    # image_fft_shift_s = image_fft_shift / (image_fft_shift_abs+1e-7)
    mag_log = torch.log(mag+add) # * image_fft_shift_s
    return mag, mag_log, phase
def inverse_logarithmic_fft(x_mag, x_phase, sub=0):

    I_mag = torch.exp(torch.abs(x_mag)) - sub

    I_phase = torch.exp(1j*x_phase*torch.pi)
    Image = I_mag * I_phase
    Image = torch.fft.ifftshift(Image)
    image = torch.fft.ifft2(Image).real
    return image

def convert_psf2otf_logarithmic(ker, size, add=1.):
    psf = torch.zeros(size, device=ker.device)
    # circularly shift
    # psf = torch.fft.fftshift(ker, dim=[-1, -2])
    centre = ker.shape[2]//2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    # compute the otf
    otf_mag, otf_mag_log, otf_phase = logarithmic_fft(psf, add=add)
    # otf = torch.fft.fftn(psf, dim=[-3, -2, -1])
    return otf_mag, otf_mag_log, otf_phase
def convert_psf2otf(ker, size):
    psf = torch.zeros(size, device=ker.device)
    # circularly shift
    # psf = torch.fft.fftshift(ker, dim=[-1, -2])
    centre = ker.shape[2]//2 + 1
    psf[:, :, :centre, :centre] = ker[:, :, (centre-1):, (centre-1):]
    psf[:, :, :centre, -(centre-1):] = ker[:, :, (centre-1):, :(centre-1)]
    psf[:, :, -(centre-1):, :centre] = ker[:, :, : (centre-1), (centre-1):]
    psf[:, :, -(centre-1):, -(centre-1):] = ker[:, :, :(centre-1), :(centre-1)]
    # compute the otf
    otf = torch.fft.fft2(psf)
    # otf = torch.fft.fftn(psf, dim=[-3, -2, -1])
    return otf

if __name__=='__main__':
    import numpy as np
    import cv2
    import math
    import kornia
    x = cv2.imread('/home/ubuntu/wy-3090-1/mxt_data/GoPro/val/target_crops_384/0.png')
    x_tensor = kornia.image_to_tensor(x / 255., keepdim=False)
    # a = torch.log(x_tensor+1.)
    # b = torch.exp(torch.abs(a)) - 1

    a, b, c = logarithmic_fft(x_tensor, add=1)
    d = inverse_logarithmic_fft(b, c, sub=1)

    print(torch.sum(torch.abs(d - x_tensor)))
    # print(d.max(), d.min())
    # x_img = kornia.tensor_to_image(d, keepdim=False)
    kernel = kornia.filters.get_motion_kernel2d(31, angle=21)
    print(kernel.shape)
    kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
    x_tensor_blur = kornia.filters.filter2d(x_tensor, kernel)
    b, c, h, w = x_tensor.shape
    kernel = kernel.unsqueeze(0)
    ks = max(kernel.shape[-1], kernel.shape[-2])
    ks = ks // 2
    print(ks)
    # kernel = torch.nn.functional.pad(kernel)
    # otf = torch.fft.fft2(kernel)
    otf = convert_psf2otf(kernel, (b, 1, h + 2 * ks, w + 2 * ks))
    dim = (ks, ks, ks, ks)
    x_blur_ = torch.nn.functional.pad(x_tensor, dim, "reflect")
    x_blur = torch.fft.fft2(x_blur_) * otf
    x_blur_ = torch.fft.ifft2(x_blur).real
    x_blur = x_blur_[:, :, ks:-ks, ks:-ks]
    print(torch.mean(torch.abs(x_blur - x_tensor_blur)))

    add = math.exp(-2)
    otf_mag, otf_mag_log, otf_phase = convert_psf2otf_logarithmic(kernel, (b, 1, h + 2 * ks, w + 2 * ks), add=add)
    x_mag, x_mag_log, x_phase = logarithmic_fft(x_blur_, add=add)
    print(otf_mag_log.max(), otf_mag_log.min(), x_mag_log.max(), x_mag_log.min())
    mag = x_mag_log+otf_mag_log
    phase = x_phase+otf_phase
    y_blur = inverse_logarithmic_fft(mag, phase, sub=0)[:, :, ks:-ks, ks:-ks]
    print(torch.mean(torch.abs(y_blur - x_tensor_blur)))
    print(y_blur.max(), y_blur.min(), x_tensor_blur.max(), x_tensor_blur.min())

    add = math.exp(-2)
    otf_mag, otf_mag_log, otf_phase = convert_psf2otf_logarithmic(kernel, (b, 1, h + 2 * ks, w + 2 * ks), add=add)
    x_blur_ = torch.nn.functional.pad(x_tensor_blur, dim, "reflect")
    # otf_mag, otf_mag_log, otf_phase = convert_psf2otf_logarithmic(kernel, (b, 1, h, w), add=add)
    # x_blur_ = x_tensor_blur
    x_mag, x_mag_log, x_phase = logarithmic_fft(x_blur_, add=add)
    print(otf_mag_log.max(), otf_mag_log.min(), x_mag_log.max(), x_mag_log.min())
    mag = x_mag_log - otf_mag_log
    # mag = torch.abs(mag) # + math.exp(-3)
    phase = x_phase - otf_phase
    y_deblur = inverse_logarithmic_fft(mag, phase, sub=0)[:, :, ks:-ks, ks:-ks]
    print(torch.mean(torch.abs(y_deblur - x_tensor)))
    print(y_deblur.max(), y_deblur.min(), x_tensor.max(), x_tensor.min())
    # 显示结果
    plt.figure(figsize=(12, 4))
    plt.subplot(131); plt.imshow(x, cmap='gray'); plt.title('Original Image')
    plt.subplot(132); plt.imshow(kornia.tensor_to_image(x_tensor_blur), cmap='gray'); plt.title('x_tensor_blur')
    plt.subplot(133); plt.imshow(kornia.tensor_to_image(y_deblur), cmap='gray'); plt.title('Reconstructed Image')
    plt.show()