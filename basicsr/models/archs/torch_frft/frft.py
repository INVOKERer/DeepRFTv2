#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 14:26:30 2024

@author: dl
"""

import torch
import math


# def dfrtmtrx(N, a):
#     # Approximation order
#     app_ord = 2
#     Evec = dis_s(N,app_ord).cuda()
#     Evec = Evec.to(dtype=torch.complex64)
#     even = 1 - (N%2)
#     l = torch.tensor(list(range(0,N-1)) + [N-1+even]).cuda()
#     f = torch.diag(torch.exp(-1j*math.pi/2*a*l))
#     F1 = N**(1/2)*torch.einsum("ij,jk,ni->nk", f, Evec.T, Evec)
#     return F1

# def dis_s(N, app_ord):  
#     app_ord = int(app_ord / 2) 
#     s = torch.cat((torch.tensor([0, 1]), torch.zeros(N-1-2*app_ord), torch.tensor([1])))
#     S = cconvm(N,s) + torch.diag((torch.fft.fft(s)).real);

#     p = N
#     r = math.floor(N/2)
#     P = torch.zeros((p,p))
#     P[0,0] = 1
#     even = 1 - (p%2)
    
#     for i in range(1,r-even+1):
#         P[i,i] = 1/(2**(1/2))
#         P[i,p-i] = 1/(2**(1/2))
        
#     if even:
#         P[r,r] = 1
        
#     for i in range(r+1,p):
#         P[i,i] = -1/(2**(1/2))
#         P[i,p-i] = 1/(2**(1/2))

#     CS = torch.einsum("ij,jk,ni->nk", S, P.T, P)
#     C2 = CS[0:math.floor(N/2+1), 0:math.floor(N/2+1)]
#     S2 = CS[math.floor(N/2+1):N, math.floor(N/2+1):N]
#     ec, vc = torch.linalg.eig(C2)
#     es, vs = torch.linalg.eig(S2)
#     ec = ec.real
#     vc = vc.real
#     es = es.real
#     vs = vs.real
#     qvc = torch.vstack((vc, torch.zeros([math.ceil(N/2-1), math.floor(N/2+1)])))
#     SC2 = P@qvc # Even Eigenvector of S
#     qvs = torch.vstack((torch.zeros([math.floor(N/2+1), math.ceil(N/2-1)]),vs))
#     SS2 = P@qvs # Odd Eigenvector of S
#     idx = torch.argsort(-ec)
#     SC2 = SC2[:,idx]
#     idx = torch.argsort(-es)
#     SS2 = SS2[:,idx]
    
#     if N%2 == 0:
#         S2C2 = torch.zeros([N,N+1])
#         SS2 = torch.hstack([SS2, torch.zeros((SS2.shape[0],1))])
#         S2C2[:,range(0,N+1,2)] = SC2;
#         S2C2[:,range(1,N,2)] = SS2
#         S2C2 = S2C2[:, torch.arange(S2C2.size(1)) != N-1]
#     else:
#         S2C2 = torch.zeros([N,N])
#         S2C2[:,range(0,N+1,2)] = SC2;
#         S2C2[:,range(1,N,2)] = SS2
    
#     return S2C2 

# def cconvm(N, s):
#     M = torch.zeros((N,N))
#     dum = s
#     for i in range(N):
#         M[:,i] = dum
#         dum = torch.roll(dum,1)
#     return M

# def FRFT2D(matrix, order):
#     N, C, H, W = matrix.shape
#     h_test = dfrtmtrx(H, order).cuda()
#     w_test = dfrtmtrx(W, order).cuda()
#     h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=C, dim=0)
#     h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=N, dim=0)
#     w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=C, dim=0)
#     w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=N, dim=0)

#     out = []
#     matrix = torch.fft.fftshift(matrix, dim=(2, 3)).to(dtype=torch.complex64)

#     out = torch.matmul(h_test, matrix)
#     out = torch.matmul(out, w_test)

#     out = torch.fft.fftshift(out, dim=(2, 3))
#     return out

# def IFRFT2D(matrix, order):
#     N, C, H, W = matrix.shape
#     h_test = dfrtmtrx(H, order).cuda()
#     w_test = dfrtmtrx(W, order).cuda()
#     h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=C, dim=0)
#     h_test = torch.repeat_interleave(h_test.unsqueeze(dim=0), repeats=N, dim=0)
#     w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=C, dim=0)
#     w_test = torch.repeat_interleave(w_test.unsqueeze(dim=0), repeats=N, dim=0)

#     out = []
#     matrix = torch.fft.fftshift(matrix, dim=(2, 3)).to(dtype=torch.complex64)
    
#     out = torch.matmul(h_test, matrix)
#     out = torch.matmul(out, w_test)

#     out = torch.fft.fftshift(out, dim=(2, 3))
#     return out
import torch
from math import ceil
# device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")

# def FRFT2D(x: torch.Tensor, a: float) -> torch.Tensor:
#     # Apply 1D DFRFT along the last dimension (columns)
#     x_transformed = dfrft(x, a, dim=-1)
#     # Apply 1D DFRFT along the second to last dimension (rows)
#     x_transformed = dfrft(x_transformed, a, dim=-2)
#     return x_transformed

# def IFRFT2D(x: torch.Tensor, a: float) -> torch.Tensor:
#     # Apply 1D IDFRFT along the last dimension (columns)
#     x_transformed = idfrft(x, a, dim=-1)
#     # Apply 1D IDFRFT along the second to last dimension (rows)
#     x_transformed = idfrft(x_transformed, a, dim=-2)
#     return x_transformed

# def dfrft(x: torch.Tensor, a: float, *, dim: int = -1) -> torch.Tensor:
#     dfrft_matrix = dfrftmtx(x.size(dim), a, device=x.device)
#     dtype = torch.promote_types(dfrft_matrix.dtype, x.dtype)
#     return torch.einsum(
#         _get_dfrft_einsum_str(len(x.shape), dim),
#         dfrft_matrix.type(dtype),
#         x.type(dtype),
#     )

# def idfrft(x: torch.Tensor, a: float, *, dim: int = -1) -> torch.Tensor:
#     return dfrft(x, -a, dim=dim)

# def _get_dfrft_einsum_str(dim_count: int, req_dim: int) -> str:
#     if req_dim < -dim_count or req_dim >= dim_count:
#         raise ValueError("Dimension size error.")
#     dim = torch.remainder(req_dim, torch.tensor(dim_count))
#     diff = dim_count - dim
#     remaining_str = "".join([chr(num) for num in range(98, 98 + diff)])
#     return f"ab,...{remaining_str}->...{remaining_str.replace('b', 'a', 1)}"

# def dfrftmtx(N: int, a: float, *, approx_order: int = 2, device: torch.device = device) -> torch.Tensor:
#     if N < 1 or approx_order < 2:
#         raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")
#     evecs = _get_dfrft_evecs(N, approx_order=approx_order, device=device).type(torch.complex64)
#     idx = _dfrft_index(N, device=device)
#     evals = torch.exp(-1j * a * (torch.pi / 2) * idx).type(torch.complex64)
#     dfrft_matrix = torch.einsum("ij,j,kj->ik", evecs, evals, evecs)
#     return dfrft_matrix

# def _get_dfrft_evecs(N: int, *, approx_order: int = 2, device: torch.device = device) -> torch.Tensor:
#     if N < 1 or approx_order < 2:
#         raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")
#     H = _create_hamiltonian(N, approx_order=approx_order, device=device)
#     evals, evecs = torch.linalg.eigh(H)
#     return evecs

# def _circulant(x: torch.Tensor) -> torch.Tensor:
#     N = x.numel()
#     x = x.flip(0).tile(N)
#     return x[torch.arange(N).unsqueeze(1) + torch.arange(N)]

# def _dfrft_index(N: int, *, device: torch.device = device) -> torch.Tensor:
#     return torch.cat((torch.arange(ceil(N / 2), device=device), torch.arange(-N // 2, 0, device=device)))

# def _conv1d_full(vector: torch.Tensor, kernel1d: torch.Tensor) -> torch.Tensor:
#     padding_size = kernel1d.size(0) - 1
#     padded_input = torch.nn.functional.pad(vector, (padding_size, padding_size), mode="constant", value=0)
#     conv_output = torch.conv1d(padded_input.view(1, 1, -1), kernel1d.view(1, 1, -1).flip(-1))
#     return conv_output.reshape(-1)

# def _create_hamiltonian(N: int, *, approx_order: int = 2, device: torch.device = device) -> torch.Tensor:
#     if N < 1 or approx_order < 2:
#         raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")
#     order = approx_order // 2
#     dum0 = torch.tensor([1.0, -2.0, 1.0], device=device)
#     dum = dum0.clone()
#     s = torch.zeros(1, device=device)
#     for k in range(1, order + 1):
#         coefficient = (2 * (-1) ** (k - 1) * torch.prod(torch.arange(1, k, device=device)) ** 2 / torch.prod(torch.arange(1, 2 * k + 1, device=device)))
#         s = (coefficient * torch.cat((torch.zeros(1, device=device), dum[k + 1 : 2 * k + 1], torch.zeros(N - 1 - 2 * k, device=device), dum[:k])) + s)
#         dum = _conv1d_full(dum, dum0)
#     return _circulant(s) + torch.diag(torch.real(torch.fft.fft(s)))

# def _create_odd_even_decomp_matrix(N: int, *, device: torch.device = device) -> torch.Tensor:
#     if N < 1:
#         raise ValueError("N must be positive integer.")
#     x1 = torch.ones(1 + N // 2, dtype=torch.float32, device=device)
#     x2 = -torch.ones(N - N // 2 - 1, dtype=torch.float32, device=device)
#     diagonal = torch.diag(torch.cat((x1, x2)))
#     anti = torch.diag(torch.ones(N - 1, device=device), -1).rot90()
#     P = (diagonal + anti) / torch.sqrt(torch.tensor(2.0))
#     P[0, 0] = 1
#     if N % 2 == 0:
#         P[N // 2, N // 2] = 1
#     return P

def FRFT2D(x: torch.Tensor, a: float) -> torch.Tensor:
    # Apply 1D DFRFT along the last dimension (columns)
    x_transformed = dfrft(x, a, dim=-1)
    # Apply 1D DFRFT along the second to last dimension (rows)
    x_transformed = dfrft(x_transformed, a, dim=-2)
    return x_transformed

def IFRFT2D(x: torch.Tensor, a: float) -> torch.Tensor:
    # Apply 1D IDFRFT along the last dimension (columns)
    x_transformed = idfrft(x, a, dim=-1)
    # Apply 1D IDFRFT along the second to last dimension (rows)
    x_transformed = idfrft(x_transformed, a, dim=-2)
    return x_transformed

def dfrft(x: torch.Tensor, a: float, *, dim: int = -1) -> torch.Tensor:
    dfrft_matrix = dfrftmtx(x.size(dim), a, device=x.device)
    dtype = torch.promote_types(dfrft_matrix.dtype, x.dtype)
    return torch.einsum(
        _get_dfrft_einsum_str(len(x.shape), dim),
        dfrft_matrix.type(dtype),
        x.type(dtype),
    )

def idfrft(x: torch.Tensor, a: float, *, dim: int = -1) -> torch.Tensor:
    return dfrft(x, -a, dim=dim)

def _get_dfrft_einsum_str(dim_count: int, req_dim: int) -> str:
    if req_dim < -dim_count or req_dim >= dim_count:
        raise ValueError("Dimension size error.")
    dim = torch.remainder(req_dim, torch.tensor(dim_count))
    diff = dim_count - dim
    remaining_str = "".join([chr(num) for num in range(98, 98 + diff)])
    return f"ab,...{remaining_str}->...{remaining_str.replace('b', 'a', 1)}"

def dfrftmtx(N: int, a: float, *, approx_order: int = 2, device: torch.device) -> torch.Tensor:
    if N < 1 or approx_order < 2:
        raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")
    evecs = _get_dfrft_evecs(N, approx_order=approx_order, device=device).type(torch.complex64)
    idx = _dfrft_index(N, device=device)
    evals = torch.exp(-1j * a * (torch.pi / 2) * idx).type(torch.complex64)
    dfrft_matrix = torch.einsum("ij,j,kj->ik", evecs, evals, evecs)
    return dfrft_matrix

def _get_dfrft_evecs(N: int, *, approx_order: int = 2, device: torch.device) -> torch.Tensor:
    if N < 1 or approx_order < 2:
        raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")
    H = _create_hamiltonian(N, approx_order=approx_order, device=device)
    evals, evecs = torch.linalg.eigh(H)
    return evecs

def _circulant(x: torch.Tensor) -> torch.Tensor:
    N = x.numel()
    x = x.flip(0).tile(N)
    return x[torch.arange(N).unsqueeze(1) + torch.arange(N)]

def _dfrft_index(N: int, *, device: torch.device) -> torch.Tensor:
    return torch.cat((torch.arange(ceil(N / 2), device=device), torch.arange(-N // 2, 0, device=device)))

def _conv1d_full(vector: torch.Tensor, kernel1d: torch.Tensor) -> torch.Tensor:
    padding_size = kernel1d.size(0) - 1
    padded_input = torch.nn.functional.pad(vector, (padding_size, padding_size), mode="constant", value=0)
    conv_output = torch.conv1d(padded_input.view(1, 1, -1), kernel1d.view(1, 1, -1).flip(-1))
    return conv_output.reshape(-1)

def _create_hamiltonian(N: int, *, approx_order: int = 2, device: torch.device) -> torch.Tensor:
    if N < 1 or approx_order < 2:
        raise ValueError("Necessary conditions for integers: N > 1 and approx_order >= 2.")
    order = approx_order // 2
    dum0 = torch.tensor([1.0, -2.0, 1.0], device=device)
    dum = dum0.clone()
    s = torch.zeros(1, device=device)
    for k in range(1, order + 1):
        coefficient = (2 * (-1) ** (k - 1) * torch.prod(torch.arange(1, k, device=device)) ** 2 / torch.prod(torch.arange(1, 2 * k + 1, device=device)))
        s = (coefficient * torch.cat((torch.zeros(1, device=device), dum[k + 1 : 2 * k + 1], torch.zeros(N - 1 - 2 * k, device=device), dum[:k])) + s)
        dum = _conv1d_full(dum, dum0)
    return _circulant(s) + torch.diag(torch.real(torch.fft.fft(s)))

def _create_odd_even_decomp_matrix(N: int, *, device: torch.device ) -> torch.Tensor:
    if N < 1:
        raise ValueError("N must be positive integer.")
    x1 = torch.ones(1 + N // 2, dtype=torch.float32, device=device)
    x2 = -torch.ones(N - N // 2 - 1, dtype=torch.float32, device=device)
    diagonal = torch.diag(torch.cat((x1, x2)))
    anti = torch.diag(torch.ones(N - 1, device=device), -1).rot90()
    P = (diagonal + anti) / torch.sqrt(torch.tensor(2.0))
    P[0, 0] = 1
    if N % 2 == 0:
        P[N // 2, N // 2] = 1
    return P
def convert_ker2psf(ker, size):

    centre = [size[-2] // 2, size[-1] // 2]
    ker_centre  = [ker.shape[-2] // 2, ker.shape[-1] // 2]
    psf = torch.zeros(size).type_as(ker)
    psf[..., centre[0]-ker_centre[0]:centre[0]+ker_centre[0]+1, centre[1]-ker_centre[1]:centre[1]+ker_centre[1]+1].copy_(ker)
    psf = torch.fft.fftshift(psf, dim=[-1, -2])

    return psf
if __name__=='__main__':
    import numpy as np
    import cv2
    import math
    import kornia
    x = cv2.imread('/data/mxt_data/GoPro/val/input_crops_384/0.png')
    x = x / 255.
    x_tensor = kornia.image_to_tensor(x, keepdim=False)
    b, c, h, w = x_tensor.shape # [-2:]
    a_param = 0.5
    F = FRFT2D(x_tensor, a_param)
    z = torch.abs(F)
    z_ = torch.clamp(z, 0., 1.)*255

    cv2.imwrite('/data/mxt_code/DeepMXT/Motion_Deblurring/kernels/0_frft.png',  kornia.tensor_to_image(z_, keepdim=False))
    zx = torch.fft.ifftshift(z_, dim=[-2, -1])
    cv2.imwrite('/data/mxt_code/DeepMXT/Motion_Deblurring/kernels/0_frft_shift.png',  kornia.tensor_to_image(zx, keepdim=False))
    print(F.shape, z.max(), z.min())
    IF = IFRFT2D(F, a_param)
    print(torch.mean(torch.abs(torch.abs(IF) - x_tensor)))

    kernel = kornia.filters.get_motion_kernel2d(65, 215, -1)
    print(kernel.shape)
    # kernel = torch.rand_like(kernel) # torch.ones_like(kernel)
    # kernel[:, 32, 32] = 1.sssssssssss
    kernel = kernel / torch.sum(kernel, dim=[-2, -1], keepdim=True)
    kernel = kernel.unsqueeze(1)
    psf = convert_ker2psf(kernel, (b, c, h,w))
    # kernel = torch.rand_like(x_tensor)
    k = FRFT2D(psf, a_param)
    k_fft = torch.fft.fft2(psf)
    y_fft = torch.fft.fft2(x_tensor)
    y_mag = torch.abs(y_fft)
    F_mag = torch.abs(F)
    print(y_mag.max(), F_mag.max())
    y_fft_k = torch.fft.ifft2(k_fft * y_fft)

    deblur_f = torch.zeros_like(F)
    deblur_f.real = k.real * F.real \
                    - k.imag * F.imag
    deblur_f.imag = k.real * F.imag \
                    + k.imag * F.real
    y_k = IFRFT2D(deblur_f, a_param)
    a = torch.abs(y_fft_k) * 255.
    b = torch.abs(y_k) * 255.
    b = b * math.sqrt(h * w)
    print(b.max(), b.min())
    cv2.imwrite('/data/mxt_code/DeepMXT/Motion_Deblurring/kernels/0_fft_blur.png',
                kornia.tensor_to_image(a, keepdim=False))
    cv2.imwrite('/data/mxt_code/DeepMXT/Motion_Deblurring/kernels/0_frft_blur.png',
                kornia.tensor_to_image(b, keepdim=False))
    # x_ = IF(y)
    # print(torch.allclose(x_.real.to(x_tensor.type), x_tensor))
    # print(torch.mean(torch.abs(y_fft - y)))
    # print(torch.mean(torch.abs(x_.real - x_tensor)))
    print(torch.mean(torch.abs(torch.abs(y_k) - torch.abs(y_fft_k))))