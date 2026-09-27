## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
from skimage.metrics import structural_similarity
from tqdm import tqdm
import concurrent.futures
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from basicsr.metrics import calculate_ssim
def image_align(deblurred, gt):
  # this function is based on kohler evaluation code
  z = deblurred
  c = np.ones_like(z)
  x = gt

  zs = (np.sum(x * z) / np.sum(z * z)) * z # simple intensity matching

  warp_mode = cv2.MOTION_HOMOGRAPHY
  warp_matrix = np.eye(3, 3, dtype=np.float32)

  # Specify the number of iterations.
  number_of_iterations = 100

  termination_eps = 0

  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
              number_of_iterations, termination_eps)

  # Run the ECC algorithm. The results are stored in warp_matrix.
  (cc, warp_matrix) = cv2.findTransformECC(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cv2.cvtColor(zs, cv2.COLOR_RGB2GRAY), warp_matrix, warp_mode, criteria, inputMask=None, gaussFiltSize=5)

  target_shape = x.shape
  shift = warp_matrix

  zr = cv2.warpPerspective(
    zs,
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_CUBIC+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_REFLECT)

  cr = cv2.warpPerspective(
    np.ones_like(zs, dtype='float32'),
    warp_matrix,
    (target_shape[1], target_shape[0]),
    flags=cv2.INTER_NEAREST+ cv2.WARP_INVERSE_MAP,
    borderMode=cv2.BORDER_CONSTANT,
    borderValue=0)

  zr = zr * cr
  xr = x * cr

  return zr, xr, cr, shift

def compute_psnr(image_true, image_test, image_mask, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64) / np.sum(image_mask)
  return 10 * np.log10((data_range ** 2) / err)

def compute_psnr_(image_true, image_test, data_range=None):
  # this function is based on skimage.metrics.peak_signal_noise_ratio
  err = np.sum((image_true - image_test) ** 2, dtype=np.float64)
  return 10 * np.log10((data_range ** 2) / err)
def compute_ssim(tar_img, prd_img, cr1):
    ssim_pre, ssim_map = structural_similarity(tar_img, prd_img, channel_axis=2, gaussian_weights=True,
                                               use_sample_covariance=False, data_range=1.0, full=True)
    ssim_map = ssim_map * cr1
    r = int(3.5 * 1.5 + 0.5)  # radius as in ndimage
    win_size = 2 * r + 1
    pad = (win_size - 1) // 2
    ssim = ssim_map[pad:-pad,pad:-pad,:]
    crop_cr1 = cr1[pad:-pad,pad:-pad,:]
    ssim = ssim.sum(axis=0).sum(axis=0)/crop_cr1.sum(axis=0).sum(axis=0)
    ssim = np.mean(ssim)
    return ssim

def proc(filename):
    tar,prd = filename
    tar_img = io.imread(tar)
    prd_img = io.imread(prd)
    # h_gt, w_gt = tar_img.shape[:2]
    # h, w = prd_img.shape[:2]
    # h_pad = max(0, h - h_gt)
    # w_pad = max(0, w - w_gt)
    # h_pad, w_pad = 64, 64
    # # print(prd_img.shape, tar_img.shape)
    # pad_mode = 'center'
    # if pad_mode == 'center':
    #     h1 = h_pad//2
    #     w1 = w_pad // 2
    #     prd_img = prd_img[h1:h_gt+h1, w1:w_gt+w1, :]
    # else:
    #     prd_img = prd_img[:h_gt, :w_gt, :]
    # print(prd_img.shape, tar_img.shape)
    SSIM_ = 0 # calculate_ssim(prd_img, tar_img, crop_border=0)
    PSNR_ = psnr_loss(prd_img, tar_img)
    tar_img = tar_img.astype(np.float32)/255.0
    prd_img = prd_img.astype(np.float32)/255.0


    try:
        prd_img, tar_img, cr1, shift = image_align(prd_img, tar_img)
    except:
        print('x: ' + prd.split('/')[-1])
        cr1 = np.ones_like(prd_img)
    # print(prd_img.shape, tar_img.shape, cr1.shape, cr1.max(), cr1.min())
    PSNR = compute_psnr(tar_img, prd_img, cr1, data_range=1)
    # print(tar_img.shape, prd_img.shape)
    SSIM = compute_ssim(tar_img, prd_img, cr1)

    return [PSNR, SSIM, PSNR_, SSIM_]

datasets = ['RealBlur_R', 'RealBlur_J']
# datasets = ['RealBlur_R']
# datasets = ['RealBlur_J']
for dataset in datasets:

    # file_path = os.path.join('/home/ubuntu/90t/personal_data/mxt/MXT/exp_results/RevD-L_RealBlur', dataset)
    file_path = os.path.join('/home/ubuntu/150t/personal_data/mxt/MXT/exp_results/DeepRFTv2', dataset)
    # file_path = os.path.join('/home/ubuntu/90t/personal_data/mxt/MXT/exp_results/AdaRevD/AdaRevD-L_RealBlur_0.05', dataset)

    print(file_path)
    gt_path = os.path.join('/home/ubuntu/90t/personal_data/mxt/Dataset/MotionBlur/RealBlur', dataset, 'test/sharp')

    path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
    gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))
    print(len(path_list), len(gt_list))
    assert len(path_list) != 0, "Predicted files not found"
    assert len(gt_list) != 0, "Target files not found"

    psnr, ssim, psnr_, ssim_ = [], [], [], []
    img_files =[(i, j) for i,j in zip(gt_list,path_list)]
    # try:
    # with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    # for filename in img_files:
    #     # print(filename)
    #     PSNR_SSIM = proc(filename)
    #     psnr.append(PSNR_SSIM[0])
    #     ssim.append(PSNR_SSIM[1])
    with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
            # print(img_files)
            psnr.append(PSNR_SSIM[0])
            ssim.append(PSNR_SSIM[1])
            psnr_.append(PSNR_SSIM[2])
            ssim_.append(PSNR_SSIM[3])
    # except:
    #     print('x')
    #     psnr.append(0.)
    #     ssim.append(0.)

    avg_psnr = sum(psnr)/len(psnr)
    avg_ssim = sum(ssim)/len(ssim)
    avg_psnr_ = sum(psnr_) / len(psnr)
    avg_ssim_ = sum(ssim_) / len(ssim)
    print('For {:s} dataset PSNR: {:f} SSIM: {:f} PSNR_: {:f} SSIM_: {:f}\n'.format(dataset, avg_psnr, avg_ssim, avg_psnr_, avg_ssim_))
