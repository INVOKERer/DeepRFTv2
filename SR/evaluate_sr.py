## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881

import os
import numpy as np
from glob import glob
from natsort import natsorted
from skimage import io
import cv2
from tqdm import tqdm
import concurrent.futures
from basicsr.metrics import calculate_psnr, calculate_ssim

def compute_psnr(image_true, image_test, crop_border):

  return calculate_psnr(image_true, image_test, crop_border=crop_border, test_y_channel=True)


def compute_ssim(tar_img, prd_img, crop_border):

    ssim = calculate_ssim(tar_img, prd_img, crop_border=crop_border, test_y_channel=True)
    return ssim

def proc(filename):
    crop_border = 2
    tar,prd = filename
    tar_img = io.imread(tar)
    prd_img = io.imread(prd)
    
    tar_img = tar_img.astype(np.float32)/255.0
    prd_img = prd_img.astype(np.float32)/255.0

    PSNR = compute_psnr(tar_img, prd_img, crop_border)
    SSIM = compute_ssim(tar_img, prd_img, crop_border)
    return (PSNR,SSIM)

datasets = ['Set5', 'Set14']
# datasets = ['GoPro']
# datasets = ['RealBlur_R']
# datasets = ['RealBlur_J']
for dataset in datasets:

    # file_path = os.path.join('/home/ubuntu/90t/personal_data/mxt/MXT/exp_results/RevD-L_RealBlur', dataset)
    
    file_path = os.path.join('/home/ubuntu/150t/personal_data/mxt/MXT/exp_results/MambaIRv2-FKE', dataset)

    print(file_path)
    gt_path = os.path.join('/data/mxt_data/SR/SR', dataset, 'HR')

    path_list = natsorted(glob(os.path.join(file_path, '*.png')) + glob(os.path.join(file_path, '*.jpg')))
    gt_list = natsorted(glob(os.path.join(gt_path, '*.png')) + glob(os.path.join(gt_path, '*.jpg')))
    print(len(path_list), len(gt_list))
    assert len(path_list) != 0, "Predicted files not found"
    assert len(gt_list) != 0, "Target files not found"

    psnr, ssim = [], []
    # img_files =[(i, j) for i,j in zip(gt_list,path_list)]
    # try:
    # with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    # for filename in img_files:
    #     # print(filename)
    #     PSNR_SSIM = proc(filename)
    #     psnr.append(PSNR_SSIM[0])
    #     ssim.append(PSNR_SSIM[1])
    # with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
    #     for filename, PSNR_SSIM in zip(img_files, executor.map(proc, img_files)):
    #         # print(img_files)
    for i1, i2 in zip(gt_list, path_list):
        PSNR_SSIM = proc([i1, i2])
        psnr.append(PSNR_SSIM[0])
        ssim.append(PSNR_SSIM[1])
    # except:
    #     print('x')
    #     psnr.append(0.)
    #     ssim.append(0.)

    avg_psnr = sum(psnr)/len(psnr)
    avg_ssim = sum(ssim)/len(ssim)

    print('For {:s} dataset PSNR: {:f} SSIM: {:f}\n'.format(dataset, avg_psnr, avg_ssim))
