## Restormer: Efficient Transformer for High-Resolution Image Restoration
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
## https://arxiv.org/abs/2111.09881


import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils
import time
from natsort import natsorted
from glob import glob
import seaborn as sns
import matplotlib.pyplot as plt
# from basicsr.models.archs.AdaRevID_arch import AdaRevIDSlideV2 as Net
# from basicsr.models.archs.DeepRev_arch import DeepRev as Net
# from basicsr.models.archs.DeepReFT_arch import DeepReFT as Net
from basicsr.models.archs.ATD_arch import ATD as Net
# from basicsr.models.archs.DeepKernel_arch import DeepKernel as Net
from skimage import img_as_ubyte
from collections import OrderedDict
from pdb import set_trace as stx
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from basicsr.metrics import calculate_ssim
from basicsr.models.archs.utils_deblur import *
# from skimage.metrics import structural_similarity as ssim_loss
import cv2
import kornia
import random
parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')
# parser.add_argument('--input_dir', default='/home/ubuntu/90t/personal_data/mxt/Dataset/MotionBlur/HIDE/test/blur', type=str, help='Directory of validation images') # blur sharp_phase
# parser.add_argument('--tar_dir', default='/home/ubuntu/90t/personal_data/mxt/Dataset/MotionBlur/HIDE/test/sharp', type=str, help='Directory of gt images') # sharp sharp_phase
# parser.add_argument('--input_dir', default='../../../../Datasets/Deblur/RealBlur/RealBlur_R/test/blur', type=str, help='Directory of validation images') # blur sharp_phase
# parser.add_argument('--tar_dir', default='../../../../Datasets/Deblur/RealBlur/RealBlur_R/test/sharp', type=str, help='Directory of gt images') # sharp sharp_phase
# parser.add_argument('--input_dir', default='../../../../Datasets/Deblur/REDs/blur_300', type=str, help='Directory of validation images') # blur sharp_phase
# parser.add_argument('--tar_dir', default='../../../../Datasets/Deblur/REDs/sharp_300', type=str, help='Directory of gt images') # sharp sharp_phase

# parser.add_argument('--input_dir', default='/home/ubuntu/Data1/mxt/GoPro/test/blur', type=str, help='Directory of validation images') # blur sharp_phase
# parser.add_argument('--tar_dir', default='/home/ubuntu/Data1/mxt/GoPro/test/sharp', type=str, help='Directory of gt images') # sharp sharp_phase
parser.add_argument('--input_dir', default='/data/mxt_data/SR/SR/Urban100/LR_bicubic/X2', type=str, help='Directory of validation images') # blur sharp_phase
parser.add_argument('--tar_dir', default='/data/mxt_data/SR/SR/Urban100/HR', type=str, help='Directory of gt images') # sharp sharp_phase
# parser.add_argument('--input_dir', default='/home/ubuntu/90t/personal_data/mxt/Dataset/MotionBlur/GoPro/test/blur', type=str, help='Directory of validation images') # blur sharp_phase
# parser.add_argument('--tar_dir', default='/home/ubuntu/90t/personal_data/mxt/Dataset/MotionBlur/GoPro/test/sharp', type=str, help='Directory of gt images') # sharp sharp_phase
parser.add_argument('--result_dir', default='/home/ubuntu/150t/personal_data/mxt/TPAMI/images/feature_Urban100', type=str, help='Directory for results')
# parser.add_argument('--result_dir', default='/home/ubuntu/106-48t/personal_data/mxt/exp_results/cvpr2023/results/RealBlur_DCTformer/RealBlur_R', type=str, help='Directory for results')
parser.add_argument('--weights',
                    default='/home/ubuntu/150t/personal_data/mxt/MXT/ckpts/ATD-SR/lightx2.pth',
                    type=str, help='Path to weights')
# parser.add_argument('--weights',
#                     default='/home/ubuntu/106-48t/personal_data/mxt/exp_results/ckpt/DCTformer/DCTformerPLUS/RealBlurJ/net_g_300000.pth',
#                     type=str, help='Path to weights')
# parser.add_argument('--weights',
#                     default='/home/ubuntu/106-48t/personal_data/mxt/MXT/Deblur2022/Restormer/experiments/ICCV_Abalation_DCTformer_LN_DCT_local_channel_4gpu_freqloss/models/net_g_300000.pth',
#                     type=str, help='Path to weights')
# parser.add_argument('--weights',
                    # default='../experiments/'
                    #         'Ablation_DeepRFTv2_48_1global-localnaf-ker_max_mid_featurex2-2scale-bd-148x2_Fourierloss-0.01_1e-3_256bs16_rev_200k'
                    #         '/models/net_g_200000.pth',
                    # type=str, help='Path to weights')
#  ICCV_Abalation_DCTformer_DCT_LN_local_channel_4gpu_freqloss ICCV_Abalation_DCTformer_LN_DCT_global_channel_4gpu_freqloss
# parser.add_argument('--weights',
#                     default='../experiments/Shiformer_wave3x_4gpu_freqloss/models/net_g_90000.pth',
#                     type=str, help='Path to weights')
parser.add_argument('--multi_focus', default=False, type=bool, help='save')
parser.add_argument('--dataset', default='GoPro', type=str, help='Test Dataset')
# ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--save_result', default=True, type=bool, help='save')
parser.add_argument('--get_psnr', default=True, type=bool, help='psnr')
parser.add_argument('--get_ssim', default=False, type=bool, help='ssim')
parser.add_argument('--gpus', default='0', type=str, help='gpu')
args = parser.parse_args()
crop_size = 64
up_scale=2
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
####### Load yaml #######
# yaml_file = 'Options/Abalation-DeepReFT-4gpu.yml'
yaml_file = 'Options/test/test_ATD_lightSR_x2.yml'
# yaml_file = 'Options/Abalation-DeepReFT-400k-4gpu.yml'
# yaml_file = 'Options/Abalation-DeepReFT-RL-400k-4gpu.yml'
# yaml_file = 'Options/Kernerl_Estimation_train.yml'
# yaml_file = 'Options/Abalation_DCTformer_iccv_testcentercrop.yml'

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


strict = False
print(args.weights, strict)

if args.weights is not None:
    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    s = x['network_g'].pop('type')
    ##########################
    print(x['network_g'])
    model_restoration = Net(**x['network_g'])
    checkpoint = torch.load(args.weights)
    # model_restoration.load_state_dict(checkpoint['params'])
    try:
        model_restoration.load_state_dict(checkpoint['params'], strict=strict)
        state_dict = checkpoint["params"]
        # for k, v in state_dict.items():
        #     print(k)
    except:
        try:
            model_restoration.load_state_dict(checkpoint["state_dict"], strict=strict)
        except:
            state_dict = checkpoint["state_dict"]
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                # print(k)
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v

            model_restoration.load_state_dict(new_state_dict, strict=strict)
else:
    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
    try:
        model_restoration = Net(**x['network_g'])
    except:
        print('default model')
        model_restoration = Net()
    print('inference for time')
print("===>Testing using weights: ", args.weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

psnr_val_rgb = []
ssim_val_rgb = []
ssim = 0
psnr = 0
factor = 8
all_time = 0.
dataset = args.dataset
if args.save_result:
    result_dir = args.result_dir
    os.makedirs(result_dir, exist_ok=True)

# inp_dir = os.path.join(args.input_dir, dataset, 'test', 'blur')
# tar_dir = os.path.join(args.tar_dir, dataset, 'test', 'sharp')
inp_dir = args.input_dir
tar_dir = args.tar_dir
files = natsorted(glob(os.path.join(inp_dir, '*.png')) + glob(os.path.join(inp_dir, '*.jpg')))
random.shuffle(files)
# print(files)
# img_select = ['216', '929']
# files = [os.path.join(inp_dir, x+'.png') for x in img_select]
img_end = 15
def ker_topk(ker, k_len=61, kernel_size=65):
    print(ker.shape)
    ker = torch.mean(ker, dim=1, keepdim=True)

    ker = ker.view(-1, kernel_size * kernel_size)
    _, idx = torch.topk(torch.abs(ker), k=k_len, dim=-1)
    z = torch.zeros_like(ker)
    # print(idx)
    # print(kernels_reblur[:, idx])
    z[:, idx] = 1.
    ker = ker * z
    ker = ker.view(-1, kernel_size, kernel_size)
    # ker = ker.view(-1, kernel_size, kernel_size)
    return ker

def convet_to_freq_torch(x):
    x = torch.fft.fft2(x, dim=[-2, -1])
    x = torch.fft.fftshift(x, dim=[-2, -1])
    x = torch.log(torch.abs(x)+1e-7)
    x = (x - x.min()) / (x.max() - x.min() + 1e-7)
    return x

def convet_to_freq_numpy(x):
    x = np.fft.fft2(x, axes=[0, 1])
    x = np.fft.fftshift(x, axes=[0, 1])
    x = np.log(np.abs(x)+1e-7)
    x = (x - x.min()) / (x.max() - x.min() + 1e-7)
    return x

color_fft = 'Blues_r'
color_feature = 'Reds'
color_ker = 'Greys_r'
with torch.no_grad():
    kernel_pre = 0
    for file_ in tqdm(files[:10]): # img_end [:1]
    # for file_ in files[:]:
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(file_))/255.
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()
        # print(input_.shape, input_.dtype)
        input_ = kornia.geometry.center_crop(input_, [crop_size, crop_size])
        # Padding in case images are not multiples of 8
        # h,w = input_.shape[2], input_.shape[3]
        # H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        # padh = H-h if h%factor!=0 else 0
        # padw = W-w if w%factor!=0 else 0
        # input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        gt_file_name, ext = os.path.splitext(os.path.basename(file_))
        file_gt = gt_file_name[:-2] + ext
        gt_file_way = os.path.join(tar_dir, file_gt)
        rgb_gt = np.float32(utils.load_img(gt_file_way)) / 255.
        rgb_gt = torch.from_numpy(rgb_gt).permute(2, 0, 1)
        rgb_gt = rgb_gt.unsqueeze(0).cuda()
        rgb_gt = kornia.geometry.center_crop(rgb_gt, [crop_size*up_scale, crop_size*up_scale])
        torch.cuda.synchronize()
        start = time.time()
        restored = model_restoration(input_, rgb_gt.cuda())
        torch.cuda.synchronize()
        end = time.time()
        all_time += end - start
        input_ = kornia.geometry.rescale(input_, float(up_scale))
        if isinstance(restored, list):
            restoredx = restored[-1]
        elif isinstance(restored, dict):
            # print(restored.keys())
            if 'kernel' in restored.keys():
                kernel = restored['kernel'][1]
                # print(torch.sum(torch.abs(kernel-kernel_pre)))
                # kernel_pre = kernel
            if 'mask' in restored.keys():
                mask = restored['mask']
            # mask = kornia.enhance.normalize_min_max(mask, 0., 1.)
            # restored_deconv = restored['deconv_img']
            # restored_reblur = restored['reblur_img']
            # kernel = kornia.enhance.normalize_min_max(kernel, 0., 1.)
            # restored = restored['img']  # ['x_deblur']
            # restoredx = restored['img']
            if isinstance(restored['img'], list):
                restoredx = restored['img'][-1]
            else:
                restoredx = restored['img']

        # Unpad images to original dimensions

        # restored = restored[:,:,:h,:w]

        restoredx = torch.clamp(restoredx,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        # restored_deconvx = restored_deconv.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        input_img = input_.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        # restored_reblurx = restored_reblur.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        restored_img = img_as_ubyte(restoredx)
        # restored_deconv_img = img_as_ubyte(restored_deconvx)
        # restored_reblur_img = img_as_ubyte(restored_reblurx)
        if args.get_psnr or args.get_ssim:
            # gt_file_way = os.path.join(tar_dir, os.path.basename(file_))
            if os.path.exists(gt_file_way):
                rgb_gt = cv2.imread(gt_file_way)
            else:
                rgb_gt = cv2.imread(gt_file_way[:-3] + 'png')
            rgb_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_BGR2RGB)
            rgb_gt = torch.from_numpy(rgb_gt / 255.).permute(2, 0, 1)
            rgb_gt = rgb_gt.unsqueeze(0)
            rgb_gt = kornia.geometry.center_crop(rgb_gt, [crop_size*up_scale, crop_size*up_scale])
            res_gt = rgb_gt - input_.cpu()
            rgb_gt = rgb_gt.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            rgb_gt = img_as_ubyte(rgb_gt)
            if args.get_psnr:
                PSNR = psnr_loss(restored_img, rgb_gt)
                # PSNR2 = psnr_loss(restored_deconv_img, rgb_gt)
                psnr_val_rgb.append(PSNR)
                # ssim_val_rgb.append(PSNR2)
            # if args.get_ssim:
            #     SSIM = calculate_ssim(restored_img, rgb_gt, crop_border=0)

            # print('PSNR: ', sum(psnr_val_rgb) / len(psnr_val_rgb))
        if args.save_result:
            if isinstance(restored['img'], list):
                for i, restoredi in enumerate(restored['img']):
                    res = restoredi - input_
                    res = kornia.enhance.normalize_min_max(res, 0., 1.)
                    res = res.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                    res = np.mean(res, axis=-1)
                    restoredi = torch.clamp(restoredi, 0, 1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                    utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '_restored_' + str(i) + '.png')),
                                   img_as_ubyte(restoredi))

                    out_way = os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '_res_' + str(i) + '.png')
                    sns_plot = plt.figure()
                    sns.heatmap(res, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                    sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                    plt.close()
            res_gt = kornia.enhance.normalize_min_max(res_gt, 0., 1.)
            res_gt = res_gt.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
            res_gt = np.mean(res_gt, axis=-1)
            out_way = os.path.join(result_dir,
                                   os.path.splitext(os.path.split(file_)[-1])[0] + '_res_gt.png')
            sns_plot = plt.figure()
            sns.heatmap(res_gt, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                        xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
            plt.close()

            x_f = convet_to_freq_numpy(res_gt)
            out_way = os.path.join(result_dir,
                                   os.path.splitext(os.path.split(file_)[-1])[0] + '_res_gt_fft.png')
            sns_plot = plt.figure()
            sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                        xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
            plt.close()
            # else:
            #     utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '.png')),
            #                    img_as_ubyte(restoredx))
            if 'ker_deblur' in restored.keys():
                kernel = restored['ker_deblur']
                # x_mean = torch.mean(kernel, dim=1)
                # x_mean = torch.clamp(x_mean, -1., 1.)
                # print(kernel.max(), kernel.min())
                x_mean = kornia.enhance.normalize_min_max(kernel, 0., 1.)
                x_mean = x_mean.squeeze(1)
                x_mean = x_mean.cpu().detach().squeeze(0).numpy()

                out_way = os.path.join(result_dir,
                                       os.path.splitext(os.path.split(file_)[-1])[0] + '_ker_deblur' + '.png')
                sns_plot = plt.figure()

                sns.heatmap(x_mean, cmap=color_ker, linewidths=0., vmin=0., vmax=1.,
                            xticklabels=False, yticklabels=False, cbar=False,
                            square=True)  # Reds .invert_yaxis()
                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                plt.close()

                x_mean = kornia.enhance.normalize_min_max(restored['ker_deblur_fft'], 0., 1.)
                x_mean = x_mean.squeeze(1)
                x_mean = x_mean.cpu().detach().squeeze(0).numpy()
                out_way = os.path.join(result_dir,
                                       os.path.splitext(os.path.split(file_)[-1])[0] + '_ker_deblur_fft.png')
                sns_plot = plt.figure()
                sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                            xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                plt.close()
            if 'ker_reblur' in restored.keys():
                kernel = restored['ker_reblur']
                # x_mean = torch.mean(kernel, dim=1)
                # print(kernel.max(), kernel.min())
                # x_mean = torch.clamp(x_mean, 0., 1.)
                x_mean = kornia.enhance.normalize_min_max(kernel, 0., 1.)
                x_mean = x_mean.squeeze(1)
                x_mean = x_mean.cpu().detach().squeeze(0).numpy()

                out_way = os.path.join(result_dir,
                                       os.path.splitext(os.path.split(file_)[-1])[0] + '_ker_reblur' + '.png')
                sns_plot = plt.figure()

                sns.heatmap(x_mean, cmap=color_ker, linewidths=0., vmin=0., vmax=1.,
                            xticklabels=False, yticklabels=False, cbar=False,
                            square=True)  # Reds .invert_yaxis()
                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                plt.close()


                x_mean = kornia.enhance.normalize_min_max(restored['ker_reblur_fft'], 0., 1.)
                x_mean = x_mean.squeeze(1)
                x_mean = x_mean.cpu().detach().squeeze(0).numpy()
                out_way = os.path.join(result_dir,
                                       os.path.splitext(os.path.split(file_)[-1])[0] + '_ker_reblur_fft.png')
                sns_plot = plt.figure()
                sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                            xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                plt.close()
            if 'feature_visual' in restored.keys():
                feature_visuals = restored['feature_visual']
                for feature_visual in [feature_visuals[-1]]:
                    if 'kernel' in feature_visual.keys():
                        kernel = feature_visual['kernel']
                        # print(kernel.shape)
                        # x_mean = ker_topk(kernel)
                        x_mean = torch.mean(kernel, dim=1)
                        x_mean = kornia.enhance.normalize_min_max(x_mean, 0., 1.)
                        x_mean = x_mean.cpu().detach().squeeze(0).numpy()
                        # print(x_mean.shape)
                        out_way = os.path.join(result_dir,
                                               os.path.splitext(os.path.split(file_)[-1])[0] + '_mean_kernel' + '.png')
                        sns_plot = plt.figure()

                        sns.heatmap(x_mean, cmap=color_ker, linewidths=0., vmin=0., vmax=1.,
                                    xticklabels=False, yticklabels=False, cbar=False,
                                    square=True)  # Reds .invert_yaxis()
                        sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                        plt.close()

                        # kernel = kornia.enhance.normalize_min_max(kernel, 0., 1.)
                        kernelx = kernel.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                        for k in range(kernelx.shape[-1]):
                            out_way = os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '_kernel_' + str(k) + '.png')
                            sns_plot = plt.figure()
                            # print(kernelx[:, :, k].max())
                            ker = kernelx[:, :, k]
                            ker = (ker-ker.min()) / (ker.max()-ker.min()+1e-7)
                            sns.heatmap(ker, cmap=color_ker, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                    if 'kernel_fft' in feature_visual.keys():
                        kernel = feature_visual['kernel_fft']
                        # print(kernel.shape)
                        # x_mean = ker_topk(kernel)
                        x_mean = torch.mean(kernel, dim=1)
                        x_mean = kornia.enhance.normalize_min_max(x_mean, 0., 1.)
                        x_mean = x_mean.cpu().detach().squeeze(0).numpy()
                        # print(x_mean.shape)
                        out_way = os.path.join(result_dir,
                                               os.path.splitext(os.path.split(file_)[-1])[0] + '_mean_kernel_fft' + '.png')
                        sns_plot = plt.figure()

                        sns.heatmap(x_mean, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                    xticklabels=False, yticklabels=False, cbar=False,
                                    square=True)  # Reds .invert_yaxis()
                        sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                        plt.close()

                        # kernel = kornia.enhance.normalize_min_max(kernel, 0., 1.)
                        kernelx = kernel.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                        for k in range(kernelx.shape[-1]):
                            out_way = os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '_kernel_' + str(k) + '_fft.png')
                            sns_plot = plt.figure()
                            # print(kernelx[:, :, k].max())
                            ker = kernelx[:, :, k]
                            ker = (ker-ker.min()) / (ker.max()-ker.min()+1e-7)
                            sns.heatmap(ker, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                    if 'feature' in feature_visual.keys():
                        feature = feature_visual['feature']
                        x_mean = torch.mean(feature, dim=1)
                        x_mean = kornia.enhance.normalize_min_max(x_mean, 0., 1.)
                        x_mean = x_mean.cpu().detach().squeeze(0).numpy()

                        out_way = os.path.join(result_dir,
                                               os.path.splitext(os.path.split(file_)[-1])[0] + '_feature_mean' + '.png')
                        sns_plot = plt.figure()

                        sns.heatmap(x_mean, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                    xticklabels=False, yticklabels=False, cbar=False,
                                    square=True)  # Reds .invert_yaxis()
                        sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                        plt.close()

                        x_f = convet_to_freq_numpy(x_mean)
                        out_way = os.path.join(result_dir,
                                               os.path.splitext(os.path.split(file_)[-1])[0] + '_feature_mean_fft.png')
                        sns_plot = plt.figure()
                        sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                    xticklabels=False, yticklabels=False, cbar=False,
                                    square=True)  # Reds .invert_yaxis()
                        sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                        plt.close()
                        feature_ori = feature
                        # feature = kornia.enhance.normalize_min_max(feature, 0., 1.)
                        xs = feature.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                        xs_ori = xs
                        for k in range(xs.shape[-1]):
                            out_way = os.path.join(result_dir,
                                                   os.path.splitext(os.path.split(file_)[-1])[0] + '_feature_' + str(
                                                       k) + '.png')
                            sns_plot = plt.figure()
                            # print(kernelx[:, :, k].max())
                            x = xs[:, :, k]
                            x = (x - x.min()) / (x.max() - x.min() + 1e-7)
                            sns.heatmap(x, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False,
                                        square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                            x_f = convet_to_freq_numpy(x)
                            out_way = os.path.join(result_dir,
                                                   os.path.splitext(os.path.split(file_)[-1])[0] + '_feature_' + str(
                                                       k) + '_fft.png')
                            sns_plot = plt.figure()
                            sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False,
                                        square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                    if 'deconv_feature' in feature_visual.keys():
                        feature = feature_visual['deconv_feature'] # + feature_visual['feature']) * 0.5
                        x_mean = torch.mean(feature, dim=1)
                        x_mean = kornia.enhance.normalize_min_max(x_mean, 0., 1.)
                        x_mean = x_mean.cpu().detach().squeeze(0).numpy()

                        out_way = os.path.join(result_dir,
                                               os.path.splitext(os.path.split(file_)[-1])[0] + '_feature_mean_deconv' + '.png')
                        sns_plot = plt.figure()

                        sns.heatmap(x_mean, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                    xticklabels=False, yticklabels=False, cbar=False,
                                    square=True)  # Reds .invert_yaxis()
                        sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                        plt.close()
                        x_f = convet_to_freq_numpy(x_mean)
                        out_way = os.path.join(result_dir,
                                               os.path.splitext(os.path.split(file_)[-1])[0] + '_feature_mean_deconv_fft.png')
                        sns_plot = plt.figure()
                        sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                    xticklabels=False, yticklabels=False, cbar=False,
                                    square=True)  # Reds .invert_yaxis()
                        sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                        plt.close()
                        # feature = feature - feature_ori
                        # feature = kornia.enhance.normalize_min_max(feature, 0., 1.)
                        xs = feature.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

                        for k in range(xs.shape[-1]):
                            out_way = os.path.join(result_dir,
                                                   os.path.splitext(os.path.split(file_)[-1])[0] + '_feature_' + str(
                                                       k) + '_deconv.png')
                            sns_plot = plt.figure()
                            # print(kernelx[:, :, k].max())
                            x = xs[:, :, k]
                            x = (x - x.min()) / (x.max() - x.min() + 1e-7)
                            sns.heatmap(x, cmap=color_feature, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False,
                                        square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                            x_f = convet_to_freq_numpy(x)
                            out_way = os.path.join(result_dir,
                                                   os.path.splitext(os.path.split(file_)[-1])[0] + '_feature_' + str(
                                                       k) + '_deconv_fft.png')
                            sns_plot = plt.figure()
                            sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                                        xticklabels=False, yticklabels=False, cbar=False,
                                        square=True)  # Reds .invert_yaxis()
                            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                            plt.close()
                # if 'mask' in restored.keys():
                #     maskx = mask.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                #     for k in range(maskx.shape[-1]):
                #         out_way = os.path.join(result_dir,
                #                                os.path.splitext(os.path.split(file_)[-1])[0] + '_mask_' + str(k) + '.png')
                #         sns_plot = plt.figure()
                #         # print(maskx[:, :, k].max())
                #         sns.heatmap(maskx[:, :, k], cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                #                     xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
                #         sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
                #         plt.close()
            #     img_ = kornia.tensor_to_image(out_weiner)
            #
            #     cv2.imwrite((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '_deblur_xweiner' + str(k) + '.png')),
            #                    np.uint8(img_[:, :, ::-1]*255.))
            # residual = rgb_gt - input_img
            # utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'_blur_pattern.png')),
            #                img_as_ubyte(residual))
            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + 'gt.png')),
                           rgb_gt)
            x_f = convet_to_freq_numpy(np.mean(rgb_gt, axis=-1))
            out_way = os.path.join(result_dir,
                                   os.path.splitext(os.path.split(file_)[-1])[0] + '_rgb_gt_fft.png')
            sns_plot = plt.figure()
            sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                        xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
            plt.close()
            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '_blur.png')),
                           img_as_ubyte(input_img))
            x_f = convet_to_freq_numpy(np.mean(input_img, axis=-1))
            out_way = os.path.join(result_dir,
                                   os.path.splitext(os.path.split(file_)[-1])[0] + '_blur_fft.png')
            sns_plot = plt.figure()
            sns.heatmap(x_f, cmap=color_fft, linewidths=0., vmin=0., vmax=1.,
                        xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds .invert_yaxis()
            sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
            plt.close()
            # utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0] + '_blure.png')),
            #                img_as_ubyte(restored_reblur_img))
print('average_time: ', all_time / img_end)
if args.get_psnr:
    psnr = sum(psnr_val_rgb) / len(psnr_val_rgb)
    print("PSNR: %f" % psnr)
    # ssim = sum(ssim_val_rgb) / len(ssim_val_rgb)
    # print("PSNR2: %f" % ssim)

