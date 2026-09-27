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
# from basicsr.models.archs.DeepMultiReLogFT_arch import DeepMultiReLogFT as Net
from basicsr.models.archs.DeepRFTv2_arch import DeepRFTv2 as Net
# from basicsr.models.archs.UFPNet_code_uncertainty_arch import UFPNet_code_uncertainty_Local as Net
# from basicsr.models.archs.DeepMReFT_arch import DeepMReFT as Net
from skimage import img_as_ubyte
from collections import OrderedDict
from pdb import set_trace as stx
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from basicsr.metrics import calculate_ssim
# from skimage.metrics import structural_similarity as ssim_loss
import cv2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
# os.environ["CUDA_VISIBLE_DEVICESCUDA_VISIBLE_DEVICESCUDA_VISIBLE_DEVICESCUDA_VISIBLE_DEVICES"] = '3'
post_aug = False
parser = argparse.ArgumentParser(description='Single Image Motion Deblurring using Restormer')


parser.add_argument('--input_dir', default='/data/mxt_data/REDS/test/blur', type=str, help='Directory of validation images') # blur sharp_phase
parser.add_argument('--tar_dir', default='/data/mxt_data/REDS/test/sharp', type=str, help='Directory of gt images') # sharp sharp_phase

parser.add_argument('--result_dir', default='/home/ubuntu/150t/personal_data/mxt/MXT/exp_results/DeepRFTv2_post/REDS', type=str, help='Directory for results')
# parser.add_argument('--result_dir', default='/home/ubuntu/150t/personal_data/mxt/MXT/exp_results/UFPNet/HIDE', type=str, help='Directory for results')
# parser.add_argument('--weights',
#                     default='../experiments/RealBlur_R-DeepRFTv2-NAFEVSBlockx2-640-200k/models/net_g_70000.pth',
#                     type=str, help='Path to weights')
parser.add_argument('--weights',
                    default='../experiments/REDS-DeepRFTv2-B-512/models/net_g_600000.pth',
                    type=str, help='Path to weights')

parser.add_argument('--multi_focus', default=False, type=bool, help='save')
parser.add_argument('--dataset', default='GoPro', type=str, help='Test Dataset')
# ['GoPro', 'HIDE', 'RealBlur_J', 'RealBlur_R']
parser.add_argument('--save_result', default=True, type=bool, help='save')
parser.add_argument('--get_psnr', default=True, type=bool, help='psnr')
parser.add_argument('--get_ssim', default=True, type=bool, help='ssim')
parser.add_argument('--gpus', default='0', type=str, help='gpu')
args = parser.parse_args()
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
####### Load yaml #######

yaml_file = 'Options/REDS-DeepRFTv2-test.yml'
# yaml_file = 'Options/FFTformer.yml'
# yaml_file = 'Options/Abalation-DeepMReFT-200k-4gpu.yml'

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

# print(args.weights)
if args.weights is not None:
    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

    s = x['network_g'].pop('type')
    ##########################
    print(x['network_g'])
    model_restoration = Net(**x['network_g'])
    checkpoint = torch.load(args.weights)
    print(checkpoint.keys())
    # model_restoration.load_state_dict(checkpoint['params'])
    try:
        # model_restoration.load_state_dict(checkpoint, strict=True)
        model_restoration.load_state_dict(checkpoint['params'], strict=True)
        # state_dict = checkpoint["params"]
        # for k, v in state_dict.items():
        #     print(k)
    except:
        try:
            model_restoration.load_state_dict(checkpoint["state_dict"], strict=True)
        except:
            model_restoration.load_state_dict(checkpoint, strict=True)
        # except:
        #     state_dict = checkpoint["state_dict"]
        #     new_state_dict = OrderedDict()
        #     for k, v in state_dict.items():
        #         # print(k)
        #         name = k[7:]  # remove `module.`
        #         new_state_dict[name] = v
        #
        #     model_restoration.load_state_dict(new_state_dict)
else:
    x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)
    try:
        model_restoration = Net(**x['network_g'])
    except:
        print('default model')
        model_restoration = Net()
    print('inference for time')
print("===>Testing using weights: ", args.weights)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '12345'
def get_dist_info():
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size
rank, world_size = get_dist_info()
# print(rank, world_size)
# torch.backends.cudnn.benchmark = True
# dist.init_process_group('nccl', rank=rank, world_size=world_size)
# model_restoration = DDP(model_restoration)
model_restoration = nn.DataParallel(model_restoration)

model_restoration.cuda()
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


def check_image_size(x, padder_size):
    _, _, h, w = x.size()
    mod_pad_h = (padder_size - h % padder_size) % padder_size
    mod_pad_w = (padder_size - w % padder_size) % padder_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x
# crop_size = [720, 1024]
def post_aug_(x, post_aug=False):
    if post_aug:
        x0 = x
        x1 = torch.rot90(x, k=2, dims=[-2, -1])
        x2 = torch.flipud(x)
        x3 = torch.flipud(x1)

        x = torch.cat([x0, x1, x2, x3], dim=0)
    return x


def get_image(prd_img, post_aug=False):

    if post_aug:
        # prd_img = prd_img[:, :, h1:h+h1, w1:w+w1]
        x0, x1, x2, x3 = prd_img.chunk(4, dim=0)
        x1 = torch.rot90(x1, k=2, dims=[-2, -1])
        x2 = torch.flipud(x2)
        x3 = torch.flipud(x3)
        x3 = torch.rot90(x3, k=2, dims=[-2, -1])

        prd_img = (x0 + x1 + x2 + x3) / 4.0
    return prd_img


with torch.no_grad():
    for file_ in tqdm(files): # [:1]
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(file_))/255.
        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 8
        h,w = input_.shape[2], input_.shape[3]
        # h, w = input_.shape[-2:]
        # h_start = (h - crop_size[0]) // 2
        # h_end = (h + crop_size[0]) // 2
        # w_start = (w - crop_size[1]) // 2
        # w_end = (w + crop_size[1]) // 2
        # input_center = input_[:, :, h_start:h_end, w_start:w_end]
        # H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        # padh = H-h if h%factor!=0 else 0
        # padw = W-w if w%factor!=0 else 0
        # input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        torch.cuda.synchronize()
        start = time.time()

        input_ = post_aug_(input_, post_aug)
        # input_ = input_.to('cuda:0')
        restored = model_restoration(input_)

        torch.cuda.synchronize()
        end = time.time()
        all_time += end - start

        if isinstance(restored, list):
            restored = restored[-1]
        elif isinstance(restored, dict):
            restored = restored['img']
            if isinstance(restored, list):
                restored = restored[-1]
        # Unpad images to original dimensions
        restored = get_image(restored, post_aug)
        # Unpad images to original dimensions

        # restored = restored[:,:,:h,:w]

        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        restored_img = img_as_ubyte(restored)
        if args.get_psnr or args.get_ssim:
            gt_file_way = os.path.join(tar_dir, os.path.basename(file_))
            if os.path.exists(gt_file_way):
                rgb_gt = cv2.imread(gt_file_way)
            else:
                rgb_gt = cv2.imread(gt_file_way[:-3] + 'png')
            rgb_gt = cv2.cvtColor(rgb_gt, cv2.COLOR_BGR2RGB)

            if args.get_psnr:
                PSNR = psnr_loss(restored_img, rgb_gt)
                psnr_val_rgb.append(PSNR)
            if args.get_ssim:
                SSIM = calculate_ssim(restored_img, rgb_gt, crop_border=0)
                ssim_val_rgb.append(SSIM)
            # print('PSNR: ', sum(psnr_val_rgb) / len(psnr_val_rgb))
        if args.save_result:
            utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(file_)[-1])[0]+'.png')), img_as_ubyte(restored))
print('average_time: ', all_time / len(files))
if args.get_psnr:
    psnr = sum(psnr_val_rgb) / len(files)
    print("PSNR: %f" % psnr)
if args.get_ssim:
    ssim = sum(ssim_val_rgb) / len(files)
    print("SSIM: %f" % ssim)
# print('num_decoders: ', num_decoders)
