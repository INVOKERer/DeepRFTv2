
import numpy as np
import torch.nn as nn
import torch
from einops import rearrange
# input channel: 3


class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1))

    def forward(self, input):
        res = self.body(input)
        output = res + input
        return output


class kernel_extra_Encoding_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_Encoding_Block, self).__init__()
        self.Conv_head = nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        self.ResBlock1 = ResBlock(out_ch)
        self.ResBlock2 = ResBlock(out_ch)
        self.downsample = nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

    def forward(self, input):
        output = self.Conv_head(input)
        output = self.ResBlock1(output)
        output = self.ResBlock2(output)
        skip = self.act(output)
        output = self.downsample(skip)

        return output, skip


class kernel_extra_conv_mid(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_conv_mid, self).__init__()
        self.body = nn.Sequential(
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU(),
                        nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
                        nn.ReLU()
                        )

    def forward(self, input):
        output = self.body(input)
        return output


class kernel_extra_Decoding_Block(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_Decoding_Block, self).__init__()
        self.Conv_t = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=3, stride=2, padding=1)
        self.Conv_head = nn.Conv2d(out_ch*2, out_ch, kernel_size=3, stride=1, padding=1)
        self.ResBlock1 = ResBlock(out_ch)
        self.ResBlock2 = ResBlock(out_ch)
        self.act = nn.ReLU()

    def forward(self, input, skip):
        output = self.Conv_t(input, output_size=[skip.shape[0], skip.shape[1], skip.shape[2], skip.shape[3]])
        output = torch.cat([output, skip], dim=1)
        output = self.Conv_head(output)
        output = self.ResBlock1(output)
        output = self.ResBlock2(output)
        output = self.act(output)

        return output


class kernel_extra_conv_tail(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_conv_tail, self).__init__()
        self.mean = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
                        )

        self.kernel_size = int(np.sqrt(out_ch))

    def forward(self, input):
        kernel_mean = self.mean(input)
        # kernel_feature = kernel_mean

        kernel_mean = nn.Softmax2d()(kernel_mean)
        # kernel_mean = kernel_mean.mean(dim=[2, 3], keepdim=True)
        kernel_mean = kernel_mean.reshape(kernel_mean.shape[0], self.kernel_size, self.kernel_size, kernel_mean.shape[2] * kernel_mean.shape[3]).permute(0, 3, 1, 2)
        # kernel_mean = kernel_mean.view(-1, self.kernel_size, self.kernel_size)

        # kernel_mean --size:[B, H*W, 19, 19]
        return kernel_mean


class kernel_extra_conv_tail_mean_var(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_extra_conv_tail_mean_var, self).__init__()
        self.mean = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
                        )
        self.var = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1), nn.ReLU(),
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1), nn.Sigmoid()
                        )

        self.kernel_size = int(np.sqrt(out_ch))

    def forward(self, input):
        kernel_mean = self.mean(input)
        kernel_var = self.var(input)
        # kernel_feature = kernel_mean

        # kernel_mean = nn.Softmax2d()(kernel_mean)
        # kernel_mean = kernel_mean.mean(dim=[2, 3], keepdim=True)
        kernel_mean = kernel_mean.reshape(kernel_mean.shape[0], self.kernel_size, self.kernel_size, kernel_mean.shape[2] * kernel_mean.shape[3]).permute(0, 3, 1, 2)
        # kernel_mean = kernel_mean.view(-1, self.kernel_size, self.kernel_size)

        kernel_var = kernel_var.reshape(kernel_var.shape[0], self.kernel_size, self.kernel_size, kernel_var.shape[2] * kernel_var.shape[3]).permute(0, 3, 1, 2)
        kernel_var = kernel_var.mean(dim=[2, 3], keepdim=True)
        kernel_var = kernel_var.repeat(1, 1, self.kernel_size, self.kernel_size)

        # kernel_mean --size:[B, H*W, 19, 19]
        return kernel_mean, kernel_var

class simple_kernel_extra(nn.Module):
    def __init__(self, in_ch, out_ch, conv_kernel_size=1):
        super(simple_kernel_extra, self).__init__()
        self.mean = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=conv_kernel_size, stride=1, padding=(conv_kernel_size-1)//2), nn.ReLU(),
                        nn.Conv2d(in_ch, out_ch, kernel_size=conv_kernel_size, stride=1, padding=(conv_kernel_size-1)//2)
                        )

        self.kernel_size = int(np.sqrt(out_ch))

    def forward(self, input):
        kernel = self.mean(input)
        # kernel_feature = kernel_mean

        # kernel_mean = nn.Softmax2d()(kernel_mean)
        # kernel_mean = kernel_mean.mean(dim=[2, 3], keepdim=True)
        kernel = kernel.reshape(kernel.shape[0], self.kernel_size, self.kernel_size, kernel.shape[2] * kernel.shape[3]).permute(0, 3, 1, 2)


        # kernel_mean --size:[B, H*W, 19, 19]
        return kernel
class simple_Avgkernel_extra(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(simple_Avgkernel_extra, self).__init__()
        self.mean = nn.Sequential(
                        nn.AdaptiveAvgPool2d(1),
                        nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0), nn.ReLU(),
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
                        )

        self.kernel_size = int(np.sqrt(out_ch))

    def forward(self, input):
        kernel = self.mean(input)
        # kernel_feature = kernel_mean

        # kernel_mean = nn.Softmax2d()(kernel_mean)
        # kernel_mean = kernel_mean.mean(dim=[2, 3], keepdim=True)
        kernel = kernel.reshape(kernel.shape[0], self.kernel_size, self.kernel_size, kernel.shape[2] * kernel.shape[3]).permute(0, 3, 1, 2)


        # kernel_mean --size:[B, H*W, 19, 19]
        return kernel
class simple_Adaptive_extra(nn.Module):
    def __init__(self, in_ch, out_ch, num_ker=1):
        super(simple_Adaptive_extra, self).__init__()

        self.mean = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0), nn.ReLU(),
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
                        )
        self.conv_attn = nn.Sequential(
            nn.Conv2d(in_ch, num_ker+16, kernel_size=1, stride=1, padding=0), nn.ReLU(),
            nn.Conv2d(num_ker+16, num_ker, kernel_size=1, stride=1, padding=0)
        )
        self.kernel_size = int(np.sqrt(out_ch))
        self.temp = nn.ParameterList()
        self.temp.append(nn.Parameter(torch.ones(1, num_ker, 1) * 2, requires_grad=True))
    def forward(self, input):

        # kernel_feature = kernel_mean
        ker_attn = self.conv_attn(input)
        ker_attn = rearrange(ker_attn, 'b c h w -> b c (h w)')
        kernel_attn = rearrange(input, 'b c h w -> b (h w) c')
        ker_attn = torch.softmax(ker_attn * self.temp[0], dim=-1)
        kernel_key = ker_attn @ kernel_attn
        kernel_key = kernel_key.permute(0, 2, 1).unsqueeze(-1)
        # print(kernel_key.shape)
        # kernel_mean = nn.Softmax2d()(kernel_mean)
        # kernel_mean = kernel_mean.mean(dim=[2, 3], keepdim=True)
        kernel = self.mean(kernel_key)
        kernel = kernel.reshape(kernel.shape[0], self.kernel_size, self.kernel_size, kernel.shape[2] * kernel.shape[3]).permute(0, 3, 1, 2)


        # kernel_mean --size:[B, H*W, 19, 19]
        return kernel
class simple_kernel_mask_extra(nn.Module):
    def __init__(self, in_ch, out_ch, size_k):
        super(simple_kernel_mask_extra, self).__init__()
        self.mean = nn.Sequential(
                        nn.Conv2d(in_ch, in_ch, kernel_size=1, stride=1, padding=0), nn.ReLU(),
                        nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0)
                        )
        self.mask = nn.Sequential(
                    nn.Conv2d(size_k, 16, kernel_size=1, stride=1, padding=0, bias=True),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True),
                    nn.Conv2d(16, 2, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.1, inplace=True)
                )

        self.kernel_size = int(np.sqrt(out_ch))

    def forward(self, input):
        kernel = self.mean(input)
        # kernel_feature = kernel_mean

        # kernel_mean = nn.Softmax2d()(kernel_mean)
        # kernel_mean = kernel_mean.mean(dim=[2, 3], keepdim=True)
        kernel = kernel.reshape(kernel.shape[0], self.kernel_size, self.kernel_size, kernel.shape[2] * kernel.shape[3]).permute(0, 3, 1, 2)
        kernel_mask = self.mask(kernel)
        kernel_mask = torch.nn.functional.gumbel_softmax(kernel_mask, tau=1, hard=True, dim=1)
        kernel_mask_p1, _ = kernel_mask.chunk(2, dim=1)
        # kernel_mean --size:[B, H*W, 19, 19]
        return kernel_mask_p1
class kernel_extra_conv_tail_mean_var_numk(nn.Module):
    def __init__(self, in_ch, out_ch, spatial_size, num_k):
        super(kernel_extra_conv_tail_mean_var_numk, self).__init__()
        self.mean = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )
        self.var = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1)
        )
        self.reblur_kernel_extras_mean_var = nn.Conv2d(spatial_size, num_k, kernel_size=1, stride=1, padding=0, bias=False)
        # self.reblur_kernel_extras_var = nn.Conv2d(spatial_size, num_k, kernel_size=1, stride=1, padding=0, bias=False)
        self.kernel_size = int(np.sqrt(out_ch))

    def forward(self, input):
        kernel_mean = self.mean(input)
        kernel_var = self.var(input)
        # kernel_feature = kernel_mean

        # kernel_mean = nn.Softmax2d()(kernel_mean)
        # kernel_mean = kernel_mean.mean(dim=[2, 3], keepdim=True)
        kernel_mean = kernel_mean.reshape(kernel_mean.shape[0], self.kernel_size, self.kernel_size,
                                          kernel_mean.shape[2] * kernel_mean.shape[3]).permute(0, 3, 1, 2)
        # kernel_mean = kernel_mean.view(-1, self.kernel_size, self.kernel_size)
        # kernel_mean = self.reblur_kernel_extras_mean(kernel_mean)
        kernel_var = kernel_var.reshape(kernel_var.shape[0], self.kernel_size, self.kernel_size,
                                        kernel_var.shape[2] * kernel_var.shape[3]).permute(0, 3, 1, 2)
        kernel_mean, kernel_var = self.reblur_kernel_extras_mean_var(torch.cat([kernel_mean, kernel_var], dim=1)).chunk(2, dim=1)
        kernel_var = kernel_var.mean(dim=[2, 3], keepdim=True)
        kernel_var = kernel_var.repeat(1, 1, self.kernel_size, self.kernel_size)

        # kernel_mean --size:[B, H*W, 19, 19]
        return kernel_mean, torch.sigmoid(kernel_var)
class num_kernel_extra_conv_tail_mean_var(nn.Module):
    def __init__(self, in_ch, kernel_size, dw=2):
        super(num_kernel_extra_conv_tail_mean_var, self).__init__()
        # in_ch = kernel_size**2
        hid_dim = int(dw * in_ch)
        self.mean = nn.Sequential(
                        nn.Linear(in_ch, hid_dim), nn.ReLU(),
                        nn.Linear(hid_dim, in_ch)
                        )
        self.var = nn.Sequential(
                        nn.Linear(in_ch, hid_dim), nn.ReLU(),
                        nn.Linear(hid_dim, in_ch), nn.Sigmoid()
                        )

        self.kernel_size = kernel_size

    def forward(self, input):
        # input b c ker ker
        input = rearrange(input, 'b c k1 k2 -> b (k1 k2) c')
        # print(input.shape)
        kernel_mean = self.mean(input)
        kernel_var = self.var(input)
        # kernel_feature = kernel_mean

        kernel_mean = torch.softmax(kernel_mean, dim=1)
        # kernel_mean = kernel_mean.mean(dim=[2, 3], keepdim=True)
        kernel_mean = rearrange(kernel_mean, 'b (k1 k2) c -> b c k1 k2', k1=self.kernel_size, k2=self.kernel_size)
            # .reshape(kernel_mean.shape[0], self.kernel_size, self.kernel_size, kernel_mean.shape[2]).permute(0, 3, 1, 2)
        # kernel_mean = kernel_mean.view(-1, self.kernel_size, self.kernel_size)

        kernel_var = rearrange(kernel_var, 'b (k1 k2) c -> b c k1 k2', k1=self.kernel_size, k2=self.kernel_size)
        kernel_var = kernel_var.mean(dim=[2, 3], keepdim=True)
        kernel_var = kernel_var.repeat(1, 1, self.kernel_size, self.kernel_size)

        # kernel_mean --size:[B, H*W, 19, 19]
        return kernel_mean, kernel_var

class kernel_extra(nn.Module):
    def __init__(self, kernel_size):
        super(kernel_extra, self).__init__()
        self.kernel_size = kernel_size
        self.Encoding_Block1 = kernel_extra_Encoding_Block(3, 64)
        self.Encoding_Block2 = kernel_extra_Encoding_Block(64, 128)
        self.Conv_mid = kernel_extra_conv_mid(128, 256)
        self.Decoding_Block1 = kernel_extra_Decoding_Block(256, 128)
        self.Decoding_Block2 = kernel_extra_Decoding_Block(128, 64)
        self.Conv_tail = kernel_extra_conv_tail(64, self.kernel_size*self.kernel_size)

    def forward(self, input):
        output, skip1 = self.Encoding_Block1(input)
        output, skip2 = self.Encoding_Block2(output)
        output = self.Conv_mid(output)
        output = self.Decoding_Block1(output, skip2)
        output = self.Decoding_Block2(output, skip1)
        kernel = self.Conv_tail(output)

        return kernel


class code_extra_mean_var(nn.Module):
    def __init__(self, kernel_size):
        super(code_extra_mean_var, self).__init__()
        self.kernel_size = kernel_size
        self.Encoding_Block1 = kernel_extra_Encoding_Block(3, 64)
        self.Encoding_Block2 = kernel_extra_Encoding_Block(64, 128)
        self.Conv_mid = kernel_extra_conv_mid(128, 256)
        self.Decoding_Block1 = kernel_extra_Decoding_Block(256, 128)
        self.Decoding_Block2 = kernel_extra_Decoding_Block(128, 64)
        self.Conv_tail = kernel_extra_conv_tail_mean_var(64, self.kernel_size*self.kernel_size)

    def forward(self, input):
        output, skip1 = self.Encoding_Block1(input)
        output, skip2 = self.Encoding_Block2(output)
        output = self.Conv_mid(output)
        output = self.Decoding_Block1(output, skip2)
        output = self.Decoding_Block2(output, skip1)
        code, var = self.Conv_tail(output)

        return code, var

class code_extra_mean_varX(nn.Module):
    def __init__(self, dim, hdim, kernel_size):
        super(code_extra_mean_varX, self).__init__()
        self.kernel_size = kernel_size
        self.Encoding_Block1 = kernel_extra_Encoding_Block(dim, hdim)
        self.Encoding_Block2 = kernel_extra_Encoding_Block(hdim, hdim*2)
        self.Conv_mid = kernel_extra_conv_mid(hdim*2, hdim*4)
        self.Decoding_Block1 = kernel_extra_Decoding_Block(hdim*4, hdim*2)
        self.Decoding_Block2 = kernel_extra_Decoding_Block(hdim*2, hdim)
        self.Conv_tail = kernel_extra_conv_tail_mean_var(hdim, self.kernel_size*self.kernel_size)

    def forward(self, input):
        output, skip1 = self.Encoding_Block1(input)
        output, skip2 = self.Encoding_Block2(output)
        output = self.Conv_mid(output)
        output = self.Decoding_Block1(output, skip2)
        output = self.Decoding_Block2(output, skip1)
        code, var = self.Conv_tail(output)

        return code, var
if __name__ == '__main__':
    import kornia
    input1 = torch.rand(1, 1, 65, 65).cuda()
    input1 = kornia.geometry.subpix.spatial_softmax2d(input1)
    otf = torch.fft.fft2(input1)
    otf = torch.conj(otf) / (torch.abs(otf) ** 2 + 1e-6)
    kernel = torch.fft.ifft2(otf).real  # [:, :, :self.kernel_size[-1], :self.kernel_size[-1]]
    print(kernel)
    # net = code_extra_mean_var(19).cuda()
    # code, var = net(input1)
    # print(net)
    # print(var.size())
    # print("Total number of paramerters in networks is {}  ".format(sum(x.numel() for x in net.parameters())))








