import kornia.filters

import torch.nn.functional as F
from basicsr.models.archs.arch_util import *  # LayerNorm2d, window_reversex, window_partitionx, FFT_ReLU, Attention_win
from einops.layers.torch import Rearrange, Reduce
from basicsr.models.archs.ops_dcnv3.dcnv3_torch import *

from basicsr.models.archs.local_arch import *
from basicsr.models.archs.attn_util import *
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
from mamba_ssm import Mamba, Mamba2
from torch.cuda.amp import autocast
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers.activations import ACT2FN
from transformers.cache_utils import DynamicCache
# from mamba_ssm import Mamba
try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_update, causal_conv1d_fn = None, None
try:
    from mamba_ssm.ops.selective_scan_interface import mamba_inner_fn, selective_scan_fn
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update, selective_scan_fn, mamba_inner_fn = None, None, None
def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r


def semantic_neighbor(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim - 1, index=index)
    return shuffled_x

class ASSM(nn.Module):
    def __init__(self, dim, d_state=8, input_resolution=[256,256], num_tokens=64, inner_rank=128, mlp_ratio=2.):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.inner_rank = inner_rank

        # Mamba params
        self.expand = mlp_ratio
        hidden = int(self.dim * self.expand)
        self.d_state = d_state
        self.selectiveScan = Selective_Scan(d_model=hidden, d_state=self.d_state, expand=1)
        self.out_norm = nn.LayerNorm(hidden)
        self.act = nn.SiLU()
        self.out_proj = nn.Linear(hidden, dim, bias=True)

        self.in_proj = nn.Sequential(
            nn.Conv2d(self.dim, hidden, 1, 1, 0),
        )

        self.CPE = nn.Sequential(
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden),
        )

        self.embeddingB = nn.Embedding(self.num_tokens, self.inner_rank)  # [64,32] [32, 48] = [64,48]
        self.embeddingB.weight.data.uniform_(-1 / self.num_tokens, 1 / self.num_tokens)

        self.route = nn.Sequential(
            nn.Linear(self.dim, self.dim // 3),
            nn.GELU(),
            nn.Linear(self.dim // 3, self.num_tokens),
            nn.LogSoftmax(dim=-1)
        )

    def forward(self, x, x_size, token):
        B, n, C = x.shape
        H, W = x_size

        full_embedding = self.embeddingB.weight @ token.weight  # [128, C]

        pred_route = self.route(x)  # [B, HW, num_token]
        cls_policy = F.gumbel_softmax(pred_route, hard=True, dim=-1)  # [B, HW, num_token]

        prompt = torch.matmul(cls_policy, full_embedding).view(B, n, self.d_state)

        detached_index = torch.argmax(cls_policy.detach(), dim=-1, keepdim=False).view(B, n)  # [B, HW]
        x_sort_values, x_sort_indices = torch.sort(detached_index, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = x.permute(0, 2, 1).reshape(B, C, H, W).contiguous()
        x = self.in_proj(x)
        x = x * torch.sigmoid(self.CPE(x))
        cc = x.shape[1]
        x = x.view(B, cc, -1).contiguous().permute(0, 2, 1)  # b,n,c

        semantic_x = semantic_neighbor(x, x_sort_indices) # SGN-unfold
        y = self.selectiveScan(semantic_x, prompt)
        y = self.out_proj(self.out_norm(y))
        x = semantic_neighbor(y, x_sort_indices_reverse) # SGN-fold

        return x


class Selective_Scan(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.selective_scan = selective_scan_fn

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor, prompt):
        B, L, C = x.shape
        K = 1  # mambairV2 needs noly 1 scan
        xs = x.permute(0, 2, 1).view(B, 1, C, L).contiguous()  # B, 1, C ,L

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        #  our ASE here ---
        Cs = Cs.float().view(B, K, -1, L) + prompt  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, prompt, **kwargs):
        b, l, c = prompt.shape
        prompt = prompt.permute(0, 2, 1).contiguous().view(b, 1, c, l)
        y = self.forward_core(x, prompt)  # [B, L, C]
        y = y.permute(0, 2, 1).contiguous()
        return y
class NAFEVSFourierBlock(nn.Module):
    def __init__(self, c, train_size, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, DW_Fourier=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        base_size = [int(train_size[-2]*1.5), int(train_size[-1]*1.5)]
        # base_size = train_size
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.inner_rank = 32
        self.norm_mamba = LayerNorm2d(dw_channel // 2)
        self.attn = EVSBlock(dw_channel // 2, d_state=d_state)
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.fft_block2 = fft_bench_complex_linear(c, DW_Fourier, window_size=None, bias=True, act_method=nn.GELU)
        self.alpha = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    def forward(self, inp):
        x = inp

        x_norm = self.norm1(x)

        x = self.conv1(x_norm)
        x = self.conv2(x)
        x = self.sg(x)
        x_local = x * self.sca(x)

        x_local = self.conv3(x_local)

        x_global = self.norm_mamba(x)
        # b, c, h, w = x.shape
        # x_size = [h, w]
        # x_global = x_global.permute(0, 2, 3, 1).view(b, h * w, c)
        # x_norm = x_norm.view(b, c, -1)
        x_global = self.attn(x_global)
        x = x_local * self.alpha + x_global + self.fft_block2(x_norm) # .view(b, h, w, c).permute(0, 3, 1, 2)

        x = self.dropout1(x)

        y = inp + x * self.beta
        x_norm = self.norm2(y)
        x = self.conv4(x_norm)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class LocalNAFEVSBlockv1(nn.Module):
    def __init__(self, c, train_size, DW_Expand=2, FFN_Expand=2, patch_size=None,DW_Fourier=1, drop_out_rate=0., d_state=16):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.patch_size = patch_size
        base_size = [int(train_size[-2]*1.5), int(train_size[-1]*1.5)]
        # base_size = train_size
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.inner_rank = 32
        self.norm_mamba = LayerNorm2d(dw_channel // 2)
        self.attn = EVSBlock(dw_channel // 2, d_state=d_state)
        # SimpleGate
        self.sg = SimpleGate()
        # self.fft_block1 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True,
        #                                               act_method=nn.GELU)  # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True,
        #                                               act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.alpha = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        # x_fft = self.fft_block1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x_local = x * self.sca(x)

        x_local = self.conv3(x_local)

        x_global = self.norm_mamba(x)
        H, W = x_global.shape[-2:]
        # x_size = [h, w]
        # x_global = x_global.permute(0, 2, 3, 1).view(b, h * w, c)
        # x_norm = x_norm.view(b, c, -1)
        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x_global, batch_list = window_partitionx(x_global, self.patch_size)
            # print(x.shape, batch_list)
        x_global = self.attn(x_global)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)


        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x_global = window_reversex(x_global, self.patch_size, H, W, batch_list)


        x = x_local * self.alpha + x_global  # .view(b, h, w, c).permute(0, 3, 1, 2)

        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class EVSS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=8,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            patch_size=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.patch_size = patch_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),

        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(in_channels=(self.dt_rank + self.d_state * 2),
                                out_channels=(self.dt_rank + self.d_state * 2), kernel_size=7, padding=3,
                                groups=(self.dt_rank + self.d_state * 2))

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape, Bs.shape, Cs.shape, Ds.shape, dts.shape)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):
            x, batch_list = window_partitionx(x, self.patch_size)
            # print(x.shape, batch_list)
        b, c, h, w = x.shape

        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(b, h, w, -1)

        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):
            y = rearrange(y, 'b h w c -> b c h w')
            y = window_reversex(y, self.patch_size, H, W, batch_list)
            y = rearrange(y, 'b c h w -> b h w c')
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')
        return out
class HybridConbaAttentionDynamicCache(DynamicCache):
    def __init__(self, config, batch_size, dtype=torch.float16, device=None):
        self.dtype = dtype
        self.layers_block_type = config.layers_block_type
        self.has_previous_state = False
        intermediate_size = config.conba_expand * config.hidden_size
        ssm_state_size = config.conba_d_state
        conv_kernel_size = config.conba_d_conv
        self.conv_states = []
        self.ssm_states = []
        for i in range(config.num_hidden_layers):
            if self.layers_block_type[i] == "conba":
                self.conv_states += [
                    torch.zeros(batch_size, intermediate_size, conv_kernel_size, device=device, dtype=dtype)
                ]
                self.ssm_states += [
                    torch.zeros(batch_size, intermediate_size, ssm_state_size, device=device, dtype=dtype)
                ]
            else:
                self.conv_states += [torch.tensor([[]] * batch_size, device=device)]
                self.ssm_states += [torch.tensor([[]] * batch_size, device=device)]

        self.key_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]
        self.value_cache = [torch.tensor([[]] * batch_size, device=device) for _ in range(config.num_hidden_layers)]

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.key_cache[layer_idx].shape[-1] == 0:
            self.key_cache[layer_idx] = key_states
            self.value_cache[layer_idx] = value_states
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def reorder_cache(self, beam_idx: torch.LongTensor):
        for layer_idx in range(len(self.key_cache)):
            device = self.key_cache[layer_idx].device
            self.key_cache[layer_idx] = self.key_cache[layer_idx].index_select(0, beam_idx.to(device))
            device = self.value_cache[layer_idx].device
            self.value_cache[layer_idx] = self.value_cache[layer_idx].index_select(0, beam_idx.to(device))

            device = self.conv_states[layer_idx].device
            self.conv_states[layer_idx] = self.conv_states[layer_idx].index_select(0, beam_idx.to(device))
            device = self.ssm_states[layer_idx].device
            self.ssm_states[layer_idx] = self.ssm_states[layer_idx].index_select(0, beam_idx.to(device))

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        raise NotImplementedError("HybridConbaAttentionDynamicCache does not have a legacy cache equivalent.")

    @classmethod
    def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None) -> "DynamicCache":
        raise NotImplementedError("HybridConbaAttentionDynamicCache does not have a legacy cache equivalent.")
class MixConRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
class Conba2D(nn.Module):
    def __init__(
            self,
            d_model,
            conba_d_state=16,
            conba_d_conv=3,
            conba_expand=2.,
            conba_dt_rank="auto",
            hidden_act='gelu',
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            learning_rate=1e-4,
            online_learning_rate=1e-5,
            rms_norm_eps=1e-6,
            conba_conv_bias=True,
            conba_proj_bias=False,
            hidden_size=4096,
            layer_idx=0,
            bias=False,
            device=None,
            dtype=None,
            use_conba_kernels=True,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        # self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = hidden_size
        self.ssm_state_size = conba_d_state
        self.conv_kernel_size = conba_d_conv
        self.intermediate_size = int(conba_expand * hidden_size)
        self.time_step_rank = conba_dt_rank
        self.use_conv_bias = conba_conv_bias
        self.use_bias = conba_proj_bias
        self.conv1d = nn.Conv1d(
            in_channels=self.intermediate_size,
            out_channels=self.intermediate_size,
            bias=self.use_conv_bias,
            kernel_size=self.conv_kernel_size,
            groups=self.intermediate_size,
            padding=self.conv_kernel_size - 1,
        )

        self.activation = hidden_act
        self.act = ACT2FN[hidden_act]

        self.use_fast_kernels = use_conba_kernels

        self.in_proj = nn.Linear(self.hidden_size, self.intermediate_size * 2, bias=self.use_bias)
        self.x_proj = nn.Linear(self.intermediate_size, self.time_step_rank + self.ssm_state_size * 2, bias=False)
        self.dt_proj = nn.Linear(self.time_step_rank, self.intermediate_size, bias=True)

        A = torch.arange(1, self.ssm_state_size + 1, dtype=torch.float32)[None, :]
        A = A.expand(self.intermediate_size, -1).contiguous()

        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.intermediate_size))
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=self.use_bias)

        self.dt_layernorm = MixConRMSNorm(self.time_step_rank, eps=rms_norm_eps)
        self.b_layernorm = MixConRMSNorm(self.ssm_state_size, eps=rms_norm_eps)
        self.c_layernorm = MixConRMSNorm(self.ssm_state_size, eps=rms_norm_eps)

        # Dynamic state scaling mechanism
        self.scaling_function = nn.Linear(self.ssm_state_size, self.ssm_state_size, bias=True)

        # Delayed state representation
        self.delayed_state_function = nn.Linear(self.ssm_state_size, self.ssm_state_size, bias=True)

        # Adaptive Control mechanism
        self.control_gain_matrix = nn.Parameter(torch.randn(self.intermediate_size, self.intermediate_size))
        self.learning_rate = learning_rate

        # Online Learning mechanism
        self.online_learning_rate = online_learning_rate

    def cuda_kernels_forward(self, hidden_states: torch.Tensor, cache_params: HybridConbaAttentionDynamicCache = None):
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = (
                cache_params is not None
                and cache_params.has_previous_state
                and seq_len == 1
                and cache_params.conv_states[self.layer_idx].shape[0]
                == cache_params.ssm_states[self.layer_idx].shape[0]
                == batch_size
        )
        projected_states = self.in_proj(hidden_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)

        conv_weights = self.conv1d.weight.view(self.conv1d.weight.size(0), self.conv1d.weight.size(2))
        if use_precomputed_states:
            hidden_states = causal_conv1d_update(hidden_states.squeeze(-1), cache_params.conv_states[self.layer_idx],
                                                 conv_weights, self.conv1d.bias, self.activation, )
            # hidden_states = causal_conv1d_update(hidden_states.squeeze(-1), cache_params.conv_states[self.layer_idx],conv_weights,self.conv1d.bias, self.activation,)
            hidden_states = hidden_states.unsqueeze(-1)
        else:
            if cache_params is not None:
                conv_states = nn.functional.pad(hidden_states, (self.conv_kernel_size - hidden_states.shape[-1], 0))
                cache_params.conv_states[self.layer_idx].copy_(conv_states)
            # hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv1d.bias, activation=self.activation)
            hidden_states = causal_conv1d_fn(hidden_states, conv_weights, self.conv1d.bias, activation=self.activation)
        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )

        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

        time_proj_bias = self.dt_proj.bias
        self.dt_proj.bias = None
        discrete_time_step = self.dt_proj(time_step).transpose(1, 2)
        self.dt_proj.bias = time_proj_bias

        A = -torch.exp(self.A_log.float())
        time_proj_bias = time_proj_bias.float() if time_proj_bias is not None else None
        if use_precomputed_states:
            scan_outputs = selective_state_update(
                cache_params.ssm_states[self.layer_idx],
                hidden_states[..., 0],
                discrete_time_step[..., 0],
                A,
                B[:, 0],
                C[:, 0],
                self.D,
                gate[..., 0],
                time_proj_bias,
                dt_softplus=True,
            ).unsqueeze(-1)
        else:
            scan_outputs, ssm_state = selective_scan_fn(
                hidden_states,
                discrete_time_step,
                A,
                B.transpose(1, 2),
                C.transpose(1, 2),
                self.D.float(),
                gate,
                time_proj_bias,
                delta_softplus=True,
                return_last_state=True,
            )
            if ssm_state is not None and cache_params is not None:
                cache_params.ssm_states[self.layer_idx].copy_(ssm_state)

        contextualized_states = self.out_proj(scan_outputs.transpose(1, 2))

        # Dynamic state scaling mechanism
        scaled_states = self.scaling_function(contextualized_states)
        contextualized_states = contextualized_states * scaled_states

        # Delayed state representation
        delayed_states = self.delayed_state_function(contextualized_states)
        contextualized_states = contextualized_states + delayed_states

        # Adaptive Control mechanism
        desired_output = torch.zeros_like(contextualized_states)  # Placeholder for desired output
        tracking_error = contextualized_states - desired_output
        control_input = -torch.matmul(self.control_gain_matrix, tracking_error.transpose(1, 2)).transpose(1, 2)
        contextualized_states = contextualized_states + control_input

        # Online Learning mechanism
        self.control_gain_matrix = self.control_gain_matrix - self.online_learning_rate * torch.matmul(
            tracking_error.transpose(1, 2), tracking_error.transpose(1, 2).transpose(1, 2)
        )

        return contextualized_states

    def slow_forward(self, input_states, cache_params: HybridConbaAttentionDynamicCache = None):
        batch_size, seq_len, _ = input_states.shape
        dtype = input_states.dtype
        projected_states = self.in_proj(input_states).transpose(1, 2)
        hidden_states, gate = projected_states.chunk(2, dim=1)

        use_cache = isinstance(cache_params, HybridConbaAttentionDynamicCache)
        if use_cache and cache_params.ssm_states[self.layer_idx].shape[0] == batch_size:
            if self.training:
                ssm_state = cache_params.ssm_states[self.layer_idx].clone()
            else:
                ssm_state = cache_params.ssm_states[self.layer_idx]

            if cache_params.has_previous_state and seq_len == 1 and \
                    cache_params.conv_states[self.layer_idx].shape[0] == batch_size:
                conv_state = cache_params.conv_states[self.layer_idx]
                conv_state = torch.roll(conv_state, shifts=-1, dims=-1)
                conv_state[:, :, -1] = hidden_states[:, :, 0]
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = torch.sum(conv_state * self.conv1d.weight[:, 0, :], dim=-1)
                if self.use_conv_bias:
                    hidden_states += self.conv1d.bias
                hidden_states = self.act(hidden_states).to(dtype).unsqueeze(-1)
            else:
                conv_state = nn.functional.pad(
                    hidden_states,
                    (self.conv_kernel_size - hidden_states.shape[-1], 0)
                )
                cache_params.conv_states[self.layer_idx] = conv_state
                hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])
        else:
            ssm_state = torch.zeros(
                (batch_size, self.intermediate_size, self.ssm_state_size),
                device=hidden_states.device, dtype=dtype
            )
            hidden_states = self.act(self.conv1d(hidden_states)[..., :seq_len])

        ssm_parameters = self.x_proj(hidden_states.transpose(1, 2))
        time_step, B, C = torch.split(
            ssm_parameters, [self.time_step_rank, self.ssm_state_size, self.ssm_state_size], dim=-1
        )

        time_step = self.dt_layernorm(time_step)
        B = self.b_layernorm(B)
        C = self.c_layernorm(C)

        discrete_time_step = self.dt_proj(time_step)
        discrete_time_step = nn.functional.softplus(discrete_time_step).transpose(1, 2)

        A = -torch.exp(self.A_log.float())
        discrete_A = torch.exp(A[None, :, None, :] * discrete_time_step[:, :, :, None])
        discrete_B = discrete_time_step[:, :, :, None] * B[:, None, :, :].float()
        deltaB_u = discrete_B * hidden_states[:, :, :, None].float()

        scan_outputs = []
        for i in range(seq_len):
            ssm_state = discrete_A[:, :, i, :] * ssm_state + deltaB_u[:, :, i, :]
            scan_output = torch.matmul(ssm_state.to(dtype), C[:, i, :].unsqueeze(-1))
            scan_outputs.append(scan_output[:, :, 0])
        scan_output = torch.stack(scan_outputs, dim=-1)
        scan_output = scan_output + (hidden_states * self.D[None, :, None])
        scan_output = (scan_output * self.act(gate))

        if use_cache:
            cache_params.ssm_states[self.layer_idx] = ssm_state

        contextualized_states = self.out_proj(scan_output.transpose(1, 2))

        # Dynamic state scaling mechanism
        scaled_states = self.scaling_function(contextualized_states)
        contextualized_states = contextualized_states * scaled_states

        # Delayed state representation
        delayed_states = self.delayed_state_function(contextualized_states)
        contextualized_states = contextualized_states + delayed_states

        # Adaptive Control mechanism
        desired_output = torch.zeros_like(contextualized_states)  # Placeholder for desired output
        tracking_error = contextualized_states - desired_output
        control_input = -torch.matmul(self.control_gain_matrix, tracking_error.transpose(1, 2)).transpose(1, 2)
        contextualized_states = contextualized_states + control_input

        # Online Learning mechanism
        self.control_gain_matrix = self.control_gain_matrix - self.online_learning_rate * torch.matmul(
            tracking_error.transpose(1, 2), tracking_error.transpose(1, 2).transpose(1, 2)
        )

        return contextualized_states

    def forward_core(self, hidden_states, cache_params: HybridConbaAttentionDynamicCache = None):
        if self.use_fast_kernels:
            if "cuda" not in self.x_proj.weight.device.type:
                raise ValueError(
                    "Fast Conba kernels are not available."
                )
            return self.cuda_kernels_forward(hidden_states, cache_params)
        else:
            return self.slow_forward(hidden_states, cache_params)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, 'b h w c -> b c h w')

        return out
class EVSBlock(nn.Module):
    def __init__(self, dim):
        super(EVSBlock, self).__init__()


        self.norm1 = LayerNorm2d(dim)
        self.attn = EVSS2D(d_model=dim)
        # self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.register_parameter('alpha', nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True))
    def forward(self, x):

        return x + self.attn(self.norm1(x)) * self.alpha
class LocalNAFEVSBlock(nn.Module):
    def __init__(self, c, train_size, DW_Expand=2, FFN_Expand=2, patch_size=None,DW_Fourier=1, drop_out_rate=0., d_state=16, idx=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.patch_size = patch_size
        base_size = [int(train_size[-2]*1.5), int(train_size[-1]*1.5)]
        # base_size = train_size
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.idx = idx
        self.inner_rank = 32
        self.norm_mamba = LayerNorm2d(dw_channel // 2)
        self.attn = EVSS2D(dw_channel // 2, patch_size=patch_size)
        # SimpleGate
        self.sg = SimpleGate()
        # self.fft_block1 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True,
        #                                               act_method=nn.GELU)  # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True,
        #                                               act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.register_parameter('lamb_l', nn.Parameter(torch.ones((1, c, 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('lamb_g', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha', nn.Parameter(torch.ones((1, c, 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                       requires_grad=True))
        # self.lamb_l = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        # self.lamb_g = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        #
        # self.alpha = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        #
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        #
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    def forward(self, inp):
        if self.idx != -1:

            if self.idx % 2 == 1:
                inp = torch.flip(inp, dims=(-2, -1)).contiguous()
            if self.idx % 2 == 0:
                inp = torch.transpose(inp, dim0=-2, dim1=-1).contiguous()

        x = inp

        x = self.norm1(x)
        # x_fft = self.fft_block1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x_local = x * self.sca(x)

        x_local = self.conv3(x_local)

        x_global = self.norm_mamba(x)
        # H, W = x_global.shape[-2:]
        # x_size = [h, w]
        # x_global = x_global.permute(0, 2, 3, 1).view(b, h * w, c)
        # x_norm = x_norm.view(b, c, -1)
        # if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
        #     x_global, batch_list = window_partitionx(x_global, self.patch_size)
        #     # print(x.shape, batch_list)
        x_global = self.attn(x_global) * (self.alpha + 1.)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)


        # if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
        #     x_global = window_reversex(x_global, self.patch_size, H, W, batch_list)


        x = x_local * self.lamb_l + x_global * self.lamb_g  # .view(b, h, w, c).permute(0, 3, 1, 2)

        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class NAFEVSBlock(nn.Module):
    def __init__(self, c, train_size, DW_Expand=2, FFN_Expand=2, patch_size=None,DW_Fourier=1, drop_out_rate=0., d_state=16, idx=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.patch_size = patch_size
        base_size = [int(train_size[-2]*1.5), int(train_size[-1]*1.5)]
        # base_size = train_size
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.idx = idx
        self.inner_rank = 32
        self.norm_mamba = LayerNorm2d(dw_channel // 2)
        self.attn = EVSS2D(dw_channel // 2, patch_size=patch_size)
        # SimpleGate
        self.sg = SimpleGate()
        # self.fft_block1 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True,
        #                                               act_method=nn.GELU)  # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True,
        #                                               act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.register_parameter('lamb_l', nn.Parameter(torch.ones((1, c, 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('lamb_g', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('alpha', nn.Parameter(torch.ones((1, c, 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                       requires_grad=True))
        # self.lamb_l = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        # self.lamb_g = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        #
        # self.alpha = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        #
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        #
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    def forward(self, inp):
        if self.idx != -1:

            if self.idx % 2 == 1:
                inp = torch.flip(inp, dims=(-2, -1)).contiguous()
            if self.idx % 2 == 0:
                inp = torch.transpose(inp, dim0=-2, dim1=-1).contiguous()

        x = inp

        x = self.norm1(x)
        # x_fft = self.fft_block1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x_local = x * self.sca(x)

        x_local = self.conv3(x_local)

        x_global = self.norm_mamba(x)
        # H, W = x_global.shape[-2:]
        # x_size = [h, w]
        # x_global = x_global.permute(0, 2, 3, 1).view(b, h * w, c)
        # x_norm = x_norm.view(b, c, -1)
        # if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
        #     x_global, batch_list = window_partitionx(x_global, self.patch_size)
        #     # print(x.shape, batch_list)
        x_global = self.attn(x_global) * (self.alpha + 1.)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)


        # if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
        #     x_global = window_reversex(x_global, self.patch_size, H, W, batch_list)


        x = x_local * self.lamb_l + x_global * self.lamb_g  # .view(b, h, w, c).permute(0, 3, 1, 2)

        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class LocaNAFEVSBlock(nn.Module):
    def __init__(self, c, train_size, DW_Expand=2, FFN_Expand=2, patch_size=None,DW_Fourier=1, drop_out_rate=0., d_state=16, idx=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.patch_size = patch_size
        base_size = [int(train_size[-2]*1.5), int(train_size[-1]*1.5)]
        # base_size = train_size
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.idx = idx
        self.inner_rank = 32
        self.norm_mamba = LayerNorm2d(dw_channel // 2)
        self.attn = EVSS2D(dw_channel // 2)
        # SimpleGate
        self.sg = SimpleGate()
        # self.fft_block1 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True,
        #                                               act_method=nn.GELU)  # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True,
        #                                               act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.lamb_l = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.lamb_g = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.alpha = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    def forward(self, inp):
        if self.idx % 2 == 1:
            inp = torch.flip(inp, dims=(-2, -1)).contiguous()
        if self.idx % 2 == 0:
            inp = torch.transpose(inp, dim0=-2, dim1=-1).contiguous()

        x = inp

        x = self.norm1(x)
        # x_fft = self.fft_block1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x_local = x * self.sca(x)

        x_local = self.conv3(x_local)

        x_global = self.norm_mamba(x)
        H, W = x_global.shape[-2:]
        # x_size = [h, w]
        # x_global = x_global.permute(0, 2, 3, 1).view(b, h * w, c)
        # x_norm = x_norm.view(b, c, -1)
        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x_global, batch_list = window_partitionx(x_global, self.patch_size)
            # print(x.shape, batch_list)
        x_global = self.attn(x_global) * (self.alpha + 1.)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)


        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x_global = window_reversex(x_global, self.patch_size, H, W, batch_list)


        x = x_local * self.lamb_l + x_global * self.lamb_g  # .view(b, h, w, c).permute(0, 3, 1, 2)

        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.norm2(y)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class LocalNAFEVSBlock_FourierSplit(nn.Module):
    def __init__(self, c, train_size, DW_Expand=2, FFN_Expand=2, patch_size=None, window_size_fft=None,DW_Fourier=1, drop_out_rate=0., d_state=16):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        base_size = [int(train_size[-2]*1.5), int(train_size[-1]*1.5)]
        # base_size = train_size
        self.patch_size = patch_size
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.inner_rank = 32
        self.norm_mamba = LayerNorm2d(dw_channel // 2)
        self.attn = EVSBlock(dw_channel // 2, d_state=d_state)
        # SimpleGate
        self.sg = SimpleGate()
        self.fft_block1 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True,
                                                      act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True,
                                                      act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.alpha = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x_fft = self.fft_block1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x_local = x * self.sca(x)

        x_local = self.conv3(x_local)

        x_global = self.norm_mamba(x)
        H, W = x_global.shape[-2:]
        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x_global, batch_list = window_partitionx(x_global, self.patch_size)
            # print(x.shape, batch_list)
        x_global = self.attn(x_global)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x_global = window_reversex(x_global, self.patch_size, H, W, batch_list)
        x = x_local * self.alpha + x_global + x_fft # .view(b, h, w, c).permute(0, 3, 1, 2)

        x = self.dropout1(x)

        y = inp + x * self.beta
        x = self.norm2(y)
        x_fft = self.fft_block2(x)
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x) + x_fft

        x = self.dropout2(x)

        return y + x * self.gamma
class LocalNAFEVSDFFNBlock(nn.Module):
    def __init__(self, c, train_size, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        base_size = [int(train_size[-2]*1.5), int(train_size[-1]*1.5)]
        # base_size = train_size
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.inner_rank = 32
        self.norm_mamba = LayerNorm2d(dw_channel // 2)
        self.attn = EVSBlock(dw_channel // 2, d_state=d_state)
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.ffn = DFFN(c, ffn_channel, True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # self.beta = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.alpha = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x_local = x * self.sca(x)

        x_local = self.conv3(x_local)

        x_global = self.norm_mamba(x)
        # b, c, h, w = x.shape
        # x_size = [h, w]
        # x_global = x_global.permute(0, 2, 3, 1).view(b, h * w, c)
        # x_norm = x_norm.view(b, c, -1)
        x_global = self.attn(x_global)
        x = x_local * self.alpha + x_global # .view(b, h, w, c).permute(0, 3, 1, 2)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.ffn(self.norm2(y))

        x = self.dropout2(x)

        return y + x * self.gamma
class LocalNAFSpaGSBlock(nn.Module):
    def __init__(self, c, train_size, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., num_heads=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        base_size = [int(train_size[-2]*1.5), int(train_size[-1]*1.5)]
        # base_size = train_size
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.inner_rank = 32
        # self.norm_mamba = LayerNorm2d(dw_channel // 2)
        self.attn = SpaGSBlock(dw_channel // 2, num_heads=num_heads)
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.alpha = nn.Parameter(torch.ones((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x_local = x * self.sca(x)

        x_local = self.conv3(x_local)

        # x_global = self.norm_mamba(x)
        # b, c, h, w = x.shape
        # x_size = [h, w]
        # x_global = x_global.permute(0, 2, 3, 1).view(b, h * w, c)
        # x_norm = x_norm.view(b, c, -1)
        x_global = self.attn(x)
        x = x_local * self.alpha + x_global # .view(b, h, w, c).permute(0, 3, 1, 2)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class SpaGSBlock(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(SpaGSBlock, self).__init__()

        # self.norm1 = LayerNorm2d(dim)
        self.qkv = nn.Linear(dim,  dim * 4)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.act = nn.GELU()
        self.out_norm = nn.LayerNorm(dim)
        self.out_proj = nn.Linear(dim,  dim)
    def forward(self, x):
        # x_norm = self.norm1(x)
        b, c, h, w = x.shape
        # x_size = [h, w]
        x_norm = x.permute(0, 2, 3, 1).view(b, h * w, c)
        q, k, v, g = self.qkv(x_norm).chunk(4, dim=-1)
        # x_norm = x_norm.view(b, c, -1)
        x_attn, _ = self.attn(q, k, v)
        x_out = self.out_norm(x_attn) * self.act(g)
        x_out = self.out_proj(x_out)
        return x_out.view(b, h, w, c).permute(0, 3, 1, 2)
class ASSMBlock(nn.Module):
    def __init__(self, dim, d_state=8, ffn_expansion_factor=3., train_size=[256, 256], bias=True, att=True):
        super(ASSMBlock, self).__init__()

        self.att = att
        if self.att:
            self.inner_rank = 32
            self.norm1 = LayerNorm2d(dim)
            self.attn = ASSM(dim, d_state=d_state, input_resolution=train_size, inner_rank=self.inner_rank)
            self.embeddingA = nn.Embedding(self.inner_rank, d_state)
            self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)
            self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.att:
            x_norm = self.norm1(x)
            b, c, h, w = x.shape
            x_size = [h, w]
            x_norm = x_norm.permute(0, 2, 3, 1).view(b, h * w, c)
            # x_norm = x_norm.view(b, c, -1)
            x_attn = self.attn(x_norm, x_size, self.embeddingA)
            x = x + x_attn.view(b, h, w, c).permute(0, 3, 1, 2) * self.beta

        x = x + self.ffn(self.norm2(x)) * self.gamma

        return x
class AVSBlock(nn.Module):
    def __init__(self, dim, d_state,
                 input_resolution,
                 inner_rank,
                 num_tokens,
                 mlp_ratio):
        super(AVSBlock, self).__init__()

        self.inner_rank = inner_rank
        self.norm1 = LayerNorm2d(dim)
        self.attn = ASSM(dim, d_state=d_state, num_tokens=num_tokens, input_resolution=input_resolution, inner_rank=self.inner_rank, mlp_ratio=mlp_ratio)
        self.embeddingA = nn.Embedding(self.inner_rank, d_state)
        self.embeddingA.weight.data.uniform_(-1 / self.inner_rank, 1 / self.inner_rank)

    def forward(self, x):
        x_norm = self.norm1(x)
        b, c, h, w = x.shape
        x_size = [h, w]
        x_norm = x_norm.permute(0, 2, 3, 1).view(b, h * w, c)
        # x_norm = x_norm.view(b, c, -1)
        x_attn = self.attn(x_norm, x_size, self.embeddingA)
        x = x + x_attn.view(b, h, w, c).permute(0, 3, 1, 2)


        return x

class SS2D(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):

        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1, y2, y3, y4 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2)
        return out

class ShuffledMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Conv2d(self.d_model, self.d_inner * 2, kernel_size=1, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()
        self.unshuffle = nn.PixelUnshuffle(2)
        self.shuffle = nn.PixelShuffle(2)
        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = LayerNorm2d(self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model, kernel_size=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x):
        x = self.unshuffle(x)
        x1, x2, x3, x4 = x.chunk(4, dim=1)
        B, C, H, W = x1.shape
        L = H * W
        K = 4
        x_hwwh = torch.stack([x1.view(B, -1, L), torch.transpose(x2, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        x_whhw = torch.stack([x3.view(B, -1, L), torch.transpose(x4, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_whhw, dims=[-1])], dim=1)  # (1, 4, 192, 3136)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)

        out_y1 = out_y[:, 0].view(B, C, H, W)
        out_y2 = inv_y[:, 0].view(B, C, H, W)
        out_y3 = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous() # .view(B, -1, L)
        out_y4 = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous() # .view(B, -1, L)

        out = self.shuffle(torch.cat([out_y1, out_y2, out_y3, out_y4], dim=1))
        return out

    def forward(self, x: torch.Tensor, **kwargs):

        # x = x.permute(0, 2, 3, 1)
        # B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)

        # x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y = self.forward_core(x)
        assert y.dtype == torch.float32
        # y = y1 + y2 + y3 + y4
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        # out = out.permute(0, 3, 1, 2)
        return out
class MaskedMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            thr=0.0156,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.thr = thr
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        # self.point_conv = nn.Conv2d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     groups=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=1,
        #     padding=0,
        #     **factory_kwargs,
        # )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, L = x.shape
        # L = H * W
        K = 2
        # x = x.view(B, -1, L)
        x = x.unsqueeze(1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x, torch.flip(x, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        # xs = xs.transpose(2, 3)
        # print(xs.shape, self.x_proj_weight.shape, K, C, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        out_y, inv_y = out_y.chunk(2, dim=1)
        inv_y = torch.flip(inv_y, dims=[-1])
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y.view(B, -1, L), inv_y.view(B, -1, L)  # , wh_y, invwh_y

    def forward(self, x: torch.Tensor, mask: torch.Tensor, **kwargs):

        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        mask_sharp, mask_blur = mask.chunk(2, dim=1)
        mask = mask_blur.squeeze(1)
        thr = torch.sum(mask, dim=[1, 2])
        bs_blur_idx = torch.where(thr > self.thr * H * W)[0]
        bsharp_mask = torch.ones_like(thr)
        bsharp_mask[bs_blur_idx] = 0.
        # print(bs_blur_idx)
        ch = x.shape[1]
        mask_blur = mask.ge(0.5)
        mam_list = []
        for i in bs_blur_idx:
            # if torch.sum(mask[i, 1, :, :]) == 0:
            #     continue
            # print(i, mask_blur.shape)
            mask_ = mask_blur[i, :, :].contiguous()
            # mask_ = torch.index_select(mask_blur, dim=0, index=i)
            y_z = x[i, ...].clone()
            # y_zero = torch.zeros_like(y_z)
            # print(y_z.shape, mask_.shape)
            y_zf = torch.masked_select(y_z, mask_)
            y_zf = y_zf.view(1, ch, -1)
            # print(y_zf.shape)
            y1, y2 = self.forward_core(y_zf)
            assert y1.dtype == torch.float32
            y_ = y1 + y2
            y_z = y_z.masked_scatter(mask_, y_.view(1, -1))
            # y_z = y_zero.masked_scatter(mask_, y_.flatten())
            mam_list.append(y_z.unsqueeze(0))

        if len(mam_list) > 0:
            # x = x * mask_sharp.float()
            # mamba_out = torch.cat(mam_list, dim=0)
            # y = x.index_add_(0, bs_blur_idx, mamba_out, alpha=1)
            x = x * bsharp_mask.view(B, 1, 1, 1).float()
            mamba_out = torch.cat(mam_list, dim=0)
            y = x.index_add_(0, bs_blur_idx, mamba_out, alpha=1)
        else:
            y = x
        # y = self.point_conv(y)
        # x = x.permute(0, 2, 1)

        y = y.permute(0, 2, 3, 1)
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2)
        return out

class VMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            thr=0.0156,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.thr = thr
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        # self.point_conv = nn.Conv2d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     groups=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=1,
        #     padding=0,
        #     **factory_kwargs,
        # )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, L = x.shape
        # L = H * W
        K = 2
        # x = x.view(B, -1, L)
        x = x.unsqueeze(1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x, torch.flip(x, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        # xs = xs.transpose(2, 3)
        # print(xs.shape, self.x_proj_weight.shape, K, C, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        out_y, inv_y = out_y.chunk(2, dim=1)
        inv_y = torch.flip(inv_y, dims=[-1])
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y.view(B, -1, L), inv_y.view(B, -1, L)  # , wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):

        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y1, y2 = self.forward_core(x.view(B, x.shape[1], -1))
        assert y1.dtype == torch.float32
        y = y1 + y2
        y = y.view(B, -1, H, W)

        # y = self.point_conv(y)
        # x = x.permute(0, 2, 1)

        y = y.permute(0, 2, 3, 1)
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2)
        return out

class VMambaF(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            thr=0.0156,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.thr = thr
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Conv2d(self.d_model, self.d_inner * 2, kernel_size=1, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        # self.point_conv = nn.Conv2d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     groups=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=1,
        #     padding=0,
        #     **factory_kwargs,
        # )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = LayerNorm2d(self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model, kernel_size=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, L = x.shape
        # L = H * W
        K = 1
        # x = x.view(B, -1, L)
        x = x.unsqueeze(1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = x # torch.cat([x, torch.flip(x, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        # xs = xs.transpose(2, 3)
        # print(xs.shape, self.x_proj_weight.shape, K, C, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        # out_y, inv_y = out_y.chunk(2, dim=1)
        # inv_y = torch.flip(inv_y, dims=[-1])
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y.view(B, -1, L) # , inv_y.view(B, -1, L)  # , wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):

        # x = x.permute(0, 2, 3, 1)
        # B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)

        # x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        B, _, H, W = x.shape
        y = self.forward_core(x.view(B, x.shape[1], -1))
        assert y.dtype == torch.float32
        # y = y1 + y2
        y = y.view(B, -1, H, W)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        # out = out.permute(0, 3, 1, 2)
        return out
class EVSSBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3., bias=True, att=True):
        super(EVSSBlock, self).__init__()

        self.att = att
        if self.att:
            self.norm1 = LayerNorm2d(dim)
            self.attn = EVSS2D(dim)

        self.norm2 = LayerNorm2d(dim)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.att:
            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x
class EVSSNAFBlock(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3., bias=True, att=True):
        super(EVSSNAFBlock, self).__init__()

        self.att = att
        if self.att:
            self.norm1 = LayerNorm2d(dim)
            self.attn = EVSS2D(dim)

        # self.norm2 = LayerNorm2d(dim)
        self.ffn = NAFBlock(dim)

    def forward(self, x):
        if self.att:
            x = x + self.attn(self.norm1(x))

        # x = self.ffn(x)

        return self.ffn(x)


class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, global_fft=False):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)
        self.global_fft = global_fft
        if global_fft:
            self.g_fft = fft_bench_complex_linear(dim, dw=1)
            self.project_out = nn.Conv2d(hidden_features + dim, dim, kernel_size=1, bias=bias)
        else:
            self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)
        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)
        self.register_parameter('fft', nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)),
                                                    requires_grad=True))
        # self.fft = nn.Parameter(torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1)))

    def forward(self, x):
        if self.global_fft:
            x_global = self.g_fft(x)
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        if self.global_fft:
            x = torch.cat([x, x_global], dim=1)
        x = self.project_out(x)
        # if self.global_fft:
        #     x += x_global
        b, c, h, w = x.shape
        h_n = (8 - h % 8) % 8
        w_n = (8 - w % 8) % 8

        x = torch.nn.functional.pad(x, (0, w_n, 0, h_n), mode='reflect')
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)

        x = x[:, :, :h, :w]

        return x


class EDFeedForwardNetwork(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=False, global_fft=False):
        super(EDFeedForwardNetwork, self).__init__()


        self.norm2 = LayerNorm2d(dim)
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias, global_fft)

        self.register_parameter('beta', nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True))


    def forward(self, x):

        x = x * self.beta + self.ffn(self.norm2(x)) #

        return x
class EVSS(nn.Module):
    def __init__(self, dim, patch_size=None):
        super(EVSS, self).__init__()

        # self.att = att
        # if self.att:
        self.norm1 = LayerNorm2d(dim)
        self.attn = EVSS2D(dim, patch_size=patch_size)

        # self.norm2 = LayerNorm2d(dim)
        # self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # if self.att:
        x = x + self.attn(self.norm1(x))

        # x = x + self.ffn(self.norm2(x))

        return x

class EVSS_beta(nn.Module):
    def __init__(self, dim, patch_size=None):
        super(EVSS_beta, self).__init__()

        # self.att = att
        # if self.att:
        self.norm1 = LayerNorm2d(dim)
        self.attn = EVSS2D(dim, patch_size=patch_size)
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True))
        # self.norm2 = LayerNorm2d(dim)
        # self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        # if self.att:
        x = x * self.beta + self.attn(self.norm1(x))

        # x = x + self.ffn(self.norm2(x))

        return x
class EVSBlock(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            thr=0.0156,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.thr = thr
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Conv2d(self.d_model, self.d_inner * 2, kernel_size=1, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        # self.point_conv = nn.Conv2d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     groups=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=1,
        #     padding=0,
        #     **factory_kwargs,
        # )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = LayerNorm2d(self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model, kernel_size=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, L = x.shape
        # L = H * W
        K = 1
        # x = x.view(B, -1, L)
        x = x.unsqueeze(1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = x # torch.cat([x, torch.flip(x, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        # xs = xs.transpose(2, 3)
        # print(xs.shape, self.x_proj_weight.shape, K, C, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        # out_y, inv_y = out_y.chunk(2, dim=1)
        # inv_y = torch.flip(inv_y, dims=[-1])
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y.view(B, -1, L) # , inv_y.view(B, -1, L)  # , wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):

        # x = x.permute(0, 2, 3, 1)
        # B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)

        # x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        B, _, H, W = x.shape
        y = self.forward_core(x.view(B, x.shape[1], -1))
        assert y.dtype == torch.float32
        # y = y1 + y2
        y = y.view(B, -1, H, W)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        # out = out.permute(0, 3, 1, 2)
        return out
class VMambaFv1(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            thr=0.0156,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.thr = thr
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        # self.point_conv = nn.Conv2d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     groups=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=1,
        #     padding=0,
        #     **factory_kwargs,
        # )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, L = x.shape
        # L = H * W
        K = 1
        # x = x.view(B, -1, L)
        x = x.unsqueeze(1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = x # torch.cat([x, torch.flip(x, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        # xs = xs.transpose(2, 3)
        # print(xs.shape, self.x_proj_weight.shape, K, C, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        # out_y, inv_y = out_y.chunk(2, dim=1)
        # inv_y = torch.flip(inv_y, dims=[-1])
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y.view(B, -1, L) # , inv_y.view(B, -1, L)  # , wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):

        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))

        y = self.forward_core(x.view(B, x.shape[1], -1))
        assert y.dtype == torch.float32
        # y = y1 + y2
        y = y.view(B, -1, H, W)

        # y = self.point_conv(y)
        # x = x.permute(0, 2, 1)

        y = y.permute(0, 2, 3, 1)
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2)
        return out
class DownScaleMambaF(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            thr=0.0156,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.thr = thr
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Conv2d(self.d_model, self.d_inner * 2, kernel_size=1, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=2,
            stride=2,
            padding=0,
            **factory_kwargs,
        )
        # self.upconv = nn.ConvTranspose2d(self.d_inner, self.d_inner, kernel_size=2, stride=2)
        # self.point_conv = nn.Conv2d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     groups=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=1,
        #     padding=0,
        #     **factory_kwargs,
        # )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = LayerNorm2d(self.d_inner)
        self.out_proj = nn.Conv2d(self.d_inner, self.d_model,kernel_size=1, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, L = x.shape
        # L = H * W
        K = 1
        # x = x.view(B, -1, L)
        x = x.unsqueeze(1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = x # torch.cat([x, torch.flip(x, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        # xs = xs.transpose(2, 3)
        # print(xs.shape, self.x_proj_weight.shape, K, C, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        # out_y, inv_y = out_y.chunk(2, dim=1)
        # inv_y = torch.flip(inv_y, dims=[-1])
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y.view(B, -1, L) # , inv_y.view(B, -1, L)  # , wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)

        # x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        # x = x.permute(0, 2, 3, 1).contiguous()
        B, _, H, W = x.shape
        y = self.forward_core(x.view(B, x.shape[1], -1))
        assert y.dtype == torch.float32
        # y = y1 + y2
        y = y.view(B, -1, H, W)
        y = F.interpolate(y, scale_factor=2)
        # y = y.permute(0, 2, 3, 1)
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        # y = self.upconv(y)

        y = y * F.gelu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        # out = out
        return out
class LocalMambaF(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            thr=0.0156,
            window_size=8,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.thr = thr
        self.window_size = window_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        # self.point_conv = nn.Conv2d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     groups=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=1,
        #     padding=0,
        #     **factory_kwargs,
        # )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, L = x.shape
        # L = H * W
        K = 1
        # x = x.view(B, -1, L)
        x = x.unsqueeze(1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = x # torch.cat([x, torch.flip(x, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        # xs = xs.transpose(2, 3)
        # print(xs.shape, self.x_proj_weight.shape, K, C, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        # out_y, inv_y = out_y.chunk(2, dim=1)
        # inv_y = torch.flip(inv_y, dims=[-1])
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y.view(B, -1, L) # , inv_y.view(B, -1, L)  # , wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):

        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        C_ = x.shape[1]
        x = x.view(B, C_, H // self.window_size, self.window_size, W // self.window_size, self.window_size)
        # print(x.permute(0, 1, 2, 4, 3, 5).shape)
        x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C_, -1, self.window_size*self.window_size)
        y = self.forward_core(x.view(B, x.shape[1], -1))
        y = y.view(B, C_, H // self.window_size, W // self.window_size, self.window_size, self.window_size).permute(0, 1, 2, 4, 3, 5).contiguous()

        assert y.dtype == torch.float32
        # y = y1 + y2
        y = y.view(B, -1, H, W)

        # y = self.point_conv(y)
        # x = x.permute(0, 2, 1)

        y = y.permute(0, 2, 3, 1)
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2)
        return out
def index_reverse(index):
    index_r = torch.zeros_like(index)
    ind = torch.arange(0, index.shape[-1]).to(index.device)
    for i in range(index.shape[0]):
        index_r[i, index[i, :]] = ind
    return index_r

def feature_shuffle(x, index):
    dim = index.dim()
    assert x.shape[:dim] == index.shape, "x ({:}) and index ({:}) shape incompatible".format(x.shape, index.shape)

    for _ in range(x.dim() - index.dim()):
        index = index.unsqueeze(-1)
    index = index.expand(x.shape)

    shuffled_x = torch.gather(x, dim=dim-1, index=index)
    return shuffled_x


class ATD_CA(nn.Module):
    r"""
    Adaptive Token Dictionary Cross-Attention.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        num_tokens (int): Number of tokens in external token dictionary. Default: 64
        reducted_dim (int, optional): Reducted dimension number for query and key matrix. Default: 4
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
    """

    def __init__(self, dim, num_tokens=64, reducted_dim=16, qkv_bias=True):
        super().__init__()
        self.dim = dim
        # self.input_resolution = input_resolution
        self.num_tokens = num_tokens
        self.rc = reducted_dim
        self.qkv_bias = qkv_bias
        self.wqtd = nn.Linear(dim, num_tokens, bias=qkv_bias)
        self.wktd = nn.Linear(dim, dim, bias=qkv_bias)
        self.wq = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wk = nn.Linear(dim, reducted_dim, bias=qkv_bias)
        self.wv = nn.Linear(dim, dim, bias=qkv_bias)

        self.scale = nn.Parameter(torch.ones([self.num_tokens]) * 0.5, requires_grad=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        r"""
        Args:
            x: input features with shape of (b, n, c)
            td: token dicitionary with shape of (b, m, c)
            x_size: size of the input x (h, w)
        """
        b, n, c = x.shape

        q_td = self.wqtd(x)
        k_td = self.wktd(x)

        td = (F.normalize(q_td, dim=-1).transpose(-2, -1) @ F.normalize(k_td, dim=-1))  # (b, m, c)
        b, m, c = td.shape

        # Q: b, n, c
        q = self.wq(x)
        # K: b, m, c
        k = self.wk(td)
        # V: b, m, c
        v = self.wv(td)

        # Q @ K^T
        attn = (F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1))  # b, n, n_tk
        scale = torch.clamp(self.scale, 0, 1)
        attn = attn * (1 + scale * np.log(self.num_tokens))
        attn = self.softmax(attn)

        # Attn * V
        x = (attn @ v).reshape(b, n, c)

        return x, attn

    def flops(self, n):
        n_tk = self.num_tokens
        flops = 0
        # qkv = self.wq(x)
        flops += n * self.dim * self.rc
        # k = self.wk(gc)
        flops += n_tk * self.dim * self.rc
        # v = self.wv(gc)
        flops += n_tk * self.dim * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += n * self.dim * self.rc
        #  x = (attn @ v)
        flops += n * n_tk * self.dim

        return flops
class ATDMambaF(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            thr=0.0156,
            num_mask=8,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.num_mask = num_mask
        self.thr = thr
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Conv2d(self.d_model, self.d_inner * 2, kernel_size=1, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.mask_proj = nn.Conv2d(self.d_inner, self.num_mask, kernel_size=1, bias=bias, **factory_kwargs)
        self.sim_proj = nn.Conv2d(self.num_mask, self.d_inner, kernel_size=1, bias=bias, **factory_kwargs)
        # self.sim_proj = ATD_CA(self.d_inner, self.num_mask, 16)
        # self.point_conv = nn.Conv2d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     groups=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=1,
        #     padding=0,
        #     **factory_kwargs,
        # )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=1, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, L = x.shape
        # L = H * W
        K = 1
        # x = x.view(B, -1, L)
        x = x.unsqueeze(1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = x # torch.cat([x, torch.flip(x, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        # xs = xs.transpose(2, 3)
        # print(xs.shape, self.x_proj_weight.shape, K, C, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        # out_y, inv_y = out_y.chunk(2, dim=1)
        # inv_y = torch.flip(inv_y, dims=[-1])
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y.view(B, -1, L) # , inv_y.view(B, -1, L)  # , wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):

        # x = x.permute(0, 2, 3, 1)
        B, C, H, W = x.shape
        # print(x.shape)
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=1)

        # x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        sim = self.mask_proj(x)
        z = self.sim_proj(sim) * z
        z = z.permute(0, 2, 3, 1)
        # z = z.view(B, z.shape[1], -1)
        # z, sim = self.sim_proj(z.permute(0, 2, 1))
        # z = z.view(B, H, W, -1)
        sim = sim.view(B, sim.shape[1], -1).permute(0, 2, 1).contiguous()
        x = x.view(B, x.shape[1], -1).permute(0, 2, 1).contiguous()
        # sim = torch.softmax(sim, dim=-1)
        # sim = F.gumbel_softmax(sim, tau=1,hard=True,dim=-1)
        tk_id = torch.argmax(sim, dim=-1, keepdim=False)
        # print(sim.shape, tk_id.shape)
        # _, tk_id = torch.split(tk_id, [1, m-1], dim=-1)
        # sort features by type
        x_sort_values, x_sort_indices = torch.sort(tk_id, dim=-1, stable=False)
        x_sort_indices_reverse = index_reverse(x_sort_indices)

        x = feature_shuffle(x, x_sort_indices)

        y = self.forward_core(x.permute(0, 2, 1).contiguous())
        assert y.dtype == torch.float32
        y = feature_shuffle(y.permute(0, 2, 1), x_sort_indices_reverse)
        # y = y1 + y2
        y = y.view(B, H, W, -1)

        # y = self.point_conv(y)
        # x = x.permute(0, 2, 1)

        # y = y.permute(0, 2, 3, 1)
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.gelu(z) # .permute(0, 2, 3, 1)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2)
        return out
class MaskedMambaV1(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, L = x.shape
        # L = H * W
        K = 2
        # x = x.view(B, -1, L)
        x = x.unsqueeze(1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x, torch.flip(x, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        # xs = xs.transpose(2, 3)
        # print(xs.shape, self.x_proj_weight.shape, K, C, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        out_y, inv_y = out_y.chunk(2, dim=1)
        inv_y = torch.flip(inv_y, dims=[-1])
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y.view(B, -1, L), inv_y.view(B, -1, L)  # , wh_y, invwh_y

    def forward(self, x: torch.Tensor, mask: torch.Tensor, **kwargs):

        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        mask_sharp, mask_blur = mask.chunk(2, dim=1)
        mask = mask_blur.squeeze(1)
        bs_blur_idx = torch.where(torch.sum(mask, dim=[1, 2]) > 0)[0]
        # print(bs_blur_idx)
        ch = x.shape[1]
        mask_blur = mask.ge(0.5)
        mam_list = []
        for i in bs_blur_idx:
            # if torch.sum(mask[i, 1, :, :]) == 0:
            #     continue
            # print(i, mask_blur.shape)
            mask_ = mask_blur[i, :, :].contiguous()
            # mask_ = torch.index_select(mask_blur, dim=0, index=i)
            y_z = x[i, ...].clone()
            y_zero = torch.zeros_like(y_z)
            # print(y_z.shape, mask_.shape)
            y_zf = torch.masked_select(y_z, mask_)
            y_zf = y_zf.view(1, ch, -1)
            # print(y_zf.shape)
            y1, y2 = self.forward_core(y_zf)
            assert y1.dtype == torch.float32
            y_ = y1 + y2
            # y_ = y_.permute(0, 2, 1)
            y_z = y_zero.masked_scatter(mask_, y_.flatten())
            mam_list.append(y_z.unsqueeze(0))
        if len(mam_list) > 0:
            x = x * mask_sharp.float()
            mamba_out = torch.cat(mam_list, dim=0)
            y = x.index_add_(0, bs_blur_idx, mamba_out, alpha=1)
        else:
            y = x
        # x = x.permute(0, 2, 1)

        y = y.permute(0, 2, 3, 1)
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2)
        return out
class PatchMaskedMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            patch_size=[64, 64],
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.patch_size = patch_size
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )

        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, L = x.shape
        # L = H * W
        K = 2
        # x = x.view(B, -1, L)
        x = x.unsqueeze(1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x, torch.flip(x, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        # xs = xs.transpose(2, 3)
        # print(xs.shape, self.x_proj_weight.shape, K, C, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        out_y, inv_y = out_y.chunk(2, dim=1)
        inv_y = torch.flip(inv_y, dims=[-1])
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y.view(B, -1, L), inv_y.view(B, -1, L)  # , wh_y, invwh_y

    def forward(self, x: torch.Tensor, mask: torch.Tensor, **kwargs):

        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        x, batch_list = window_partitionx(x, self.patch_size)
        # print(mask.shape, x.shape, self.patch_size)
        bs_blur_idx = torch.where(torch.sum(mask, dim=[1, 2]) > 0)[0]
        # print(bs_blur_idx)
        ch = x.shape[1]
        mask_sharp, mask_blur = mask.chunk(2, dim=1)
        mask = mask_blur.squeeze(1)
        mask_blur = mask.ge(0.5)
        mam_list = []
        for i in bs_blur_idx:
            # if torch.sum(mask[i, 1, :, :]) == 0:
            #     continue
            # print(i, mask_blur.shape)
            mask_ = mask_blur[i, :, :].contiguous()
            # mask_ = torch.index_select(mask_blur, dim=0, index=i)
            y_z = x[i, ...].clone()
            y_zero = torch.zeros_like(y_z)
            y_zf = torch.masked_select(y_z, mask_)
            y_zf = y_zf.view(1, ch, -1)
            # print(y_zf.shape)
            y1, y2 = self.forward_core(y_zf)
            assert y1.dtype == torch.float32
            y_ = y1 + y2
            # y_ = y_.permute(0, 2, 1)
            y_z = y_zero.masked_scatter(mask_, y_.flatten())
            mam_list.append(y_z.unsqueeze(0))
        if len(mam_list) > 0:
            x = x * mask_sharp.float()
            mamba_out = torch.cat(mam_list, dim=0)
            y = x.index_add_(0, bs_blur_idx, mamba_out, alpha=1)
        else:
            y = x
        # x = x.permute(0, 2, 1)
        y = window_reversex(y, self.patch_size, H, W, batch_list)
        y = y.permute(0, 2, 3, 1)
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        out = out.permute(0, 3, 1, 2)
        return out

class FRMamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=3,
            expand=2.,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            device=None,
            dtype=None,
            **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        # self.conv2d = nn.Conv2d(
        #     in_channels=self.d_inner,
        #     out_channels=self.d_inner,
        #     groups=self.d_inner,
        #     bias=conv_bias,
        #     kernel_size=d_conv,
        #     padding=(d_conv - 1) // 2,
        #     **factory_kwargs,
        # )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
            # nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs),
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
                         **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
            # self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor,
            #              **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=2, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True)  # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True)  # (K=2, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, L = x.shape
        # L = H * W
        K = 2
        # x = x.view(B, -1, L)
        x = x.unsqueeze(1)
        # x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x, torch.flip(x, dims=[-1])], dim=1)  # (1, 4, 192, 3136)
        # xs = xs.transpose(2, 3)
        # print(xs.shape, self.x_proj_weight.shape, K, C, L)
        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape)
        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float
        out_y, inv_y = out_y.chunk(2, dim=1)
        inv_y = torch.flip(inv_y, dims=[-1])
        # wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        # invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y.view(B, -1, L), inv_y.view(B, -1, L)  # , wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):

        # x = x.permute(0, 2, 3, 1)
        # B, H, W, C = x.shape

        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        # x = x.permute(0, 3, 1, 2).contiguous()
        # x = self.act(self.conv2d(x))
        x = x.permute(0, 2, 1)
        y1, y2 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1 + y2
        y = y.permute(0, 2, 1)
        # y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        # out = out.permute(0, 3, 1, 2)
        return out


def MLP_cubic(h, w, c, deep):
    return nn.Sequential(*[nn.Sequential(
        nn.Linear(w, w),
        nn.PReLU(),
        Rearrange('b c h w -> b c w h'),
        nn.Linear(h, h),
        nn.PReLU(),
        Rearrange('b c w h -> b w h c'),
        nn.Linear(c, c),
        nn.PReLU(),
        Rearrange('b w h c -> b c h w'),
        nn.Linear(h, h),
        nn.PReLU(),
    ) for _ in range(deep)])


class MLP_UHD(nn.Module):
    def __init__(self, size_pry=[256, 256, 32], deep=1):
        super().__init__()
        h, w, c = size_pry
        self.path_real = MLP_cubic(h, w, c, deep)

        self.path_imag = MLP_cubic(h, w, c, deep)

    def forward(self, x):
        a_f = torch.fft.fft2(x)
        a_f_real = a_f.real
        a_f_imag = a_f.imag

        # print(a_f.shape)
        coeff_a = torch.real(torch.fft.ifft2(torch.complex(self.path_real(a_f_real), self.path_imag(a_f_imag))))

        return coeff_a + x


class ResBlock(nn.Module):
    def __init__(self, ch):
        super(ResBlock, self).__init__()
        self.body = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            # nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input):
        res = self.body(input)
        output = res + input
        return output


class ResBlock1(nn.Module):
    def __init__(self, ch):
        super(ResBlock1, self).__init__()
        self.body = nn.Sequential(
            # nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(ch, ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input):
        res = self.body(input)
        output = res + input
        return output


class kernel_attention(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attention, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(kernel_size * kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        kernel = self.conv_kernel(kernel)
        # print(x.shape, kernel.shape)
        att = torch.cat([x, kernel], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output


class kernel_attentionZ(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attentionZ, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(kernel_size * kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        kernel = self.conv_kernel(kernel)
        # print(x.shape, kernel.shape)
        att = torch.cat([x, kernel], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output


class kernel_attention_deblur(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attention_deblur, self).__init__()
        self.kernel_size = kernel_size
        self.low_pass = dynamic_filter(inchannels=out_ch, group=8)
        # self.conv_1 = nn.Sequential(
        #     nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0),
        #     nn.GELU()
        # )
        self.convx1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        # self.conv_attn = nn.Sequential(
        #     nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0),
        # )

        self.conv_kernel = nn.Sequential(
            nn.Conv2d(kernel_size * kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        # self.temp = torch.nn.Parameter(torch.ones(1, out_ch, 1) * 3, requires_grad=True)

    def forward(self, input, kernel, kernel_feature):
        # x = self.conv_1(input) # .chunk(2, dim=1)

        # ker_attn = self.conv_attn(input)
        # ker_attn = rearrange(ker_attn, 'b c h w -> b c (h w)')
        # kernel_attn = rearrange(kernel, 'b c h w -> b (h w) c')
        # ker_attn = torch.softmax(ker_attn * self.temp, dim=-1)
        # kernel_key = ker_attn @ kernel_attn
        #
        # kernel_key = kernel_key.reshape(kernel_key.shape[0], kernel_key.shape[1], self.kernel_size, self.kernel_size)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        x1 = self.convx1(input)
        b_i = self.low_pass(x1)
        NSR = torch.std(input - b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        # x = featurefilter_deconv_fft(x, kernel, NSR=None)
        x_deblur = featurefilter_deconv_fft(x1, kernel, NSR=NSR)
        # x = x * torch.sigmoid(x_deblur)
        kernel_feature = self.conv_kernel(kernel_feature)
        # print(x.shape, kernel.shape)
        # x = self.conv_1(torch.cat([x1, x_deblur], dim=1))  # .chunk(2, dim=1)
        # x = self.conv_1(input)
        x = x1
        att = torch.cat([x, kernel_feature, x_deblur], dim=1)
        att = self.conv_2(att)  # + x_deblur
        x = x * att  # torch.sigmoid(att)
        output = x + input
        # output = self.conv_attn(torch.cat([x, x_deblur], dim=1)) + input
        return output

class deep_richardson_lucy_deblur(nn.Module):
    def __init__(self, kernel_size):
        super(deep_richardson_lucy_deblur, self).__init__()
        self.kernel_size = kernel_size


    def forward(self, estimated_image, blur, kernel, prior, map, mask=None):
        num_k = kernel.shape[1]
        ch = blur.shape[1]
        ks = max(kernel.shape[-1], kernel.shape[-2])
        ks = ks // 2
        dim = (ks, ks, ks, ks)
        if mask is not None:
            mask = mask.unsqueeze(2).expand(-1, -1, ch, -1, -1)
        # print(estimated_image.shape)
        b, nk = kernel.shape[:2]
        h, w = blur.shape[-2:]
        otf = convert_psf2otf(kernel, (b, nk, h + 2 * ks, w + 2 * ks))
        otf = otf.unsqueeze(2).expand(-1, -1, ch, -1, -1)
        otf = rearrange(otf, 'b k c h w -> (b k) c h w')
        otf_conj = torch.conj(otf)
        image = blur.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
        prior = prior.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
        map = map.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
        # if mask is not None:
        #     estimated_image = estimated_image * mask
        image = rearrange(image, 'b k c h w -> (b k) c h w')
        prior = rearrange(prior, 'b k c h w -> (b k) c h w')
        map = rearrange(map, 'b k c h w -> (b k) c h w')
        if estimated_image is None:
            estimated_image = torch.ones_like(image)
        else:
            estimated_image = estimated_image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)
            estimated_image = rearrange(estimated_image, 'b k c h w -> (b k) c h w')
        # for _ in range(iterations):
        blurred_image = conv_fft(estimated_image, otf, num_k, dim, ks, ch)
        error = image / torch.clamp(blurred_image - map + 1., 1e-9)  # (blurred_image + 1e-9)
        # print(error.shape)
        estimated_image = estimated_image * conv_fft(error, otf_conj, num_k, dim, ks, ch)
        estimated_image = estimated_image / torch.clamp(prior + 1., 1e-9) # torch.clamp(prior + 1., 1e-9)
        # estimated_image = estimated_image / (torch.abs(prior + 1.)+1e-9) # v8
            # print(estimated_image.shape)

            # print(estimated_image.shape)
        estimated_image = rearrange(estimated_image, '(b k) c h w -> b k c h w', k=num_k, c=ch)
        if mask is not None:
            estimated_image = estimated_image * mask
        # print(estimated_image.shape)
        return torch.sum(estimated_image, dim=1)

class BatchPixConv(nn.Module):
    def __init__(self, l=19, padmode='reflection'):
        super(BatchPixConv, self).__init__()
        self.l = l
        if padmode == 'reflection':
            if l % 2 == 1:
                self.pad = nn.ReflectionPad2d(l // 2)
            else:
                self.pad = nn.ReflectionPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'zero':
            if l % 2 == 1:
                self.pad = nn.ZeroPad2d(l // 2)
            else:
                self.pad = nn.ZeroPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))
        elif padmode == 'replication':
            if l % 2 == 1:
                self.pad = nn.ReplicationPad2d(l // 2)
            else:
                self.pad = nn.ReplicationPad2d((l // 2, l // 2 - 1, l // 2, l // 2 - 1))


    def forward(self, input, kernel):
        # kernel of size [N,Himage*Wimage,H,W]
        B, C, H, W = input.size()
        pad = self.pad(input)
        H_p, W_p = pad.size()[-2:]
        # print(input.shape, kernel.shape)
        if len(kernel.size()) == 2:
            input_CBHW = pad.view((C * B, 1, H_p, W_p))
            kernel_var = kernel.contiguous().view((1, 1, self.l, self.l))
            return F.conv2d(input_CBHW, kernel_var, padding=0).view((B, C, H, W))
        else:
            pad = pad.view(C * B, 1, H_p, W_p)
            pad = F.unfold(pad, self.l).transpose(1, 2)   # [CB, HW, k^2]
            kernel = kernel.flatten(2).unsqueeze(0).expand(3, -1, -1, -1)

            kernel = kernel.contiguous().view(-1, kernel.size(2), kernel.size(3))
            # print(self.l, pad.shape, kernel.shape)
            # kernel = kernel.transpose(1, 2)
            out_unf = (pad * kernel).sum(2).unsqueeze(1)
            out = F.fold(out_unf, (H, W), 1).view(B, C, H, W)

            return out
class deep_richardson_lucy_pix_deblur(nn.Module):
    def __init__(self, kernel_size):
        super(deep_richardson_lucy_pix_deblur, self).__init__()
        self.kernel_size = kernel_size
        self.pix_conv = BatchPixConv(kernel_size, 'replication')

    def forward(self, estimated_image, blur, kernel, prior, map):
        # kernel : B H W k1 k2

        kernel_conj = torch.flip(kernel, [-2, -1])
        image = blur

        if estimated_image is None:
            estimated_image = torch.ones_like(image)

        # for _ in range(iterations):
        # print(estimated_image.shape, kernel.shape)
        blurred_image = self.pix_conv(estimated_image, kernel)

        error = image / torch.clamp(blurred_image - map + 1., 1e-9)  # (blurred_image + 1e-9)
        # print(error.shape)
        estimated_image = estimated_image * self.pix_conv(error, kernel_conj)
        estimated_image = estimated_image / torch.clamp(prior + 1., 1e-9) # torch.clamp(prior + 1., 1e-9)

        return estimated_image, blurred_image

class deep_dynamic_richardson_lucy_pix_deblur(nn.Module):
    def __init__(self, width, kernel_size):
        super(deep_dynamic_richardson_lucy_pix_deblur, self).__init__()
        self.kernel_size = kernel_size
        self.weight_estimator1 = nn.Sequential(
            nn.Conv2d(width, kernel_size ** 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Softmax(1)
        )

        self.offset_estimator1 = nn.Sequential(
            nn.Conv2d(width, 2 * kernel_size ** 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.reblur_kernel_estimator=nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(128, kernel_size ** 2, 1, 1, 0, bias=True),
        )
        self.weight_estimator2 = nn.Sequential(
            nn.Conv2d(width, kernel_size ** 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Softmax(1)
        )
        self.offset_estimator2 = nn.Sequential(
            nn.Conv2d(width, 2 * kernel_size ** 2, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.deblur_kernel_estimator = nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(128, kernel_size ** 2, 1, 1, 0, bias=True),
        )
        self.pen = nn.Sequential(
            LayerNorm2d(width ),
            nn.Conv2d(width, 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1, bias=True),
            nn.Tanh()
        )
        self.men = nn.Sequential(
            LayerNorm2d(width),
            nn.Conv2d(width , 16, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(16, 3, 3, 1, 1, bias=True),
            nn.Sigmoid()
        )
        self.pix_conv = MiSCFilter_mxt2(3, kernel_size=kernel_size, pad=(kernel_size - 1) // 2)

    def forward(self, estimated_image, blur, dg, gt=None, only_reblur=False):
        kernel = self.reblur_kernel_estimator(dg)
        offset1 = self.offset_estimator1(dg)
        weight1 = self.weight_estimator1(dg)
        if only_reblur:
            return self.pix_conv(estimated_image, kernel, offset1, weight1)
        else:
            image = blur
            prior = self.pen(dg)
            map = self.men(dg)



            kernel_conj = self.deblur_kernel_estimator(dg)
            offset2 = self.offset_estimator2(dg)
            weight2 = self.weight_estimator2(dg)
            # b, c, h, w = kernel.shape
            # kernel_conj = torch.flip(kernel.view(b, self.kernel_size, self.kernel_size, h, w), [1, 2]).view(b,
            #                                                                                                 self.kernel_size * self.kernel_size,
            #                                                                                                 h, w)
            # offset2 = offset1
            # weight2 = weight1
            blurred_image = self.pix_conv(estimated_image, kernel, offset1, weight1)

            error = image / torch.clamp(blurred_image - map + 1., 1e-9)  # (blurred_image + 1e-9)
            # print(error.shape)
            estimated_image = estimated_image * self.pix_conv(error, kernel_conj, offset2, weight2)
            estimated_image = estimated_image / torch.clamp(prior + 1., 1e-9) # torch.clamp(prior + 1., 1e-9)
            if gt is not None:
                blurred_image = self.pix_conv(gt, kernel, offset1, weight1)
            return estimated_image, blurred_image

class deep_dynamic_pix_reblur(nn.Module):
    def __init__(self, width, kernel_size):
        super(deep_dynamic_pix_reblur, self).__init__()
        self.kernel_size = kernel_size
        self.weight_estimator1 = nn.Sequential(
            nn.Conv2d(width, kernel_size ** 2, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Softmax(1)
        )

        self.offset_estimator1 = nn.Sequential(
            nn.Conv2d(width, 2 * kernel_size ** 2, kernel_size=1, stride=1, padding=0, bias=True)
        )

        self.reblur_kernel_estimator=nn.Sequential(
            nn.Conv2d(width, 128, kernel_size=1, stride=1, padding=0, bias=True),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(128, kernel_size ** 2, 1, 1, 0, bias=True),
        )

        self.pix_conv = MiSCFilter_mxt2(3, kernel_size=kernel_size, pad=(kernel_size - 1) // 2)

    def forward(self, estimated_image, dg):
        kernel = self.reblur_kernel_estimator(dg)
        offset1 = self.offset_estimator1(dg)
        weight1 = self.weight_estimator1(dg)
        return self.pix_conv(estimated_image, kernel, offset1, weight1)

class kernel_attention_xdeblur(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attention_xdeblur, self).__init__()
        self.kernel_size = kernel_size

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.conv_kernel = nn.Sequential(
            nn.Conv2d(kernel_size * kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        # self.temp = torch.nn.Parameter(torch.ones(1, out_ch, 1) * 3, requires_grad=True)

    def forward(self, input, x_deconv, kernel_feature):
        x = self.conv_1(torch.cat([input, x_deconv], dim=1))
        # x = x * torch.sigmoid(x_deblur)
        kernel_feature = self.conv_kernel(kernel_feature)
        # print(x.shape, kernel.shape)
        att = torch.cat([x, kernel_feature], dim=1)
        att = self.conv_2(att)  # + x_deblur
        x = x * att  # torch.sigmoid(att)
        output = x + input
        return output


class kernel_attention_reblur(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attention_reblur, self).__init__()
        self.kernel_size = kernel_size

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        # self.conv_attn = nn.Sequential(
        #     nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0),
        # )

        self.conv_kernel = nn.Sequential(
            nn.Conv2d(kernel_size * kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        # self.temp = torch.nn.Parameter(torch.ones(1, out_ch, 1) * 3, requires_grad=True)

    def forward(self, input, kernel, kernel_feature):
        x = self.conv_1(input)

        # x = featurefilter_deconv_fft(x, kernel, NSR=None)
        x_reblur = featurefilter_fft(x, kernel)
        # x = x * torch.sigmoid(x_deblur)
        kernel_feature = self.conv_kernel(kernel_feature)
        # print(x.shape, kernel.shape)
        att = torch.cat([x, kernel_feature, x_reblur], dim=1)
        att = self.conv_2(att)  # + x_deblur
        x = x * att  # torch.sigmoid(att)
        output = x + input
        # output = self.conv_attn(torch.cat([output], dim=1))
        return output


class kernel_attention_phasedeblur(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attention_phasedeblur, self).__init__()
        self.kernel_size = kernel_size

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        # self.conv_attn = nn.Sequential(
        #     nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0),
        # )

        self.conv_kernel = nn.Sequential(
            nn.Conv2d(kernel_size * kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        # self.temp = torch.nn.Parameter(torch.ones(1, out_ch, 1) * 3, requires_grad=True)

    def forward(self, input, kernel, kernel_feature):
        x = self.conv_1(input)

        # x = featurefilter_deconv_fft(x, kernel, NSR=None)
        x_deblur = featurefilter_deconv_fft(x, kernel)
        # x = x * torch.sigmoid(x_deblur)
        kernel_feature = self.conv_kernel(kernel_feature)
        # print(x.shape, kernel.shape)
        att = torch.cat([x, kernel_feature, x_deblur], dim=1)
        att = self.conv_2(att)  # + x_deblur
        x = x * att  # torch.sigmoid(att)
        output = x + input
        # output = self.conv_attn(torch.cat([output], dim=1))
        return output


class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)
        self.bn = nn.BatchNorm2d(group * kernel_size ** 2)
        self.act = nn.Softmax(dim=-2)
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        # self.lamb_l = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        # self.lamb_h = nn.Parameter(torch.zeros(inchannels), requires_grad=True)
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        self.ap = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        # identity_input = x  # 3,32,64,64
        low_filter = self.ap(x)
        low_filter = self.conv(low_filter)
        low_filter = self.bn(low_filter)

        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(n, self.group, c // self.group,
                                                                        self.kernel_size ** 2, h * w)

        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q).unsqueeze(2)

        low_filter = self.act(low_filter)

        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)
        return low_part


class kernel_deblur(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=19, pad_method='replicate'):
        super(kernel_deblur, self).__init__()
        self.pad_method = pad_method

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        group = 8 if in_ch % 8 == 0 else 1
        self.low_pass = dynamic_filter(inchannels=in_ch, group=group)

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=out_ch, out_channels=kernel_size ** 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
            nn.Sigmoid()
        )
        # self.low_pass = kornia.filters.MedianBlur([3, 3])
        # self.gen_prompt = PromptGenBlock(prompt_dim=in_ch, prompt_len=1, prompt_size=train_size[0],
        #                                  lin_dim=out_ch)

    def forward(self, input, kernel):
        x = self.conv_1(input)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = self.gen_prompt(x)
        # NSR = torch.mean(torch.abs(NSR), dim=[-2, -1], keepdim=True)
        b_i = self.low_pass(x)
        NSR = torch.std(input - b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        # x = featurefilter_deconv_fft(x, kernel, NSR=None)

        s_k = self.sca(x).view(x.shape[0], 1, kernel.shape[2], kernel.shape[3])
        kernel = kernel * s_k
        x = featurefilter_deconv_fft(x, kernel, NSR=NSR, pad_method=self.pad_method)
        # if self.pad_method is None:
        #     ks1 = kernel.shape[-2] // 2
        #     ks2 = kernel.shape[-1] // 2
        #     input = input[:, :, ks1:-ks1, ks1:-ks1].contiguous()
        att = torch.cat([x, input], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output


class kernel_deblurZ(nn.Module):
    def __init__(self, in_ch, out_ch, train_size=[64, 64], pad_method='replicate'):
        super(kernel_deblurZ, self).__init__()
        self.pad_method = pad_method

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        group = 8 if in_ch % 8 == 0 else 1
        self.low_pass = dynamic_filter(inchannels=in_ch, group=group)
        # self.gen_prompt = PromptGenBlock(prompt_dim=in_ch, prompt_len=1, prompt_size=train_size[0],
        #                                  lin_dim=out_ch)

    def forward(self, input, kernel):
        x = self.conv_1(input)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = self.gen_prompt(x)
        # NSR = torch.mean(torch.abs(NSR), dim=[-2, -1], keepdim=True)
        b_i = self.low_pass(x)
        NSR = torch.std(x - b_i, dim=[-2, -1], keepdim=True) / torch.var(x, dim=[-2, -1], keepdim=True)
        # x = featurefilter_deconv_fft(x, kernel, NSR=None)
        x = featurefilter_deconv_fft(x, kernel, NSR=NSR, pad_method=self.pad_method)
        # if self.pad_method is None:
        #     ks1 = kernel.shape[-2] // 2
        #     ks2 = kernel.shape[-1] // 2
        #     input = input[:, :, ks1:-ks1, ks1:-ks1].contiguous()
        att = torch.cat([x, input], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input
        return output


class simple_kernel_deblur(nn.Module):
    def __init__(self, in_ch, out_ch, train_size=[64, 64], num_heads=1, pad_method='replicate'):
        super(simple_kernel_deblur, self).__init__()
        self.pad_method = pad_method

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, num_heads, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(num_heads, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        group = 8 if in_ch % 8 == 0 else 1
        self.low_pass = dynamic_filter(inchannels=in_ch, group=1)
        # self.gen_prompt = PromptGenBlock(prompt_dim=in_ch, prompt_len=1, prompt_size=train_size[0],
        #                                  lin_dim=out_ch)

    def forward(self, input, kernel):
        x = self.conv_1(input)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = self.gen_prompt(x)
        # NSR = torch.mean(torch.abs(NSR), dim=[-2, -1], keepdim=True)
        b_i = self.low_pass(x)
        NSR = torch.std(x - b_i, dim=[-2, -1], keepdim=True) / torch.var(x, dim=[-2, -1], keepdim=True)
        # x = featurefilter_deconv_fft(x, kernel, NSR=None)
        x_ = featurefilter_deconv_fft(x, kernel, NSR=NSR, pad_method=self.pad_method)
        # if self.pad_method is None:
        #     ks1 = kernel.shape[-2] // 2
        #     ks2 = kernel.shape[-1] // 2
        #     input = input[:, :, ks1:-ks1, ks1:-ks1].contiguous()
        # att = torch.cat([x, input], dim=1)
        att = self.conv_2(x_)
        x = x * att
        output = x + input

        return output


class kernel_deblurV2(nn.Module):
    def __init__(self, in_ch, out_ch, train_size=[64, 64], pad_method='replicate'):
        super(kernel_deblurV2, self).__init__()
        self.pad_method = pad_method

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        group = 8 if in_ch % 8 == 0 else 1
        self.low_pass = dynamic_filter(inchannels=in_ch, group=group)
        # self.gen_prompt = PromptGenBlock(prompt_dim=in_ch, prompt_len=1, prompt_size=train_size[0],
        #                                  lin_dim=out_ch)

    def forward(self, input, kernel):
        x = self.conv_1(input)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = self.gen_prompt(x)
        # NSR = torch.mean(torch.abs(NSR), dim=[-2, -1], keepdim=True)
        b_i = self.low_pass(x)
        NSR = torch.std(x - b_i, dim=[-2, -1], keepdim=True) / torch.var(x, dim=[-2, -1], keepdim=True)
        # x = featurefilter_deconv_fft(x, kernel, NSR=None)
        x = featurefilter_deconv_fft(x, kernel, NSR=NSR, pad_method=self.pad_method)
        # if self.pad_method is None:
        #     ks1 = kernel.shape[-2] // 2
        #     ks2 = kernel.shape[-1] // 2
        #     input = input[:, :, ks1:-ks1, ks1:-ks1].contiguous()
        att = torch.cat([x, input], dim=1)
        output = self.conv_2(att)
        return output


class kernel_deblur_fourierconv(nn.Module):
    def __init__(self, in_ch, out_ch, train_size=[64, 64]):
        super(kernel_deblur_fourierconv, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_fourier = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch * 2, kernel_size=1, stride=1, padding=0),
            nn.GELU(),
            nn.Conv2d(out_ch * 2, out_ch * 2, kernel_size=1, stride=1, padding=0),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.low_pass = dynamic_filter(inchannels=out_ch, group=8)
        # self.gen_prompt = PromptGenBlock(prompt_dim=in_ch, prompt_len=1, prompt_size=train_size[0],
        #                                  lin_dim=out_ch)

    def forward(self, input, kernel):
        x = self.conv_1(input)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = self.gen_prompt(x)
        # NSR = torch.mean(torch.abs(NSR), dim=[-2, -1], keepdim=True)
        b_i = self.low_pass(x)
        NSR = torch.std(x - b_i, dim=[-2, -1], keepdim=True) / torch.var(x, dim=[-2, -1], keepdim=True)
        # x = featurefilter_deconv_fft(x, kernel, NSR=None)
        # x = featurefilter_deconv_fft(x, kernel, NSR=NSR)
        ks = max(kernel.shape[-1], kernel.shape[-2])
        dim = (ks, ks, ks, ks)
        x = torch.nn.functional.pad(x, dim, "replicate")
        # image = image.unsqueeze(1).expand(-1, num_k, -1, -1, -1)

        # image = rearrange(image, 'b (g c) h w -> b g c h w', g=groups)

        otf = convert_psf2otf(kernel, x.size())
        # otf = torch.conj(otf) / (torch.abs(otf) + 1e-7)
        if NSR is None:
            otf = torch.conj(otf) / (torch.abs(otf) ** 2 + 1e-5)
        else:
            otf = torch.conj(otf) / (torch.abs(otf) ** 2 + NSR)
        # otf = torch.conj(otf) / (torch.abs(otf) + 1e-5)
        # otf_real = torch.clamp(otf.real, -100., 100.)
        # otf_imag = torch.clamp(otf.imag, -100., 100.)
        # otf = torch.complex(otf_real, otf_imag)
        # otf = otf.unsqueeze(1).expand(-1, groups, -1, -1, -1)
        # otf = rearrange(otf, 'b k c h w -> b (k c) h w')
        x = torch.fft.rfft2(x) * otf
        x = torch.cat([x.real, x.imag], dim=1)
        x = self.conv_fourier(x)
        x_r, x_i = x.chunk(2, dim=1)
        x = torch.complex(x_r, x_i)
        x = torch.fft.irfft2(x)[:, :, ks:-ks, ks:-ks].contiguous()
        att = torch.cat([x, input], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output


class kernel_deblur_feature(nn.Module):
    def __init__(self, in_ch, out_ch, train_size=[64, 64]):
        super(kernel_deblur_feature, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_attn = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0),
            # nn.Softmax(1)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.low_pass = dynamic_filter(inchannels=out_ch, group=8)
        self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch * 2, 1, 1), requires_grad=True)
        # self.gen_prompt = PromptGenBlock(prompt_dim=in_ch, prompt_len=1, prompt_size=train_size[0],
        #                                  lin_dim=out_ch)

    def forward(self, input, kernel, kernel_feature):
        x = self.conv_1(input)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = self.gen_prompt(x)
        # NSR = torch.mean(torch.abs(NSR), dim=[-2, -1], keepdim=True)
        b_i = self.low_pass(x)
        NSR = torch.std(input - b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        # x = featurefilter_deconv_fft(x, kernel, NSR=None)
        x_deblur = featurefilter_deconv_fft(x, kernel, NSR=NSR)
        # att = torch.cat([x, input], dim=1)
        att = torch.cat([x, kernel_feature], dim=1)
        att = self.conv_2(att * self.gamma)
        x = x * att
        output = x + input
        output = self.conv_attn(torch.cat([output, x_deblur], dim=1))

        return output


class kernel_attn_deblur(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attn_deblur, self).__init__()
        self.kernel_size = kernel_size
        self.low_pass = dynamic_filter(inchannels=out_ch, group=8)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        # self.conv_kernel = nn.Sequential(
        #     nn.Conv2d(kernel_size * kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.GELU()
        # )
        self.conv_attn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
            # nn.Softmax(1)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch * 3, 1, 1), requires_grad=True)

    def forward(self, input, kernel, kernel_feature):
        x = self.conv_1(input)
        # ker_attn = 1
        # ker_attn = self.conv_attn(x)
        ker_attn = self.conv_attn(torch.nn.functional.interpolate(x, size=[kernel.shape[-2], kernel.shape[-1]]))
        ker_attn = rearrange(ker_attn, 'b c h w -> b c (h w)')
        kernel_attn = rearrange(kernel, 'b c h w -> b (h w) c')
        ker_attn = torch.softmax(ker_attn, dim=-1)
        kernel_key = ker_attn @ kernel_attn

        kernel_key = kernel_key.reshape(kernel_key.shape[0], kernel_key.shape[1], self.kernel_size, self.kernel_size)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        b_i = self.low_pass(x)
        NSR = torch.std(input - b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        # x = featurefilter_deconv_fft(x, kernel, NSR=None)
        x = featurefilter_deconv_fft(x, kernel_key, NSR=NSR)
        # x = featurefilter_fft(x, kernel_key, NSR=None)
        # k_y = self.conv_kernel(kernel)
        # att = torch.cat([x, input, k_y], dim=1)
        att = torch.cat([x, input, kernel_feature], dim=1)
        att = self.conv_2(att * self.gamma)
        x = x * att
        output = x + input

        return output


class kernel_reblur(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_reblur, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        # self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch*2, 1, 1), requires_grad=True)

    def forward(self, input, kernel):
        x = self.conv_1(input)
        # ker_attn = 1
        # ker_attn = rearrange(ker_attn, 'b c h w -> b c (h w)')
        # ker_attn = torch.softmax(ker_attn, dim=-1)
        # kernel_code_uncertain = ker_attn @ kernel_code_uncertain.reshape(kernel_code.shape[0], kernel_code_uncertain.shape[1], -1)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        x = featurefilter_fft(x, kernel, NSR=None)
        att = torch.cat([x, input], dim=1)
        att = self.conv_2(att)  # *self.gamma
        x = x * att
        output = x + input

        return output


class kernel_logdeblur(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_logdeblur, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )

        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        # self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch*2, 1, 1), requires_grad=True)

    def forward(self, input, kernel):
        x = self.conv_1(input)
        # ker_attn = 1
        # ker_attn = rearrange(ker_attn, 'b c h w -> b c (h w)')
        # ker_attn = torch.softmax(ker_attn, dim=-1)
        # kernel_code_uncertain = ker_attn @ kernel_code_uncertain.reshape(kernel_code.shape[0], kernel_code_uncertain.shape[1], -1)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        x = logarithmic_fft(x, kernel, NSR=None)
        att = torch.cat([x, input], dim=1)
        att = self.conv_2(att)  # *self.gamma
        x = x * att
        output = x + input

        return output


class simple_kernel_reblur(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(simple_kernel_reblur, self).__init__()

        self.conv_out = nn.Sequential(
            nn.Conv2d(in_ch * 2, out_ch, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, input, kernel):
        x = featurefilter_fft(input, kernel, NSR=None)
        output = torch.cat([x, input], dim=1)
        output = self.conv_out(output)
        return output


class kernel_attn_reblur(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attn_reblur, self).__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        # self.conv_blur = nn.Sequential(
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        #     # nn.GELU(),
        #     # nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        #     # nn.GELU()
        # )
        self.kernel_up = nn.AdaptiveAvgPool2d(kernel_size)
        self.conv_attn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch * 2, 1, 1), requires_grad=True)

    def forward(self, input, kernel):
        x = self.conv_1(input)
        # ker_attn = 1
        ker_attn = self.conv_attn(torch.nn.functional.interpolate(x, size=[kernel.shape[-2], kernel.shape[-1]]))
        ker_attn = rearrange(ker_attn, 'b c h w -> b c (h w)')
        kernel_attn = rearrange(kernel, 'b c h w -> b (h w) c')
        ker_attn = torch.softmax(ker_attn, dim=-1)
        kernel_key = ker_attn @ kernel_attn
        in_ker_size = int(np.sqrt(kernel_key.shape[-1]))
        if in_ker_size != self.kernel_size:
            kernel_key = kernel_key.reshape(kernel_key.shape[0], kernel_key.shape[1], in_ker_size,
                                            in_ker_size)
            kernel_key = self.kernel_up(kernel_key)
        else:
            kernel_key = kernel_key.reshape(kernel_key.shape[0], kernel_key.shape[1], self.kernel_size,
                                            self.kernel_size)
        # import seaborn as sns
        # import matplotlib.pyplot as plt
        # result_dir = '/home/ubuntu/90t/personal_data/mxt/MXT/RevIR/Motion_Deblurring/results_kernel_feature_reblur'
        # kernelx = kernel_key.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
        # for k in range(kernelx.shape[-1]):
        #     out_way = os.path.join(result_dir, 'x_kernel_' + str(k) + '.png')
        #     sns_plot = plt.figure()
        #     # print(kernelx[:, :, k].max())
        #     ker = kernelx[:, :, k]
        #     sns.heatmap(ker, cmap='Reds', linewidths=0., vmin=ker.min(), vmax=ker.max(),
        #                 xticklabels=False, yticklabels=False, cbar=False, square=True)  # Reds_r .invert_yaxis()
        #     sns_plot.savefig(out_way, dpi=80, pad_inches=0, bbox_inches='tight')
        #     plt.close()
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        x = featurefilter_fft(x, kernel_key, NSR=None)
        # k_y = self.conv_kernel(kernel)
        att = torch.cat([x, input], dim=1)
        att = self.conv_2(att * self.gamma)
        x = x * att  # + self.conv_blur(x)
        output = x + input

        return output


class kernel_attn_reblur_feature(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_attn_reblur_feature, self).__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        # self.conv_kernel = nn.Sequential(
        #     nn.Conv2d(kernel_size * kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.GELU()
        # )
        self.conv_attn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=1, stride=1, padding=0),
            # nn.Softmax(1)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 3, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch * 3, 1, 1), requires_grad=True)

    def forward(self, input, kernel, kernel_feature):
        x = self.conv_1(input)
        # ker_attn = 1
        ker_attn = self.conv_attn(x)
        ker_attn = rearrange(ker_attn, 'b c h w -> b c (h w)')
        kernel_attn = rearrange(kernel, 'b c h w -> b (h w) c')
        ker_attn = torch.softmax(ker_attn, dim=-1)
        kernel_key = ker_attn @ kernel_attn

        kernel_key = kernel_key.view(kernel_key.shape[0], kernel_key.shape[1], self.kernel_size, self.kernel_size)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        x = featurefilter_fft(x, kernel_key, NSR=None)
        # k_y = self.conv_kernel(kernel)
        att = torch.cat([x, input, kernel_feature], dim=1)
        att = self.conv_2(att * self.gamma)
        x = x * att
        output = x + input

        return output


class kernel_reblur_feature(nn.Module):
    def __init__(self, kernel_size, in_ch, out_ch):
        super(kernel_reblur_feature, self).__init__()
        self.kernel_size = kernel_size
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        # self.conv_kernel = nn.Sequential(
        #     nn.Conv2d(kernel_size * kernel_size, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.GELU(),
        #     nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
        #     nn.GELU()
        # )
        self.conv_attn = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=1, stride=1, padding=0),
            # nn.Softmax(1)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )
        self.gamma = torch.nn.Parameter(torch.zeros(1, out_ch * 2, 1, 1), requires_grad=True)

    def forward(self, input, kernel, kernel_feature):
        x = self.conv_1(input)
        # ker_attn = 1
        # ker_attn = self.conv_attn(x)
        # ker_attn = rearrange(ker_attn, 'b c h w -> b c (h w)')
        # kernel_attn = rearrange(kernel, 'b c h w -> b (h w) c')
        # ker_attn = torch.softmax(ker_attn, dim=-1)
        # kernel_key = ker_attn @ kernel_attn
        #
        # kernel_key = kernel_key.view(kernel_key.shape[0], kernel_key.shape[1], self.kernel_size, self.kernel_size)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        x_reblur = featurefilter_fft(x, kernel, NSR=None)
        # k_y = self.conv_kernel(kernel)
        att = torch.cat([x, kernel_feature], dim=1)
        att = self.conv_2(att * self.gamma)
        x = x * att
        output = x + input
        output = self.conv_attn(torch.cat([output, x_reblur], dim=1))
        return output


class Spatial_Conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Spatial_Conv, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.dwconv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, groups=out_ch, stride=1, padding=1)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input):
        x = self.conv_1(input)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        # x = deblurfeaturefilter_fft(x, kernel, NSR=None)
        x = self.dwconv(x)
        att = torch.cat([x, input], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output


class kernel_deblur2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(kernel_deblur2, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_deconv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        # b_i = kornia.filters.median_blur(x, [3, 3])
        # NSR = torch.std(input-b_i, dim=[-2, -1], keepdim=True) / torch.var(input, dim=[-2, -1], keepdim=True)
        x_deconv = deblurfeaturefilter_fft(x, kernel, NSR=None)
        x_deconv = self.conv_deconv(x_deconv)
        att = torch.cat([x, x_deconv], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output


class feature_attention(nn.Module):
    def __init__(self, feat_ch, in_ch, out_ch):
        super(feature_attention, self).__init__()

        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_kernel = nn.Sequential(
            nn.Conv2d(feat_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1),
            nn.GELU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(out_ch * 2, out_ch, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

    def forward(self, input, kernel):
        x = self.conv_1(input)
        kernel = self.conv_kernel(kernel)
        att = torch.cat([x, kernel], dim=1)
        att = self.conv_2(att)
        x = x * att
        output = x + input

        return output


class NAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_attion = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp

        # kernel [B, 19*19, H, W]
        # x = self.kernel_attion(x, kernel)
        x = self.norm1(x)
        x = self.kernel_attion(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_kernel_deblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_attn = kernel_attention_deblur(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel, kf = xk
        x = inp

        # kernel [B, 19*19, H, W]

        x = self.norm1(x)
        x = self.kernel_attn(x, kernel, kf)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_kernel_deblurX(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sca2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c * 2, out_channels=c * 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # self.norm1 = LayerNorm2d(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_attn = kernel_attention(kernel_size, in_ch=c, out_ch=c)
        self.deblur_attn = kernel_deblur(c, c)

    def forward(self, xk):
        inp, kernel, kf = xk
        x = inp

        # kernel [B, 19*19, H, W]

        x = self.norm1(x)
        b, c, h, w = x.shape
        x1 = self.kernel_attn(x, kf)
        x2 = self.deblur_attn(x, kernel)

        xca = self.sca2(torch.cat([x1, x2], dim=1)).view(b, c, 2, 1)
        xca = torch.softmax(xca, dim=2)
        # print(xca.shape)
        xca1, xca2 = xca.chunk(2, dim=2)
        # print(xca1.shape)
        x = x1 * xca1 + x2 * xca2
        x = self.conv1(x)
        # x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_kernel_deblurXY(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.convz = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.convz0 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                groups=1, bias=True)
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.scaz = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=c * 2, out_channels=c * 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm0 = LayerNorm2d(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.dropout0 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_attn = kernel_attentionZ(kernel_size, in_ch=c, out_ch=c)
        self.deblur_attn = kernel_deblurZ(c, c)

    def forward(self, xk):
        inp, kernel, kf = xk
        x = inp

        # kernel [B, 19*19, H, W]

        x = self.norm0(x)
        x = self.convz0(x)
        b, c, h, w = x.shape
        x1 = self.kernel_attn(x, kf)
        x2 = self.deblur_attn(x, kernel)

        xca = self.scaz(torch.cat([x1, x2], dim=1)).view(b, c, 2, 1)
        xca = torch.softmax(xca, dim=2)
        # print(xca.shape)
        xca1, xca2 = xca.chunk(2, dim=2)
        # print(xca1.shape)
        x = x1 * xca1 + x2 * xca2
        x = self.convz(x)
        x = self.dropout0(x)
        z = inp + x * self.alpha

        x = self.norm1(z)
        x = self.conv1(x)
        # x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = z + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_kernel_xdeblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_attn = kernel_attention_xdeblur(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, xf, kf = xk
        x = inp

        # kernel [B, 19*19, H, W]
        x = self.kernel_attn(x, xf, kf)

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_kernel_xattn_deblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, num_heads=1, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm00 = LayerNorm2d(c)
        self.norm01 = LayerNorm2d(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.spatial_attn = MyAttention(c, num_heads, bias=True, cs='global_channel')
        self.kernel_attn = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, xf, kf = xk
        x = inp

        # kernel [B, 19*19, H, W]
        x = self.kernel_attn(x, kf)

        x_ = torch.cat([self.norm00(x), self.norm01(xf)], dim=1)
        x_ = self.spatial_attn(x_) * self.alpha + inp

        x = self.norm1(x_)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = x_ + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_kernel_kfattn_deblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, num_heads=1, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # self.norm00 = LayerNorm2d(c)
        # self.norm01 = LayerNorm2d(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.spatial_attn = ICCVAttention(c, num_heads, bias=True, cs='global_channel')
        self.kernel_attn = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kf = xk
        x = inp

        x_ = self.spatial_attn(x) * self.alpha + self.kernel_attn(x, kf)

        x = self.norm1(x_)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = x_ + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_kernel_xkattn_deblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, num_heads=1, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # self.norm00 = LayerNorm2d(c)
        # self.norm01 = LayerNorm2d(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.spatial_attn = RecoordAttention(c, num_heads, kernel_size, pad=kernel_size // 2, div_ker=True, deconv=True,
                                             bias=True, cs='global_channel')
        self.kernel_attn = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, ker, kf = xk
        x = inp

        x_ = self.spatial_attn(x, ker) * self.alpha + self.kernel_attn(x, kf)

        x = self.norm1(x_)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = x_ + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_kernel_phasedeblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_attn = kernel_attention_phasedeblur(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel, kf = xk
        x = inp

        # kernel [B, 19*19, H, W]
        x = self.kernel_attn(x, kernel, kf)

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_kernel_reblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_attn = kernel_attention_reblur(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel, kf = xk
        x = inp

        # kernel [B, 19*19, H, W]
        x = self.kernel_attn(x, kernel, kf)

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_kernel_ffndeblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_attn1 = kernel_attention(kernel_size, in_ch=c, out_ch=c)

        self.kernel_attn2 = kernel_deblur(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel, kf = xk
        x = inp

        # kernel [B, 19*19, H, W]
        x = self.kernel_attn1(x, kf)

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_kernel_feature(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., feature_ch=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_atttion = feature_attention(feature_ch, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp

        # kernel [B, 19*19, H, W]
        x = self.kernel_atttion(x, kernel)

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class NAFBlock_deblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_deblur(in_ch=c, out_ch=c, kernel_size=kernel_size)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk
        x = inp

        x = self.norm1(x)
        x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_deblur(y, kernel)
        return y


class NAFBlock_mamba_deblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_deblur(in_ch=c, out_ch=c, train_size=train_size)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)
        x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_deblur(y, kernel)
        return y


class NAFBlock_deformdeblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        group = 1
        self.kernel_deform_deblur = DCNv3_ker_pytorch(1, kernel_size=7, motion_blur_kernel_size=kernel_size,
                                                      dw_kernel_size=3, pad=3, group=group)
        self.kernel_deblur = kernel_deblur(c, c)
        self.proj_ker = nn.Conv2d(c, 1, 1)
        self.proj_x = nn.Conv2d(c, 1, 1)
        # self.temp = nn.Parameter(torch.ones([1, group, 1, 1]) * 5., requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)

        x_deform, ker = self.kernel_deform_deblur(self.proj_x(x), self.proj_ker(kernel))
        kernel = torch.sigmoid(ker) * kernel
        x = self.kernel_deblur(x, kernel) * torch.sigmoid(x_deform)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_deblur(y, kernel)
        return y


class NAFBlock_deformdeblurV1(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        group = 4
        self.kernel_deblur = DCNv3_ker_pytorch(c, kernel_size=7, motion_blur_kernel_size=kernel_size, dw_kernel_size=3,
                                               pad=3, group=group)
        self.kernel_deblur2 = kernel_deblurV2(group, c)
        self.projx = nn.Conv2d(dw_channel, group, 1)
        # self.temp = nn.Parameter(torch.ones([1, group, 1, 1]) * 5., requires_grad=True)
        # h, w = kernel_size, kernel_size
        # x_coords = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, steps=w)  # .view(1, w).expand(h, w)
        # y_coords = torch.linspace(-kernel_size // 2 + 1, kernel_size // 2, steps=h)  # .view(h, 1).expand(h, w)
        # # print(x_coords, y_coords)
        # grid_x, grid_y = torch.meshgrid(x_coords, y_coords)
        # self.offseth = nn.Parameter(grid_x, requires_grad=False)
        # self.offsetw = nn.Parameter(grid_y, requires_grad=False)
        #
        # deform_kersize = 7
        # self.deform_kersize = deform_kersize ** 2
        # print(grid_x, grid_y)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)

        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x1, x2 = x.chunk(2, dim=1)
        x1, kernel = self.kernel_deblur(x1, kernel)
        # kernel = kornia.geometry.subpix.spatial_softmax2d(self.temp * kernel)
        x_ = self.projx(x)
        # print(x1.shape)
        # print(x2.shape)
        # print(x_.shape)
        x = x1 * x2 * self.kernel_deblur2(x_, kernel) * self.alpha
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_deblur(y, kernel)
        return y


class FCNAFBlock_deblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True,
                                                 act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_deblur(in_ch=c, out_ch=c, train_size=train_size)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x_norm = self.norm1(x)
        x = self.kernel_deblur(x_norm, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x) + self.fft_block1(x_norm)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x) + self.fft_block2(y)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_deblur(y, kernel)
        return y


class NAFBlock_patch_deblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.kernel_size = kernel_size
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_deblur(in_ch=c, out_ch=c, train_size=train_size)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)
        batch_k = kernel.shape[0]
        batch_x = x.shape[0]
        num_patch = batch_k // batch_x
        num_patch_sqrt = int(math.sqrt(num_patch))
        x_local = rearrange(x, 'b c (h1 h) (w1 w) -> (b h1 w1) c h w', h1=num_patch_sqrt, w1=num_patch_sqrt)
        # print(x_local.shape, kernel.shape)
        x_local = self.kernel_deblur(x_local, kernel)
        x = rearrange(x_local, '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=num_patch_sqrt, w1=num_patch_sqrt)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_deblur(y, kernel)
        return y


class NAFBlock_patch_grid_deblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64],
                 patch_size=[96, 96]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.kernel_size = kernel_size
        self.overlap_size = [0, 0]
        # self.patch_size = [train_size[0]//4+kernel_size[0]//2 * 2, train_size[1]//4+kernel_size[1]//2 * 2]
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_deblur(in_ch=c, out_ch=c, train_size=train_size, pad_method='replicate')
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def grids(self, x, level=0):
        b, c, h, w = x.shape
        n_level = 2 ** level

        # assert b == 1
        k1, k2 = self.patch_size[0] // n_level, self.patch_size[1] // n_level
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = [self.overlap_size[0] // n_level, self.overlap_size[1] // n_level]  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])

        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1

        # import math
        step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                # if not self.ek1:
                #     # print(step_i, i, k1, h)
                #     self.ek1 = i + k1 - h # - self.overlap_size[0]
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    # if not self.ek2:
                    #     self.ek2 = j + k2 - w # + self.overlap_size[1]
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i * self.up_scale, 'j': j * self.up_scale})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        if level == 0:
            self.original_size = (b, self.out_channels, h * self.up_scale, w * self.up_scale)
            self.stride = (stride[0] * self.up_scale, stride[1] * self.up_scale)
            self.nr = num_row
            self.nc = num_col
            self.idxes = idxes
        return parts

    def get_overlap_matrix(self, h, w):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        self.h = h
        self.w = w
        self.ek1 = self.nr * self.stride[0] + self.overlap_size_up[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.overlap_size_up[1] * 2 - w
        # self.ek1, self.ek2 = 48, 224
        # print(self.ek1, self.ek2, self.nr)
        # print(self.overlap_size)
        # self.overlap_size = [8, 8]
        # self.overlap_size = [self.overlap_size[0] * 2, self.overlap_size[1] * 2]
        self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
        self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
        self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
        self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1)
        self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1)

    def grids_inverse(self, outs, out_device=None, pix_max=1.):
        if out_device is None:
            out_device = outs.device
        # print(out_device)

        type_out = torch.uint8 if pix_max == 255. else torch.float32
        # type_out = torch.int16
        preds = torch.zeros(self.original_size, device=out_device, dtype=type_out)  # .to(out_device)
        b, c, h, w = self.original_size
        # print(self.original_size)
        # outs = torch.clamp(outs, 0, 1) # * pix_max
        # outs = outs.type(type_out)
        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size_up
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)
        # rounding_mode = 'floor'
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            if i != 0 and i + k1 != h:
                outs[cnt, :, :self.overlap_size_up[0], :] = torch.mul(outs[cnt, :, :self.overlap_size_up[0], :],
                                                                      self.fuse_matrix_h2.to(outs.device))
                # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
            if i + k1 * 2 - self.ek1 < h:
                outs[cnt, :, -self.overlap_size_up[0]:, :] = torch.mul(outs[cnt, :, -self.overlap_size_up[0]:, :],
                                                                       self.fuse_matrix_h1.to(outs.device))
            # print(self.fuse_matrix_eh1.dtype)
            if i + k1 == h:
                outs[cnt, :, :self.ek1, :] = torch.mul(outs[cnt, :, :self.ek1, :], self.fuse_matrix_eh2.to(outs.device))
            if i + k1 * 2 - self.ek1 == h:
                outs[cnt, :, -self.ek1:, :] = torch.mul(outs[cnt, :, -self.ek1:, :],
                                                        self.fuse_matrix_eh1.to(outs.device))

            if j != 0 and j + k2 != w:
                outs[cnt, :, :, :self.overlap_size_up[1]] = torch.mul(outs[cnt, :, :, :self.overlap_size_up[1]],
                                                                      self.fuse_matrix_w2.to(outs.device))
            if j + k2 * 2 - self.ek2 < w:
                # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, -self.overlap_size_up[1]:] = torch.mul(outs[cnt, :, :, -self.overlap_size_up[1]:],
                                                                       self.fuse_matrix_w1.to(outs.device))
            if j + k2 == w:
                # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                outs[cnt, :, :, :self.ek2] = torch.mul(outs[cnt, :, :, :self.ek2], self.fuse_matrix_ew2.to(outs.device))
            if j + k2 * 2 - self.ek2 == w:
                # print('j + k2*2 - self.ek2 == w: ')
                outs[cnt, :, :, -self.ek2:] = torch.mul(outs[cnt, :, :, -self.ek2:],
                                                        self.fuse_matrix_ew1.to(outs.device))
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            # pred_win = outs[cnt, :, :, :].clone() * pix_max
            # pred_win = pred_win.type(type_out)
            # print(pred_win.dtype, preds.dtype, type_out, outs.dtype)
            # print(outs[cnt, :, :, :].max())
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].type(type_out).to(out_device)
            # preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].to(out_device) * pix_max
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds  # / count_mt

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)
        batch_k = kernel.shape[0]
        batch_x = x.shape[0]
        num_patch = batch_k // batch_x
        num_patch_sqrt = int(math.sqrt(num_patch))
        # dim = (self.kernel_size[0]//2, self.kernel_size[0]//2, self.kernel_size[0]//2, self.kernel_size[0]//2)
        # x_pad = torch.nn.functional.pad(x, dim, 'replicate')
        # x_local = self.grids_k(x)
        _, _, H, W = x.shape
        x_local, batch_list = window_partitionx(x, window_size=self.patch_size[0])
        # print(x_local.shape, kernel.shape)
        x_local = self.kernel_deblur(x_local, kernel)
        x = window_reversex(x_local, self.patch_size[0], H, W, batch_list)
        # x = rearrange(x_local, '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=num_patch_sqrt, w1=num_patch_sqrt)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_deblur(y, kernel)
        return y


class NAFBlock_local_patch_grid_deblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64],
                 patch_size=[96, 96]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.kernel_size = kernel_size
        self.overlap_size = [0, 0]
        # self.patch_size = [train_size[0]//4+kernel_size[0]//2 * 2, train_size[1]//4+kernel_size[1]//2 * 2]
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        base_size = [1.5 * patch_size[0], 1.5 * patch_size[1]]
        self.sca = nn.Sequential(
            AvgPool2d(base_size=base_size, train_size=patch_size, fast_imp=False),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_deblur(in_ch=c, out_ch=c, train_size=train_size, pad_method='replicate')
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def grids(self, x, level=0):
        b, c, h, w = x.shape
        n_level = 2 ** level

        # assert b == 1
        k1, k2 = self.patch_size[0] // n_level, self.patch_size[1] // n_level
        k1 = min(h, k1)
        k2 = min(w, k2)
        overlap_size = [self.overlap_size[0] // n_level, self.overlap_size[1] // n_level]  # (64, 64)

        stride = (k1 - overlap_size[0], k2 - overlap_size[1])

        num_row = (h - overlap_size[0] - 1) // stride[0] + 1
        num_col = (w - overlap_size[1] - 1) // stride[1] + 1

        # import math
        step_j = k2 if num_col == 1 else stride[1]  # math.ceil((w - stride[1]) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else stride[0]  # math.ceil((h - stride[0]) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        self.ek1, self.ek2 = None, None
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                # if not self.ek1:
                #     # print(step_i, i, k1, h)
                #     self.ek1 = i + k1 - h # - self.overlap_size[0]
                i = h - k1
                last_i = True

            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    # if not self.ek2:
                    #     self.ek2 = j + k2 - w # + self.overlap_size[1]
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i * self.up_scale, 'j': j * self.up_scale})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        if level == 0:
            self.original_size = (b, self.out_channels, h * self.up_scale, w * self.up_scale)
            self.stride = (stride[0] * self.up_scale, stride[1] * self.up_scale)
            self.nr = num_row
            self.nc = num_col
            self.idxes = idxes
        return parts

    def get_overlap_matrix(self, h, w):
        # if self.grid:
        # if self.fuse_matrix_h1 is None:
        self.h = h
        self.w = w
        self.ek1 = self.nr * self.stride[0] + self.overlap_size_up[0] * 2 - h
        self.ek2 = self.nc * self.stride[1] + self.overlap_size_up[1] * 2 - w
        # self.ek1, self.ek2 = 48, 224
        # print(self.ek1, self.ek2, self.nr)
        # print(self.overlap_size)
        # self.overlap_size = [8, 8]
        # self.overlap_size = [self.overlap_size[0] * 2, self.overlap_size[1] * 2]
        self.fuse_matrix_w1 = torch.linspace(1., 0., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
        self.fuse_matrix_w2 = torch.linspace(0., 1., self.overlap_size_up[1]).view(1, 1, self.overlap_size_up[1])
        self.fuse_matrix_h1 = torch.linspace(1., 0., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
        self.fuse_matrix_h2 = torch.linspace(0., 1., self.overlap_size_up[0]).view(1, self.overlap_size_up[0], 1)
        self.fuse_matrix_ew1 = torch.linspace(1., 0., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_ew2 = torch.linspace(0., 1., self.ek2).view(1, 1, self.ek2)
        self.fuse_matrix_eh1 = torch.linspace(1., 0., self.ek1).view(1, self.ek1, 1)
        self.fuse_matrix_eh2 = torch.linspace(0., 1., self.ek1).view(1, self.ek1, 1)

    def grids_inverse(self, outs, out_device=None, pix_max=1.):
        if out_device is None:
            out_device = outs.device
        # print(out_device)

        type_out = torch.uint8 if pix_max == 255. else torch.float32
        # type_out = torch.int16
        preds = torch.zeros(self.original_size, device=out_device, dtype=type_out)  # .to(out_device)
        b, c, h, w = self.original_size
        # print(self.original_size)
        # outs = torch.clamp(outs, 0, 1) # * pix_max
        # outs = outs.type(type_out)
        # count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size_up
        k1 = min(h, k1)
        k2 = min(w, k2)
        # if not self.h or not self.w:
        self.get_overlap_matrix(h, w)
        # rounding_mode = 'floor'
        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            if i != 0 and i + k1 != h:
                outs[cnt, :, :self.overlap_size_up[0], :] = torch.mul(outs[cnt, :, :self.overlap_size_up[0], :],
                                                                      self.fuse_matrix_h2.to(outs.device))
                # print(outs[cnt, :,  i + k1 - self.overlap_size[0]:i + k1, :].shape,
                #       self.fuse_matrix_h1.shape)
            if i + k1 * 2 - self.ek1 < h:
                outs[cnt, :, -self.overlap_size_up[0]:, :] = torch.mul(outs[cnt, :, -self.overlap_size_up[0]:, :],
                                                                       self.fuse_matrix_h1.to(outs.device))
            # print(self.fuse_matrix_eh1.dtype)
            if i + k1 == h:
                outs[cnt, :, :self.ek1, :] = torch.mul(outs[cnt, :, :self.ek1, :], self.fuse_matrix_eh2.to(outs.device))
            if i + k1 * 2 - self.ek1 == h:
                outs[cnt, :, -self.ek1:, :] = torch.mul(outs[cnt, :, -self.ek1:, :],
                                                        self.fuse_matrix_eh1.to(outs.device))

            if j != 0 and j + k2 != w:
                outs[cnt, :, :, :self.overlap_size_up[1]] = torch.mul(outs[cnt, :, :, :self.overlap_size_up[1]],
                                                                      self.fuse_matrix_w2.to(outs.device))
            if j + k2 * 2 - self.ek2 < w:
                # print(j, j + k2 - self.overlap_size[1], j + k2, self.fuse_matrix_w1.shape)
                outs[cnt, :, :, -self.overlap_size_up[1]:] = torch.mul(outs[cnt, :, :, -self.overlap_size_up[1]:],
                                                                       self.fuse_matrix_w1.to(outs.device))
            if j + k2 == w:
                # print('j + k2 == w: ', self.ek2, outs[cnt, :, :, :self.ek2].shape, self.fuse_matrix_ew1.shape)
                outs[cnt, :, :, :self.ek2] = torch.mul(outs[cnt, :, :, :self.ek2], self.fuse_matrix_ew2.to(outs.device))
            if j + k2 * 2 - self.ek2 == w:
                # print('j + k2*2 - self.ek2 == w: ')
                outs[cnt, :, :, -self.ek2:] = torch.mul(outs[cnt, :, :, -self.ek2:],
                                                        self.fuse_matrix_ew1.to(outs.device))
            # print(preds[0, :, i:i + k1, j:j + k2].shape)
            # pred_win = outs[cnt, :, :, :].clone() * pix_max
            # pred_win = pred_win.type(type_out)
            # print(pred_win.dtype, preds.dtype, type_out, outs.dtype)
            # print(outs[cnt, :, :, :].max())
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].type(type_out).to(out_device)
            # preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :].to(out_device) * pix_max
            # count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds  # / count_mt

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)
        batch_k = kernel.shape[0]
        batch_x = x.shape[0]
        num_patch = batch_k // batch_x
        num_patch_sqrt = int(math.sqrt(num_patch))
        # dim = (self.kernel_size[0]//2, self.kernel_size[0]//2, self.kernel_size[0]//2, self.kernel_size[0]//2)
        # x_pad = torch.nn.functional.pad(x, dim, 'replicate')
        # x_local = self.grids_k(x)
        _, _, H, W = x.shape
        x_local, batch_list = window_partitionx(x, window_size=self.patch_size[0])
        # print(x_local.shape, kernel.shape)
        x_local = self.kernel_deblur(x_local, kernel)
        x = window_reversex(x_local, self.patch_size[0], H, W, batch_list)
        # x = rearrange(x_local, '(b h1 w1) c h w -> b c (h1 h) (w1 w)', h1=num_patch_sqrt, w1=num_patch_sqrt)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_deblur(y, kernel)
        return y


class NAFBlock_noker_deblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        # self.kernel_deblur = kernel_deblur(in_ch=c, out_ch=c, train_size=train_size)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_deblur(y, kernel)
        return y


class FC1NAFBlock_deblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_deblur(in_ch=c, out_ch=c, train_size=train_size)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)
        # x_fourier = self.fft_block1(x)
        x = self.kernel_deblur(x, kernel)
        x_fourier = self.fft_block1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x) + x_fourier

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_deblur(y, kernel)
        return y


class NAFBlock_deblur_fourierconv(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_deblur_fourierconv(in_ch=c, out_ch=c, train_size=train_size)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)
        x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_deblur(y, kernel)
        return y


class NAFBlock_deblur_feature(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, train_size=[64, 64]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_deblur_feature(in_ch=c, out_ch=c, train_size=train_size)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel, kf = xk
        x = inp
        x = self.norm1(x)
        x = self.kernel_deblur(x, kernel, kf)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_deblur(y, kernel, kf)
        return y


class NAFBlock_reblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_reblur = kernel_reblur(in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk

        # inp = self.kernel_deblur(inp, kernel)
        x = inp
        x = self.norm1(x)
        x = self.kernel_reblur(x, kernel)
        #
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        # x = self.kernel_deblur(x, kernel)
        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_reblur(y, kernel)
        return y


class NAFBlock_patch_reblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19, patch_size=[64, 64]):
        super().__init__()
        self.patch_size = patch_size
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sk = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_reblur = kernel_reblur(in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk

        # inp = self.kernel_deblur(inp, kernel)
        x = inp
        H, W = x.shape[-2:]
        x = self.norm1(x)
        x_patch, batch_list = window_partitionx(x, self.patch_size)
        kernel = kernel * self.sk(x_patch)
        x_patch = self.kernel_reblur(x_patch, kernel)
        x = window_reversex(x_patch, self.patch_size, H, W, batch_list)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        # x = self.kernel_deblur(x, kernel)
        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_reblur(y, kernel)
        return y


class NAFBlock_logdeblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_reblur = kernel_logdeblur(in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk

        # inp = self.kernel_deblur(inp, kernel)
        x = inp
        x = self.norm1(x)
        x = self.kernel_reblur(x, kernel)
        #
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        # x = self.kernel_deblur(x, kernel)
        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_reblur(y, kernel)
        return y


class F1NAFBlock_reblur(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.fft_conv = fft_bench_complex_mlp(c, dw=DW_Expand)
        self.kernel_reblur = kernel_reblur(in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk

        # inp = self.kernel_deblur(inp, kernel)
        x = inp
        x = self.norm1(x)

        x = self.kernel_reblur(x, kernel)

        x_freq = self.fft_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x) + x_freq

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)
        # x = self.kernel_deblur(x, kernel)
        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_reblur(y, kernel)
        return y


class NAFBlock_reblur_attn(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_reblur = kernel_attn_reblur(kernel_size=kernel_size, in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        # x = self.kernel_reblur(x, kernel)
        x = self.norm1(x)
        x = self.kernel_reblur(x, kernel)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        # y = self.kernel_reblur(y, kernel)
        return y


class NAFBlock_reblur_attn_feature(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_attn_reblur_feature(kernel_size=kernel_size, in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel, kernel_feature = xk
        x = inp
        x = self.norm1(x)
        # x = self.kernel_deblur(x, kernel, kernel_feature)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        y = self.kernel_deblur(y, kernel, kernel_feature)
        return y


class NAFBlock_reblur_feature(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_reblur_feature(kernel_size=kernel_size, in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel, kernel_feature = xk
        x = inp
        x = self.norm1(x)
        # x = self.kernel_deblur(x, kernel, kernel_feature)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        y = self.kernel_deblur(y, kernel, kernel_feature)
        return y


class NAFBlock_deblur_attn(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_attn_deblur(kernel_size=kernel_size, in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel, kf = xk
        x = inp
        x = self.norm1(x)
        x = self.kernel_deblur(x, kernel, kf)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma

        return y


class NAFBlock_reblur_maxidx(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        self.kernel_size = kernel_size
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = kernel_deblur(in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        y_xk = torch.nn.functional.interpolate(y, size=[kernel.shape[-2], kernel.shape[-1]])
        b, c, h, w = y_xk.shape
        _, max_idx = torch.max(y_xk.view(b, c, -1), dim=-1)
        ks2 = kernel.shape[1]
        kernel = kernel.view(b, 1, ks2, -1)
        kernel = kernel.repeat(1, c, 1, 1)
        # print(max_idx.shape)
        max_idx = max_idx.unsqueeze(-1).repeat(1, 1, ks2).unsqueeze(-1)

        # print(max_idx.shape, max_idx)
        # print(max_idx.shape, kernel.shape)
        kernel = torch.gather(kernel, dim=-1, index=max_idx)
        # print(kernel.shape)
        kernel = kernel.view(b, c, self.kernel_size, self.kernel_size)
        y = self.kernel_deblur(y, kernel)
        return y


##########################################################################
##---------- Prompt Gen Module -----------------------
class PromptGenBlock(nn.Module):
    def __init__(self, prompt_dim=128, prompt_len=5, prompt_size=96, lin_dim=192):
        super(PromptGenBlock, self).__init__()
        self.prompt_param = nn.Parameter(torch.rand(1, prompt_len, prompt_dim, prompt_size, prompt_size))
        self.linear_layer = nn.Linear(lin_dim, prompt_len)
        self.conv3x3 = nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, stride=1, padding=1, bias=False)

    def forward(self, x):
        B, C, H, W = x.shape
        emb = x.mean(dim=(-2, -1))
        # print(x.shape)
        prompt_weights = F.softmax(self.linear_layer(emb), dim=1)
        prompt = prompt_weights.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.prompt_param.unsqueeze(0).repeat(B, 1,
                                                                                                                  1, 1,
                                                                                                                  1,
                                                                                                                  1).squeeze(
            1)
        prompt = torch.sum(prompt, dim=1)
        prompt = F.interpolate(prompt, (H, W), mode="bilinear")
        prompt = self.conv3x3(prompt)

        return prompt


class NAFBlock_reblur_softmax(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        self.kernel_size = kernel_size
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.yk_conv = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                                 groups=1, bias=True)
        self.kernel_deblur = kernel_reblur(in_ch=c, out_ch=c)
        # init_kers = torch.zeros([1, c, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.norm1(x)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        sq = int(np.sqrt(kernel.shape[1]))
        y_xk = torch.nn.functional.interpolate(y, size=[sq, sq])
        y_xk = self.yk_conv(y_xk)
        # print(kernel.shape)
        ker_attn = rearrange(y_xk, 'b c h w -> b c (h w)')
        # ker_attn = torch.sigmoid(ker_attn)
        ker_attn = torch.softmax(ker_attn, dim=-1)
        kernel = rearrange(kernel, 'b n k1 k2 -> b n (k1 k2)')
        # print(kernel.shape, ker_attn.shape)
        kernel = ker_attn @ kernel
        kernel = torch.softmax(kernel, dim=-1)
        kernel = rearrange(kernel, 'b n (k1 k2) -> b n k1 k2', k1=self.kernel_size, k2=self.kernel_size)
        y = self.kernel_deblur(y, kernel)
        return y


class NAFBlock_conv(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., feature_ch=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

        self.kernel_deblur = Spatial_Conv(in_ch=c, out_ch=c)

    def forward(self, inp):
        # inp, kernel = xk
        x = inp
        x = self.norm1(x)
        # x = self.kernel_deblur(x, kernel)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        y = y + x * self.gamma
        y = self.kernel_deblur(y)
        return y


class FNAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.kernel_atttion = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.kernel_atttion(x, kernel)
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)

        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FNAFBlock_kernel_feature(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., feature_ch=19):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.kernel_atttion = feature_attention(feature_ch, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.kernel_atttion(x, kernel)
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)

        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FCNAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True,
                                                 act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.kernel_atttion = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.kernel_atttion(x, kernel)
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)

        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)

        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FC1NAFBlock_kernel(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True,
                                                 act_method=nn.GELU)  # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.kernel_atttion = kernel_attention(kernel_size, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.kernel_atttion(x, kernel)
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)

        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)

        x = self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FCNAFBlock_kernel_feature(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., feature_ch=19):
        super().__init__()
        dw_channel = c * DW_Expand

        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True,
                                                 act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.kernel_atttion = feature_attention(feature_ch, in_ch=c, out_ch=c)

    def forward(self, xk):
        inp, kernel = xk
        x = inp
        x = self.kernel_atttion(x, kernel)
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)

        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)

        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


def generate_k(model, code, n_row=1):
    # eval???
    model.eval()

    # unconditional model
    # for a random Gaussian vector, its l2norm is always close to 1.
    # therefore, in optimization, we can constrain the optimization space to be on the sphere with radius of 1

    u = code  # [B, 19*19]
    samples, _ = model.inverse(u)

    samples = model.post_process(samples)

    return samples


def generate_k_forward(model, code, n_row=1):
    # eval???
    # model.eval()

    # unconditional model
    # for a random Gaussian vector, its l2norm is always close to 1.
    # therefore, in optimization, we can constrain the optimization space to be on the sphere with radius of 1

    u = code  # [B, 19*19]
    # samples, _ = model(u)
    samples, _ = model.inverse(u)
    samples = model.post_process(samples)

    return samples


class SimpleGate(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=self.dim)
        return x1 * x2

class ComplexSimpleGate(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        x1_real, x1_imag = x1.chunk(2, dim=0)
        x2_real, x2_imag = x2.chunk(2, dim=0)
        x12 = x1_real * x2_real 
        x21 = x1_imag * x2_imag
        x_imag = x1_real * x2_real + x1_imag * x2_imag
        x_real = x1_real * x2_imag - x2_real * x1_imag
        return torch.cat([x_real, x_imag], dim=0)
class NAFBlockold(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1,
                 DW_Expand=2, FFN_Expand=2, drop_out_rate=0., attn_type='SCA'):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.attn_type = attn_type
        # Simplified Channel Attention
        # if attn_type == 'SCA':
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            # AvgPool2d(base_size=[], fast_imp=fast_imp, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        # elif attn_type == 'grid_dct_SE':
        #     self.sca = WDCT_SE(dw_channel // 2, num_heads=num_heads, bias=True, window_size=window_size)

        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout3d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout3d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        if self.attn_type == 'SCA':
            x = x * self.sca(x)
        elif self.attn_type == 'grid_dct_SE':
            x = self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class ComplexNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = ComplexSimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.ones((1, c, 1, 1)),
                                                      requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                      requires_grad=True))
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma





class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class Glayer(nn.Module):
    def __init__(self, dim, d_state=32, d_conv=4, expand=2):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,  # Model dimension d_model
            d_state=d_state,  # SSM state expansion factor
            d_conv=d_conv,  # Local convolution width
            expand=expand,  # Block expansion factor
        )

    @autocast(enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        B, C = x.shape[:2]
        assert C == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, C, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, C, *img_dims)

        return out


class Llayer(nn.Module):
    def __init__(self, c, DW_Expand=2, train_size=[256, 256]):
        super().__init__()
        dw_channel = c * DW_Expand
        base_size = [int(train_size[-2] * 1.5), int(train_size[-1] * 1.5)]
        # base_size = train_size
        self.ap = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size)
        )

        self.conv1 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0,
                               stride=1,
                               groups=1, bias=True)

        self.conv2 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

    def forward(self, x):
        inp = x
        x = self.ap(x)
        x = self.conv1(x)
        x = inp * x
        x = self.conv2(x)
        return x


class ALGBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., train_size=[256, 256], patch_size=[256, 256]):
        super().__init__()
        self.sg = SimpleGate()
        self.patch_size = patch_size
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)

        # self.conv3_2 = nn.Conv2d(in_channels=c * 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
        #                          bias=True)
        # Simplified Channel Attention


        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.g_layer = Glayer(c)
        self.l_layer = Llayer(c, train_size=train_size)

        self.inside_all = nn.Parameter(torch.zeros(c, 1, 1), requires_grad=True)
        self.lamb_g = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.lamb_l = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        H, W = x.shape[-2:]
        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):
            x_g, batch_list = window_partitionx(x, self.patch_size)
            # print(x.shape, batch_list)
        else:
            x_g = x
        x_g = self.g_layer(x_g)
        if self.patch_size is not None and (H > self.patch_size[0] or W > self.patch_size[1]):

            x_g = window_reversex(x_g, self.patch_size, H, W, batch_list)



        x_l = self.l_layer(x)

        x_g = x_g * (self.inside_all + 1.)
        x_g = x_g * self.lamb_g # [None, :, None, None]
        x_l = x_l * self.lamb_l # [None, :, None, None]
        x = x_g + x_l

        x = self.dropout1(x)
        y = inp + x

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.beta


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class NAFBlock_patch(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., patch_size=[8,8]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.patch_size=patch_size
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))


    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        b, c, H, W = x.shape

        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x, batch_list = window_partitionx(x, self.patch_size)
        # print(x.shape)
        x = x * self.sca(x)
        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x = window_reversex(x, self.patch_size, H, W, batch_list)

        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class NAFBlock_nosca(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., patch_size=[8,8]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.patch_size=patch_size
        # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))


    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)

        # b, c, H, W = x.shape
        #
        # if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
        #     x, batch_list = window_partitionx(x, self.patch_size)
        # # print(x.shape)
        # x = x * self.sca(x)
        # if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
        #     x = window_reversex(x, self.patch_size, H, W, batch_list)

        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class NAFBlock_patch_grid(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., patch_size=[8,8]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.kernel_size=patch_size
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))


    def grids(self, x):
        b, c, h, w = x.shape
        self.original_size = (b, c, h, w)
        assert b == 1
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)
        num_row = (h - 1) // k1 + 1
        num_col = (w - 1) // k2 + 1
        self.nr = num_row
        self.nc = num_col

        import math
        step_j = k2 if num_col == 1 else math.ceil((w - k2) / (num_col - 1) - 1e-8)
        step_i = k1 if num_row == 1 else math.ceil((h - k1) / (num_row - 1) - 1e-8)

        parts = []
        idxes = []
        i = 0  # 0~h-1
        last_i = False
        while i < h and not last_i:
            j = 0
            if i + k1 >= h:
                i = h - k1
                last_i = True
            last_j = False
            while j < w and not last_j:
                if j + k2 >= w:
                    j = w - k2
                    last_j = True
                parts.append(x[:, :, i:i + k1, j:j + k2])
                idxes.append({'i': i, 'j': j})
                j = j + step_j
            i = i + step_i

        parts = torch.cat(parts, dim=0)
        self.idxes = idxes
        return parts

    def grids_inverse(self, outs):
        preds = torch.zeros(self.original_size).to(outs.device)
        b, c, h, w = self.original_size

        count_mt = torch.zeros((b, 1, h, w)).to(outs.device)
        k1, k2 = self.kernel_size
        k1 = min(h, k1)
        k2 = min(w, k2)

        for cnt, each_idx in enumerate(self.idxes):
            i = each_idx['i']
            j = each_idx['j']
            preds[0, :, i:i + k1, j:j + k2] += outs[cnt, :, :, :]
            count_mt[0, 0, i:i + k1, j:j + k2] += 1.

        del outs
        torch.cuda.empty_cache()
        return preds / count_mt
    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = self.grids(x)
        x = x * self.sca(x)
        x = self.grids_inverse(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class LocalNAFBlock(nn.Module):
    def __init__(self, c, train_size, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        base_size = [int(train_size[-2]*1.5), int(train_size[-1]*1.5)]
        # base_size = train_size
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                       requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                       requires_grad=True))

        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class LocalLinearNAFBlock(nn.Module):
    def __init__(self, c, train_size, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Linear(c, dw_channel, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel, bias=True)
        self.conv3 = nn.Linear(dw_channel // 2, c, bias=True)

        # Simplified Channel Attention
        base_size = [int(train_size[-2]*1.5), int(train_size[-1]*1.5)]
        # base_size = train_size
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg1 = SimpleGate(1)
        self.sg2 = SimpleGate(-1)

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Linear(c, ffn_channel, bias=True)
        self.conv5 = nn.Linear(ffn_channel // 2, c,  bias=True)
        # self.norm1 = nn.LayerNorm(c)
        # self.norm2 = nn.LayerNorm(c)
        self.norm1 = LayerNorm2d_hwc(c)
        self.norm2 = LayerNorm2d_hwc(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, 1, 1, c)),
                                                       requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, 1, 1, c)),
                                                       requires_grad=True))

        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # B N C
        B, C, H, W = inp.shape
        x = inp

        x_inp = x.permute(0, 2, 3, 1)
        x = self.norm1(x_inp)

        x = self.conv1(x)

        x = x.permute(0, 3, 1, 2)
        x = self.conv2(x)
        x = self.sg1(x)
        x = x * self.sca(x)
        x = x.permute(0, 2, 3, 1)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = x_inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg2(x)
        x = self.conv5(x) * self.gamma

        x = self.dropout2(x)
        y = x + y
        # x = x.transpose(1, 2).contiguous().view(B, -1, H, W)
        return y.permute(0, 3, 1, 2)
class KerBNBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.act = nn.ReLU()
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        # self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = nn.BatchNorm2d(c) # LayerNorm2d(c)
        self.norm2 = nn.BatchNorm2d(c) # LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.act(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class KerLNBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.act = nn.ReLU()
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        # self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.act(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class NAFwoSGBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.act = nn.GELU()
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        # self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.act(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class NAFwoSGSCABlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.act = nn.GELU()
        # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

        # SimpleGate
        # self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        # x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.act(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class NAFwoSGSCAFFNBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.act = nn.GELU()
        # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

        # SimpleGate
        # self.sg = SimpleGate()

        # ffn_channel = FFN_Expand * c
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        # self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        # x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta
        #
        # x = self.conv4(self.norm2(y))
        # x = self.act(x)
        # x = self.conv5(x)
        #
        # x = self.dropout2(x)

        return y
class NAFwoFFNBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        # ffn_channel = FFN_Expand * c
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        # self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        # x = self.conv4(self.norm2(y))
        # x = self.sg(x)
        # x = self.conv5(x)
        #
        # x = self.dropout2(x)

        return y # + x * self.gamma

class NAFGELUwoFFNBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = nn.GELU()

        # ffn_channel = FFN_Expand * c
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        # self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x1, x2 = x.chunk(2, dim=1)
        x = x1 * self.sg(x2)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        # x = self.conv4(self.norm2(y))
        # x = self.sg(x)
        # x = self.conv5(x)
        #
        # x = self.dropout2(x)

        return y # + x * self.gamma
class NAFwoSCAFFNBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

        # SimpleGate
        self.sg = SimpleGate()

        # ffn_channel = FFN_Expand * c
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        # self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        # x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        # x = self.conv4(self.norm2(y))
        # x = self.sg(x)
        # x = self.conv5(x)
        #
        # x = self.dropout2(x)

        return y # + x * self.gamma
class NAFwoSGSCAFFNwMambaBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, d_state=16, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.act = nn.GELU()
        # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

        # SimpleGate
        # self.sg = SimpleGate()

        # ffn_channel = FFN_Expand * c
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.attn = EVSBlock(c, d_state=d_state)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x)
        # x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.attn(self.norm2(y))


        x = self.dropout2(x)

        return y + x * self.gamma
class Kerv2LNBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel//2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.act = nn.GELU()
        # Simplified Channel Attention
        # self.sca = nn.Sequential(
        #     nn.AdaptiveAvgPool2d(1),
        #     nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=1, padding=0, stride=1,
        #               groups=1, bias=True),
        # )

        # SimpleGate
        # self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel//2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2
        # x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x1, x2 = x.chunk(2, dim=1)
        x = self.act(x1) * x2
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class NAFResBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class LNResBlock(nn.Module):
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        # dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)

        self.act = nn.ReLU()
        # SimpleGate

        self.norm1 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                     requires_grad=True))
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)

        x = self.dropout1(x)


        return inp + x * self.beta
class LNResBlock1x1(nn.Module):
    def __init__(self, c, drop_out_rate=0.):
        super().__init__()
        # dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1,
                               bias=True)

        self.act = nn.ReLU()
        # SimpleGate

        self.norm1 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                     requires_grad=True))
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)

        x = self.dropout1(x)


        return inp + x * self.beta
class FuseNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.conv_1 = nn.Conv2d(c, c, 1, 1, 0)
        self.conv_2 = nn.Conv2d(c, c, 1, 1, 0)
    def forward(self, enc, dnc):
        # enc, dnc = x_
        y = self.conv_1(torch.cat((enc, dnc), dim=1))
        # print(y.shape)
        x = self.norm1(y)

        x = self.conv1(x)
        # print(x.shape)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = y + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)
        x = y + x * self.gamma
        x = self.conv_2(x)
        e, d = x.chunk(2, dim=1)
        output = e + d

        return output
class NAF1x1Block(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm2 = LayerNorm2d(c)
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):


        x = self.conv4(self.norm2(inp))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return inp + x * self.gamma

class MaskMambaNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm0 = LayerNorm2d(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = MaskedMamba(d_model=c, d_state=d_state, expand=DW_Expand, dropout=drop_out_rate, **kwargs)
        self.dropout0 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.fft_conv = fft_bench_complex_mlp(c, dw=DW_Expand)

    def bit_mask(self, d_seg):
        num_classes = d_seg.shape[1]
        if self.training:
            d_seg = F.gumbel_softmax(d_seg, tau=1, hard=True, dim=1)
        else:
            d_seg = d_seg.permute(0, 2, 3, 1)
            d_seg = torch.argmax(d_seg, dim=-1)
            d_seg = F.one_hot(d_seg, num_classes=num_classes)
            d_seg = d_seg.permute(0, 3, 1, 2)
        # d_seg = d_seg.ge(0.5)
        return d_seg

    def forward(self, inps):
        inp, mask = inps

        mask = self.bit_mask(mask)
        # mask_blur = mask[:, 1, :, :].contiguous()
        inp_mamba = self.mamba(self.norm0(inp), mask)
        inp_mamba = self.dropout0(inp_mamba) * self.alpha + inp

        x = inp_mamba

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)

        x = self.conv3(x)
        x = self.dropout1(x) * self.beta

        y = inp + x

        # bs, ch = y.shape[:2]

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class MaskMambaF1NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm0 = LayerNorm2d(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = MaskedMamba(d_model=c, d_state=d_state, expand=DW_Expand, dropout=drop_out_rate, **kwargs)
        self.dropout0 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.fft_conv = fft_bench_complex_mlp(c, dw=DW_Expand)

    def bit_mask(self, d_seg):
        num_classes = d_seg.shape[1]
        if self.training:
            d_seg = F.gumbel_softmax(d_seg, tau=1, hard=True, dim=1)
        else:
            d_seg = d_seg.permute(0, 2, 3, 1)
            d_seg = torch.argmax(d_seg, dim=-1)
            d_seg = F.one_hot(d_seg, num_classes=num_classes)
            d_seg = d_seg.permute(0, 3, 1, 2)
        # d_seg = d_seg.ge(0.5)
        return d_seg

    def forward(self, inps):
        inp, mask = inps

        mask = self.bit_mask(mask)
        # mask_blur = mask[:, 1, :, :].contiguous()
        inp_mamba = self.mamba(self.norm0(inp), mask)
        inp_mamba = self.dropout0(inp_mamba) * self.alpha + inp

        x = inp_mamba

        x = self.norm1(x)
        x_freq = self.fft_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)

        x = self.conv3(x) + x_freq
        x = self.dropout1(x) * self.beta

        y = inp + x

        # bs, ch = y.shape[:2]

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class VMambaF1NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm0 = LayerNorm2d(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = VMamba(d_model=c, d_state=d_state, expand=DW_Expand, dropout=drop_out_rate, **kwargs)
        self.dropout0 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.fft_conv = fft_bench_complex_mlp(c, dw=DW_Expand)

    def bit_mask(self, d_seg):
        num_classes = d_seg.shape[1]
        if self.training:
            d_seg = F.gumbel_softmax(d_seg, tau=1, hard=True, dim=1)
        else:
            d_seg = d_seg.permute(0, 2, 3, 1)
            d_seg = torch.argmax(d_seg, dim=-1)
            d_seg = F.one_hot(d_seg, num_classes=num_classes)
            d_seg = d_seg.permute(0, 3, 1, 2)
        # d_seg = d_seg.ge(0.5)
        return d_seg

    def forward(self, inp):


        x = inp

        x = self.norm1(x)
        x_freq = self.fft_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)

        x = self.conv3(x) + x_freq
        x = self.dropout1(x) * self.beta

        y = inp + x

        # mask_blur = mask[:, 1, :, :].contiguous()
        mamba_out = self.mamba(self.norm0(y))
        y = self.dropout0(mamba_out) * self.alpha + y

        # bs, ch = y.shape[:2]

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class MaskF1NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # self.norm0 = LayerNorm2d(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # self.mamba = MaskedMamba(d_model=c, d_state=d_state, expand=DW_Expand, dropout=drop_out_rate, **kwargs)
        # self.dropout0 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.fft_conv = fft_bench_complex_mlp(c, dw=DW_Expand)

    def bit_mask(self, d_seg):
        num_classes = d_seg.shape[1]
        if self.training:
            d_seg = F.gumbel_softmax(d_seg, tau=1, hard=True, dim=1)
        else:
            d_seg = d_seg.permute(0, 2, 3, 1)
            d_seg = torch.argmax(d_seg, dim=-1)
            d_seg = F.one_hot(d_seg, num_classes=num_classes)
            d_seg = d_seg.permute(0, 3, 1, 2)
        # d_seg = d_seg.ge(0.5)
        return d_seg

    def forward(self, inps):
        inp, mask = inps

        mask = self.bit_mask(mask)
        # mask = torch.softmax(mask.float(), dim=1)
        _, mask_blur = mask.chunk(2, dim=1)

        x = inp

        x = self.norm1(x)
        x_freq = self.fft_conv(x * mask_blur.float())
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)

        x = self.conv3(x) + x_freq
        x = self.dropout1(x) * self.beta

        y = inp + x

        # bs, ch = y.shape[:2]

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class MaskSCAF1NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        # self.norm0 = LayerNorm2d(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # self.mamba = MaskedMamba(d_model=c, d_state=d_state, expand=DW_Expand, dropout=drop_out_rate, **kwargs)
        # self.dropout0 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.fft_conv = fft_bench_complex_mlp(c, dw=DW_Expand)

    def bit_mask(self, d_seg):
        num_classes = d_seg.shape[1]
        if self.training:
            d_seg = F.gumbel_softmax(d_seg, tau=1, hard=True, dim=1)
        else:
            d_seg = d_seg.permute(0, 2, 3, 1)
            d_seg = torch.argmax(d_seg, dim=-1)
            d_seg = F.one_hot(d_seg, num_classes=num_classes)
            d_seg = d_seg.permute(0, 3, 1, 2)
        # d_seg = d_seg.ge(0.5)
        return d_seg

    def forward(self, inps):
        inp, mask = inps

        # mask = self.bit_mask(mask)
        mask = torch.softmax(mask.float(), dim=1)
        _, mask_blur = mask.chunk(2, dim=1)

        x = inp

        x = self.norm1(x)
        x_freq = self.fft_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x * mask_blur)

        x = self.conv3(x) + x_freq
        x = self.dropout1(x) * self.beta

        y = inp + x

        # bs, ch = y.shape[:2]

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class MaskMambaPatchF1NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, patch_size=[64, 64], **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm0 = LayerNorm2d(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = MaskedMamba(d_model=c, d_state=d_state, expand=DW_Expand, dropout=drop_out_rate, **kwargs)
        self.dropout0 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.fft_conv = fft_bench_complex_mlp(c, dw=DW_Expand, window_size=patch_size)

    def bit_mask(self, d_seg):
        num_classes = d_seg.shape[1]
        if self.training:
            d_seg = F.gumbel_softmax(d_seg, tau=1, hard=True, dim=1)
        else:
            d_seg = d_seg.permute(0, 2, 3, 1)
            d_seg = torch.argmax(d_seg, dim=-1)
            d_seg = F.one_hot(d_seg, num_classes=num_classes)
            d_seg = d_seg.permute(0, 3, 1, 2)
        # d_seg = d_seg.ge(0.5)
        return d_seg

    def forward(self, inps):
        inp, mask = inps

        mask = self.bit_mask(mask)
        mask_blur = mask[:, 1, :, :]
        inp_mamba = self.mamba(self.norm0(inp), mask_blur)
        inp_mamba = self.dropout0(inp_mamba) * self.alpha + inp

        x = inp_mamba

        x = self.norm1(x)
        x_freq = self.fft_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)

        x = self.conv3(x) + x_freq
        x = self.dropout1(x) * self.beta

        y = inp + x

        # bs, ch = y.shape[:2]

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma

class PatchMaskMambaF1NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, patch_size=[64, 64], **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        # print(patch_size)
        self.patch_size = patch_size
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.norm0 = LayerNorm2d(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = PatchMaskedMamba(d_model=c, d_state=d_state, expand=DW_Expand, dropout=drop_out_rate, patch_size=patch_size, **kwargs)
        self.dropout0 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.alpha = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.fft_conv = fft_bench_complex_mlp(c, dw=DW_Expand, window_size=patch_size)

    def bit_mask(self, d_seg):
        num_classes = d_seg.shape[1]
        if self.training:
            d_seg = F.gumbel_softmax(d_seg, tau=1, hard=True, dim=1)
        else:
            d_seg = d_seg.permute(0, 2, 3, 1)
            d_seg = torch.argmax(d_seg, dim=-1)
            d_seg = F.one_hot(d_seg, num_classes=num_classes)
            d_seg = d_seg.permute(0, 3, 1, 2)
        # d_seg = d_seg.ge(0.5)
        return d_seg

    def forward(self, inps):
        inp, mask = inps

        mask = self.bit_mask(mask)
        mask, _ = window_partitionx(mask, self.patch_size)

        mask_blur = mask[:, 1, :, :]
        # print(mask_blur.shape, inp.shape)
        inp_mamba = self.mamba(self.norm0(inp), mask_blur)
        inp_mamba = self.dropout0(inp_mamba) * self.alpha + inp

        x = inp_mamba

        x = self.norm1(x)
        x_freq = self.fft_conv(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x, batch_list = window_partitionx(x, self.patch_size)
        x = x * self.sca(x)
        if self.patch_size is not None and (H != self.patch_size[0] or W != self.patch_size[1]):
            x = window_reversex(x, self.patch_size, H, W, batch_list)
        x = self.conv3(x) + x_freq
        x = self.dropout1(x) * self.beta

        y = inp + x

        # bs, ch = y.shape[:2]

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class MambaNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = VMamba(d_model=c, d_state=d_state, expand=DW_Expand, dropout=drop_out_rate, **kwargs)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # x = inp
        y = inp

        y = self.mamba(self.norm2(y))
        # x = self.sg(x)
        # x = self.conv5(x)

        y = self.dropout2(y)
        y = inp + y * self.gamma

        x = self.norm1(y)

        x = self.conv1(x)
        x = self.conv2(x)
        x_ = self.sg(x)
        x = x_ * self.sca(x_)
        x = self.conv3(x)

        y = y + self.dropout1(x) * self.beta # + self.mamba(x_)


        return y

class MambaForwardNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, d_conv=3, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = VMambaF(d_model=c, d_state=d_state, d_conv=d_conv, expand=DW_Expand, dropout=drop_out_rate, **kwargs)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # x = inp
        y = inp

        y = self.mamba(self.norm2(y))
        # x = self.sg(x)
        # x = self.conv5(x)

        y = self.dropout2(y)
        y = inp + y * self.gamma

        x = self.norm1(y)

        x = self.conv1(x)
        x = self.conv2(x)
        x_ = self.sg(x)
        x = x_ * self.sca(x_)
        x = self.conv3(x)

        y = y + self.dropout1(x) * self.beta # + self.mamba(x_)


        return y
class FMambaForwardNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, d_conv=3, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()
        self.fft_block1 = fft_bench_complex_linear(c, 1, window_size=None, bias=True,
                                                   act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_linear(c, 1, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = VMambaF(d_model=c, d_state=d_state, d_conv=d_conv, expand=DW_Expand, dropout=drop_out_rate, **kwargs)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # x = inp
        y = inp
        y = self.norm2(y)
        y = self.mamba(y) + self.fft_block2(y)
        # x = self.sg(x)
        # x = self.conv5(x)

        y = self.dropout2(y)
        y = inp + y * self.gamma

        x = self.norm1(y)
        x_freq = self.fft_block1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x) + x_freq

        y = y + self.dropout1(x) * self.beta # + self.mamba(x_)


        return y
class FdownMambaForwardNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, d_conv=3, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()
        self.fft_block1 = fft_bench_complex_linear(c, 1, window_size=None, bias=True,
                                                   act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_linear(c, 1, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = DownScaleMambaF(d_model=c, d_state=d_state, d_conv=d_conv, expand=DW_Expand, dropout=drop_out_rate, **kwargs)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # x = inp
        y = inp
        y = self.norm2(y)
        y = self.mamba(y) + self.fft_block2(y)
        # x = self.sg(x)
        # x = self.conv5(x)

        y = self.dropout2(y)
        y = inp + y * self.gamma

        x = self.norm1(y)
        x_freq = self.fft_block1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x) + x_freq

        y = y + self.dropout1(x) * self.beta # + self.mamba(x_)


        return y

class FshuffleMambaForwardNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, d_conv=3, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()
        self.fft_block1 = fft_bench_complex_linear(c, 1, window_size=None, bias=True,
                                                   act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_linear(c, 1, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = ShuffledMamba(d_model=c, d_state=d_state, d_conv=d_conv, expand=DW_Expand, dropout=drop_out_rate, **kwargs)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # x = inp
        y = inp
        y = self.norm2(y)
        y = self.mamba(y) + self.fft_block2(y)
        # x = self.sg(x)
        # x = self.conv5(x)

        y = self.dropout2(y)
        y = inp + y * self.gamma

        x = self.norm1(y)
        x_freq = self.fft_block1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x) + x_freq

        y = y + self.dropout1(x) * self.beta # + self.mamba(x_)


        return y
class FLocalMambaForwardNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, d_conv=3, window_size=8, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()
        self.fft_block1 = fft_bench_complex_linear(c, 1, window_size=None, bias=True,
                                                   act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_linear(c, 1, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = LocalMambaF(d_model=c, d_state=d_state, d_conv=d_conv, expand=DW_Expand, dropout=drop_out_rate, **kwargs)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # x = inp
        y = inp
        y = self.norm2(y)
        y = self.mamba(y) + self.fft_block2(y)
        # x = self.sg(x)
        # x = self.conv5(x)

        y = self.dropout2(y)
        y = inp + y * self.gamma

        x = self.norm1(y)
        x_freq = self.fft_block1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x) + x_freq

        y = y + self.dropout1(x) * self.beta # + self.mamba(x_)


        return y
class MambaV2NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, num_heads=1, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = Mamba2(d_model=c, d_state=d_state, expand=DW_Expand, headdim=c//num_heads, **kwargs)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # x = inp
        y = inp

        y = self.mamba(self.norm2(y))
        # x = self.sg(x)
        # x = self.conv5(x)

        y = self.dropout2(y)
        y = inp + y * self.gamma

        x = self.norm1(y)

        x = self.conv1(x)
        x = self.conv2(x)
        x_ = self.sg(x)
        x = x_ * self.sca(x_)
        x = self.conv3(x)

        y = y + self.dropout1(x) * self.beta # + self.mamba(x_)


        return y

class ATDMambaNAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, num_heads=1, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        # self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
        #                        bias=True)
        # self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
        #                        groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = ATDMambaF(d_model=c, d_state=d_state, expand=DW_Expand, dropout=drop_out_rate, num_mask=num_heads*12, **kwargs)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        # x = inp
        y = inp

        y = self.mamba(self.norm2(y))
        # x = self.sg(x)
        # x = self.conv5(x)

        y = self.dropout2(y)
        y = inp + y * self.gamma

        x = self.norm1(y)

        x = self.conv1(x)
        x = self.conv2(x)
        x_ = self.sg(x)
        x = x_ * self.sca(x_)
        x = self.conv3(x)

        y = y + self.dropout1(x) * self.beta # + self.mamba(x_)


        return y

class MambaF1NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()
        self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = SS2D(d_model=c, d_state=d_state, expand=DW_Expand, dropout=drop_out_rate, **kwargs)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x_freq = self.fft_block1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x_ = self.sg(x)
        x = x_ * self.sca(x_)
        x = self.conv3(x) + x_freq

        x = self.dropout1(x) * self.beta + self.mamba(x_)

        y = inp + x

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class VMambaF1NAFBlockV1(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., d_state=16, **kwargs):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()
        self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        self.mamba = VMamba(d_model=c, d_state=d_state, expand=DW_Expand, dropout=drop_out_rate, **kwargs)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x_freq = self.fft_block1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x_ = self.sg(x)
        x = x_ * self.sca(x_)
        x = self.conv3(x) + x_freq

        x = self.dropout1(x) * self.beta + self.mamba(x_)

        y = inp + x

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma
class NAFBlock_local(nn.Module):
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0., train_size=[], patch_size=[]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        base_size = [1.5 * patch_size[0], 1.5 * patch_size[1]]
        self.sca = nn.Sequential(
            AvgPool2d(base_size=base_size, train_size=patch_size, fast_imp=False),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class DCNv3Block(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1,
                 DW_Expand=2, FFN_Expand=2, drop_out_rate=0., attn_type='SCA'):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        # self.dcn_conv2 = DCNv3(channels=dw_channel // 2, kernel_size=3, pad=1, stride=1, group=num_heads)
        self.dcn_conv2 = DCNv3_pytorch(channels=dw_channel // 2, kernel_size=3, pad=1, stride=1, group=num_heads,
                                       offset_scale=1.0,
                                       act_layer='GELU',
                                       norm_layer='LN',
                                       center_feature_scale=True,
                                       remove_center=False)
        self.conv2 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=3, padding=1,
                               stride=1,
                               groups=dw_channel // 2,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)
        self.attn_type = attn_type
        # Simplified Channel Attention
        if attn_type == 'SCA':
            self.sca = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                          groups=1, bias=True),
            )
        elif attn_type == 'grid_dct_SE':
            self.sca = WDCT_SE(dw_channel // 2, num_heads=num_heads, bias=True, window_size=window_size)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x1, x2 = x.chunk(2, dim=1)
        # x1 = x1.permute(0, 2, 3, 1)
        x1 = self.dcn_conv2(x1)
        # x1 = x1.permute(0, 3, 1, 2)
        x2 = self.conv2(x2)
        x = x1 * x2
        if self.attn_type == 'SCA':
            x = x * self.sca(x)
        elif self.attn_type == 'grid_dct_SE':
            x = self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma


class FDCNv3Block(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.dcn_conv2 = DCNv3_pytorch(channels=dw_channel // 2, kernel_size=3, pad=1, stride=1, group=num_heads,
                                       offset_scale=1.0,
                                       act_layer='GELU',
                                       norm_layer='LN',
                                       center_feature_scale=True,
                                       remove_center=False)
        self.conv2 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=3, padding=1,
                               stride=1,
                               groups=dw_channel // 2,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()

        self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x1, x2 = x.chunk(2, dim=1)
        # x1 = x1.permute(0, 2, 3, 1)
        x1 = self.dcn_conv2(x1)
        # x1 = x1.permute(0, 3, 1, 2)
        x2 = self.conv2(x2)
        x = x1 * x2

        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)

        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class LocalFourierComplexNAFBlock(nn.Module):
    def __init__(self, c, train_size, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, DW_Fourier=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        base_size = [int(train_size[-2] * 1.5), int(train_size[-1] * 1.5)]
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_linear(c, DW_Fourier, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_linear(c, DW_Fourier, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.register_parameter('beta', nn.Parameter(torch.ones((1, c, 1, 1)),
                                                     requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                      requires_grad=True))
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class LocalFourierConcatChannelNAFBlock(nn.Module):
    def __init__(self, c, train_size, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, DW_Fourier=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        base_size = [int(train_size[-2] * 1.5), int(train_size[-1] * 1.5)]
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_concat_channl_conv(c, DW_Fourier, window_size=None, bias=True, act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_concat_channl_conv(c, DW_Fourier, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(self.avgpool(x))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class LocalFourierConcatChannelDWConvNAFBlock(nn.Module):
    def __init__(self, c, train_size, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, DW_Fourier=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        base_size = [int(train_size[-2] * 1.5), int(train_size[-1] * 1.5)]
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_concat_channl_dwconv(c, DW_Fourier, window_size=None, bias=True, act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_concat_channl_dwconv(c, DW_Fourier, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class LocalFourierConcatBatchNAFBlock(nn.Module):
    def __init__(self, c, train_size, num_heads=1, window_size=8, patch_size=None, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, DW_Fourier=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        base_size = [int(train_size[-2] * 1.5), int(train_size[-1] * 1.5)]
        self.window_size = window_size  # window_size
        self.window_size_fft = patch_size
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_concat_batch_conv(c, DW_Fourier, window_size=patch_size, bias=True, act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_concat_batch_conv(c, DW_Fourier, window_size=patch_size, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                     requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)),
                                                      requires_grad=True))
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # for p in self.norm1.parameters(): p.requires_grad = False
        # for p in self.norm2.parameters(): p.requires_grad = False
        # for p in self.conv1.parameters(): p.requires_grad = False
        # for p in self.conv2.parameters(): p.requires_grad = False
        # for p in self.sca.parameters(): p.requires_grad = False
        # for p in self.conv3.parameters(): p.requires_grad = False
        # for p in self.conv4.parameters(): p.requires_grad = False
        # for p in self.conv5.parameters(): p.requires_grad = False
    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FourierConcatBatchNAFBlock(nn.Module):
    def __init__(self, c, train_size, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, DW_Fourier=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        base_size = [int(train_size[-2] * 1.5), int(train_size[-1] * 1.5)]
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_concat_batch_conv(c, DW_Fourier, window_size=None, bias=True, act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_concat_batch_conv(c, DW_Fourier, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.register_parameter('beta', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))
        self.register_parameter('gamma', nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True))
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class LocalFourierConcatSplitNAFBlock(nn.Module):
    def __init__(self, c, train_size, num_heads=1, window_size=8, window_size_fft=None, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, DW_Fourier=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        base_size = [int(train_size[-2] * 1.5), int(train_size[-1] * 1.5)]
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        # self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True, act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        # print(x.shape, self.fft_block1(x).shape)
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        # print(x.shape, self.fft_block2(x).shape)
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FourierConcatSplitNAFBlock(nn.Module):
    def __init__(self, c, train_size, num_heads=1, window_size=8, window_size_fft=None, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, DW_Fourier=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        base_size = [int(train_size[-2] * 1.5), int(train_size[-1] * 1.5)]
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True, act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_concat_split_conv(c, DW_Fourier, window_size=window_size_fft, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(self.avgpool(x))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        # print(x.shape, self.fft_block1(x).shape)
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        # print(x.shape, self.fft_block2(x).shape)
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class LocalFourierConcatSplitDwConvNAFBlock(nn.Module):
    def __init__(self, c, train_size, num_heads=1, window_size=8, window_size_fft=None, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, DW_Fourier=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        base_size = [int(train_size[-2] * 1.5), int(train_size[-1] * 1.5)]
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            AvgPool2d(kernel_size=base_size, base_size=base_size, train_size=train_size),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_concat_split_dwconv(c, DW_Fourier, window_size=window_size_fft, bias=True, act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_concat_split_dwconv(c, DW_Fourier, window_size=window_size_fft, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FourierNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, DW_Fourier=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_linear(c, DW_Fourier, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_linear(c, DW_Fourier, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(self.avgpool(x))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x

class FourierKerNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, DW_Fourier=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_ker_linear(c, DW_Fourier, window_size=None, num_ker=num_heads*4, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_linear(c, DW_Fourier, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(self.avgpool(x))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class LogFourierNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, DW_Fourier=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = log_fft_bench_complex_linear(c, DW_Fourier, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = log_fft_bench_complex_linear(c, DW_Fourier, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(self.avgpool(x))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FourierNAFResBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_linear(c, 1, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_linear(c, 1, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=3, padding=1, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FourierSCANAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_linear(c, 1, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_linear(c, 1, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x_sg = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x_sg * self.sca(x_sg)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        # x = x + self.fft_block1(x_sg)
        x = self.dropout1(x)
        y = inp + x * self.beta + self.fft_block1(x_sg)

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class Fourier2ActNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_2act_bench_complex_linear(c, 1, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_2act_bench_complex_linear(c, 1, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class FourierNAFBlock2(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_linear(c, 1, window_size=None, bias=True,
                                                   act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_linear(c, 1, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)

        self.conv5 = nn.Conv2d(in_channels=ffn_channel, out_channels=ffn_channel, kernel_size=3, padding=1, stride=1,
                               groups=ffn_channel,
                               bias=True)

        # Simplified Channel Attention
        self.sca2 = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.conv6 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # print(drop_out_rate)
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x_ = self.norm2(y)

        x = self.conv4(x_)
        x = self.conv5(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)
        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca2(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv6(x)

        x = x + self.fft_block2(x_)
        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class F1x1conv2dNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_1x1conv2d(c, 1, window_size=None, bias=True, act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_1x1conv2d(c, 1, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class Fourierx2NAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_linear(c, 2, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_linear(c, 2, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x
class F1NAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=None, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=window_size_fft, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)

        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        x = self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FCNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        # self.avg = nn.AdaptiveAvgPool2d(1)
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True,
                                                 act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))
        # x = x * self.sca(self.avg(x))
        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FC1NAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True,
                                                 act_method=nn.GELU)  # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FC1NAFBlockDFFN(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True,
                                                 act_method=nn.GELU)  # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.ffn = DFFN(dim=c, ffn_expansion_factor=FFN_Expand, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.ffn(x)

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FC1NAFBlockFFN(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True,
                                                 act_method=nn.GELU)  # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.ffn = FFN(dim=c, ffn_expansion_factor=FFN_Expand, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.ffn(x)

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, dim_out=None):
        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        if dim_out is None:
            dim_out = dim
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim_out, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, dim_out=None):
        super(FFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        if dim_out is None:
            dim_out = dim
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        # self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim_out, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        # x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
        #                     patch2=self.patch_size)
        # x_patch_fft = torch.fft.rfft2(x_patch.float())
        # x_patch_fft = x_patch_fft * self.fft
        # x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        # x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
        #               patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class FC1XNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1, train_size=[256, 256]):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv_phase_tig(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU,
                                                           train_size=train_size)  # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block1(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class SFconv(nn.Module):
    def __init__(self, features, M=2, r=2, L=32) -> None:
        super().__init__()

        d = max(int(features / r), L)
        self.features = features

        self.fc = nn.Conv2d(features, d, 1, 1, 0)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                nn.Conv2d(d, features, 1, 1, 0)
            )
        self.sc = nn.Conv2d(features * 2, features * 2, 1, 1, 0)

        self.softmax = nn.Softmax(dim=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        # self.gap = AvgPool2d(base_size=80)

        self.out = nn.Conv2d(features, features, 1, 1, 0)

    def forward(self, low, high):
        emerge = low + high
        emerge = self.gap(emerge)

        fea_z = self.fc(emerge)

        high_att = self.fcs[0](fea_z)
        low_att = self.fcs[1](fea_z)

        attention_vectors = torch.cat([high_att, low_att], dim=1)
        attention_vectors = self.sc(attention_vectors)
        # attention_vectors = self.softmax(attention_vectors)
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)

        fea_high = high * high_att
        fea_low = low * low_att
        out = fea_high + fea_low
        out = self.out(out)
        return out


class FC1SENAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.sea = SFconv(c)
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True,
                                                 act_method=nn.GELU)  # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(x)

        x_spatial = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x_fourier = self.fft_block1(x_)
        # x_all = x_spatial + x_fourier
        x = self.sea(x_spatial, x_fourier)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FC2NAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        # self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU) # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        # x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.conv5(self.sg(self.conv4(x))) + self.fft_block2(x)

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FOCNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True,
                                                 act_method=nn.GELU)  # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        # x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.conv5(self.sg(self.conv4(x)))
        x = self.dropout2(x)
        x = y + x * self.gamma + self.fft_block1(inp)
        return x


class FLC1NAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1, kernel_size=19):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_large_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU,
                                                       kernel_size=kernel_size)  # , act_method=nn.GELU
        # self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        # x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta + self.fft_block1(x_)

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.conv5(self.sg(self.conv4(x)))
        x = self.dropout2(x)
        x = y + x * self.gamma
        return x


class FCANAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_conv_multiact(c, DW_Expand, num_heads=1, bias=True,
                                                  act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_conv_multiact(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FUNAFBlock(nn.Module):
    def __init__(self, c, scale_up=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv_with_scale_up(in_dim=c, dim=c // scale_up, dw=DW_Expand, bias=True,
                                                               act_method=nn.GELU,
                                                               scale_up=scale_up)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv_with_scale_up(in_dim=c, dim=c // scale_up, dw=DW_Expand, bias=True,
                                                               act_method=nn.GELU, scale_up=scale_up)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FDCANAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv_dycalayer(c, DW_Expand, num_heads=1, bias=True,
                                                           act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        # y, ca = self.fft_block1(x_)
        # x = x * (self.sca(x)+ca)
        y = self.fft_block1(x_)
        x = x * self.sca(x)
        x = self.conv3(x)
        # if self.sin:
        #     x = x + torch.sin(self.fft_block1(x_))
        # else:
        x = x + y
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        # if self.sin:
        #     x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x))) # * self.gamma
        # else:
        x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FDCNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_conv2(c, DW_Expand, num_heads=num_heads, bias=True,
                                                  act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv2(c, DW_Expand, num_heads=num_heads, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FourNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0: multihead_fft_bench_complex_conv fft_bench_complex_conv multihead_fft_bench_real_complex_dconv
        self.fft_block1 = multihead_fft_bench_real_complex_resconv(c, DW_Expand, num_heads=num_heads, bias=True,
                                                                   act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=num_heads, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FourierNAFBlock3X3(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0: multihead_fft_bench_complex_conv fft_bench_complex_conv multihead_fft_bench_real_complex_dconv
        self.fft_block1 = multihead_fft_bench_real_complex_3x3conv(c, DW_Expand, num_heads=num_heads, bias=True,
                                                                   act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=num_heads, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FDepthCNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_depthconv(c, DW_Expand, num_heads=num_heads, bias=True,
                                                      act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_conv(c, DW_Expand, num_heads=num_heads, bias=True, act_method=nn.GELU)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FPNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = Fourier_phase_DConvblock(c, dw_channel, kernel_size=sub_idx,
                                                   bias=True)  # , act_method=nn.GELU
        self.fft_block2 = Fourier_phase_DConvblock(c, dw_channel, kernel_size=sub_idx, bias=True)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FPTNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = Fourier_phaseT_DConvblock(c, dw_channel, kernel_size=sub_idx,
                                                    bias=True)  # , act_method=nn.GELU
        self.fft_block2 = Fourier_phaseT_DConvblock(c, dw_channel, kernel_size=sub_idx, bias=True)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class FPTv2NAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = Fourier_phaseTv2_DConvblock(c, dw_channel, num_heads=num_heads, kernel_size=sub_idx,
                                                      bias=True)  # , act_method=nn.GELU
        self.fft_block2 = Fourier_phaseTv2_DConvblock(c, dw_channel, num_heads=num_heads, kernel_size=sub_idx,
                                                      bias=True)
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class Fourier_phase_DConvblock(nn.Module):
    def __init__(self, dim, hdim, kernel_size, bias=True, norm='backward'):
        super().__init__()
        self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        if kernel_size > 1:
            self.conv_phase = nn.Sequential(*[
                nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
                nn.ZeroPad2d((0, kernel_size - 1, 0, kernel_size - 1)),
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=kernel_size, padding=0, stride=1, groups=dim,
                          bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            ])
            self.conv_mag = nn.Sequential(*[
                nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
                nn.ZeroPad2d((0, kernel_size - 1, 0, kernel_size - 1)),
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=kernel_size, padding=0, stride=1, groups=dim,
                          bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            ])
        else:
            self.conv_phase = nn.Sequential(*[
                nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            ])
            self.conv_mag = nn.Sequential(*[
                nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            ])

        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        # self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        y_mag = torch.abs(y)
        y_phase = torch.angle(y)
        # y_phase = y_phase / self.pi
        y_phase = self.conv_phase(y_phase)

        y = self.conv_mag(y_mag) * torch.exp(-1j * y_phase)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class Fourier_phaseT_DConvblock(nn.Module):
    def __init__(self, dim, hdim, kernel_size, bias=True, norm='backward'):
        super().__init__()
        self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        if kernel_size > 1:
            self.conv_phase = nn.Sequential(*[
                nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
                nn.ZeroPad2d((0, kernel_size - 1, 0, kernel_size - 1)),
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=kernel_size, padding=0, stride=1, groups=dim,
                          bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            ])

        else:
            self.conv_phase = nn.Sequential(*[
                nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
            ])

        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        # self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        y_mag = torch.abs(y)
        y_phase = torch.angle(y)

        # y_phase = y_phase / self.pi
        y_mag, y_phase = self.conv_phase(torch.cat([y_mag, y_phase], dim=0)).chunk(2, dim=0)

        y = y_mag * torch.exp(-1j * y_phase)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class Fourier_phaseTv2_DConvblock(nn.Module):
    def __init__(self, dim, hdim, kernel_size, num_heads, bias=True, norm='backward'):
        super().__init__()
        self.pi = 3.141592653589793
        # self.act_mag = nn.ReLU(inplace=True)
        self.proj_in = nn.Sequential(
            *[nn.Conv2d(in_channels=dim, out_channels=hdim, kernel_size=1, padding=0, stride=1, groups=1, bias=bias),
              nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=3,
                        padding=1, stride=1, groups=dim, bias=bias),
              ])
        self.proj_out = nn.Conv2d(in_channels=hdim, out_channels=dim, kernel_size=1, padding=0, stride=1, groups=1,
                                  bias=bias)
        if kernel_size > 1:
            self.conv_phase = nn.Sequential(*[
                nn.ReflectionPad2d((0, kernel_size - 1, 0, kernel_size - 1)),
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=kernel_size, padding=0, stride=1,
                          groups=num_heads,
                          bias=bias),
                nn.GELU(),
                nn.ReflectionPad2d((0, kernel_size - 1, 0, kernel_size - 1)),
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=kernel_size, padding=0, stride=1,
                          groups=num_heads,
                          bias=bias),
            ])

        else:
            self.conv_phase = nn.Sequential(*[
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=1, padding=0, stride=1,
                          groups=num_heads, bias=bias),
                nn.GELU(),
                nn.Conv2d(in_channels=hdim, out_channels=hdim, kernel_size=1, padding=0, stride=1,
                          groups=num_heads, bias=bias),
            ])

        # Sin_ACT(alpha=0.5, beta=self.pi), nn.ReLU(inplace=True),
        self.norm = norm
        # self.window_size = window_size

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.proj_in(x)
        y = torch.fft.rfft2(x, norm=self.norm)
        # dim = 1
        y_mag = torch.abs(y)
        y_phase = torch.angle(y)

        # y_mag, y_phase = self.conv_phase(torch.cat([y_mag, y_phase], dim=0)).chunk(2, dim=0)
        y_phase = self.conv_phase(y_phase) + y_phase
        y = y_mag * torch.exp(-1j * y_phase)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.proj_out(y)


class DCTNAFBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=0):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        if sub_idx == 1:
            self.fft_block1 = nn.Sequential(DCT2x(),
                                            LayerNorm2d(c),
                                            nn.Conv2d(c, dw_channel, kernel_size=1, padding=0),
                                            nn.GELU(),
                                            nn.Conv2d(dw_channel, c, kernel_size=1, padding=0),
                                            IDCT2x())
            self.fft_block2 = nn.Sequential(DCT2x(),
                                            nn.Conv2d(c, dw_channel, kernel_size=1, padding=0),
                                            nn.GELU(),
                                            nn.Conv2d(dw_channel, c, kernel_size=1, padding=0),
                                            IDCT2x())
        else:
            self.fft_block1 = nn.Sequential(DCT2x(),
                                            LayerNorm2d(c),
                                            nn.Conv2d(c, dw_channel, kernel_size=1, padding=0),
                                            nn.ZeroPad2d((0, sub_idx - 1, 0, sub_idx - 1)),
                                            nn.Conv2d(dw_channel, dw_channel, kernel_size=sub_idx, padding=0, groups=c),
                                            nn.GELU(),
                                            nn.ZeroPad2d((0, sub_idx - 1, 0, sub_idx - 1)),
                                            nn.Conv2d(dw_channel, dw_channel, kernel_size=sub_idx, padding=0, groups=c),
                                            nn.Conv2d(dw_channel, c, kernel_size=1, padding=0),
                                            IDCT2x())
            self.fft_block2 = nn.Sequential(DCT2x(),
                                            nn.Conv2d(c, dw_channel, kernel_size=1, padding=0),
                                            nn.GELU(),
                                            nn.Conv2d(dw_channel, c, kernel_size=1, padding=0),
                                            IDCT2x())
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        # self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        # self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout1 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout2d(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp

        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        _, _, H, W = x.shape
        # print(x.shape)

        # x = x + self.fft_block(x_)
        # print(F.adaptive_avg_pool2d(x, [1, 1]).shape)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)
        y = inp + x * self.beta

        x = self.norm2(y)
        if self.sin:
            x = torch.sin(self.fft_block2(x)) + self.conv5(self.sg(self.conv4(x)))  # * self.gamma
        else:
            x = self.fft_block2(x) + self.conv5(self.sg(self.conv4(x)))

        x = self.dropout2(x)
        x = y + x * self.gamma  # + self.fft_block(inp)
        return x


class DctFuseBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, sub_idx=1):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        if sub_idx == 1:
            self.fft_block1 = nn.Sequential(DCT2x(),
                                            LayerNorm2d(c),
                                            nn.Conv2d(c, dw_channel, kernel_size=1, padding=0),
                                            nn.GELU(),
                                            nn.Conv2d(dw_channel, c, kernel_size=1, padding=0),
                                            IDCT2x())
        else:
            self.fft_block1 = nn.Sequential(DCT2x(),
                                            LayerNorm2d(c),
                                            nn.Conv2d(c, dw_channel, kernel_size=1, padding=0),
                                            nn.ZeroPad2d((0, sub_idx - 1, 0, sub_idx - 1)),
                                            nn.Conv2d(dw_channel, dw_channel, kernel_size=sub_idx, padding=0, groups=c),
                                            nn.GELU(),
                                            nn.ZeroPad2d((0, sub_idx - 1, 0, sub_idx - 1)),
                                            nn.Conv2d(dw_channel, dw_channel, kernel_size=sub_idx, padding=0, groups=c),
                                            nn.Conv2d(dw_channel, c, kernel_size=1, padding=0),
                                            IDCT2x())

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)

        return x * self.beta


class Fusext_dct(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1.33, bias=True, LayerNorm_type='WithBias', dim_head=32, sub_idx=1):
        super(Fusext_dct, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.norm1 = LayerNorm2d(dim * 2)
        self.conv_spatial = DctFuseBlock(dim * 2, sub_idx=sub_idx)
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros((1, dim*2, 1, 1)), requires_grad=True)
        # self.fuse = SKFF(dim, height=2, reduction=4, bias=True)

    def forward(self, left_down, right_up):  # left_down - right_up?
        x_cat = torch.cat([left_down, right_up], dim=1)
        x_in = self.conv1(x_cat)
        x = self.conv_spatial(x_in) + x_in
        x = self.conv2(x)
        e, d = torch.chunk(x, 2, dim=1)
        output = e + d
        return output


class DFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, dim_out=None):
        super(DFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        if dim_out is None:
            dim_out = dim
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.fft = nn.Parameter(torch.ones((hidden_features * 2, 1, 1, self.patch_size, self.patch_size // 2 + 1)))
        self.project_out = nn.Conv2d(hidden_features, dim_out, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class DCTFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias, dim_out=None):
        super(DCTFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8
        if dim_out is None:
            dim_out = dim
        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1,
                                groups=hidden_features * 2, bias=bias)

        self.dct_mix = nn.Parameter(torch.ones((1, hidden_features * 2, 1, 1, self.patch_size, self.patch_size)))
        self.project_out = nn.Conv2d(hidden_features, dim_out, kernel_size=1, bias=bias)
        self.dct = DCT2(window_size=self.patch_size)

        self.idct = IDCT2(window_size=self.patch_size)

    def forward(self, x):
        x = self.project_in(x)
        x_patch = rearrange(x, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        x_patch_dct = self.dct(x_patch)
        x_patch_dct = x_patch_dct * self.dct_mix

        x_patch = self.idct(x_patch_dct)
        x = rearrange(x_patch, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                      patch2=self.patch_size)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)

        x = F.gelu(x1) * x2
        x = self.project_out(x)
        return x


class AttnBlock(nn.Module):
    def __init__(self, dim, num_heads=1, ffn_expansion_factor=2.66, bias=True, cs='channel_mlp'):
        super(AttnBlock, self).__init__()

        self.dct = DCT2x()
        self.idct = IDCT2x()
        self.norm1 = LayerNorm2d(dim)
        self.norm2 = LayerNorm2d(dim)
        self.normz = LayerNorm2d(dim)
        self.attn = AttentionX(dim, num_heads=num_heads, bias=True, cs=cs)
        self.conv_spatial = DFFN(dim, ffn_expansion_factor, bias)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.fft_block1 = fft_bench_complex_mlp(dim, ffn_expansion_factor, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_mlp(dim, ffn_expansion_factor, window_size=None, bias=True,
                                                act_method=nn.GELU)
        # self.fuse = SKFF(dim, height=2, reduction=4, bias=True)

    def forward(self, x_in):  # left_down - right_up?

        x = self.norm1(self.dct(x_in))

        x = self.idct(self.attn(x)) + self.fft_block1(self.normz(x_in))
        x = x * self.gamma + x_in

        y = self.norm2(x)
        y = self.conv_spatial(y) + self.fft_block2(y)
        x = y * self.beta + x
        return x


class Fusext(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=2.66, bias=True, LayerNorm_type='WithBias', dim_head=32):
        super(Fusext, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim, 1, 1, 0)
        # self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.norm1 = LayerNorm2d(dim * 2)
        self.norm2 = LayerNorm2d(dim)
        self.attn = AttentionX(dim * 2, num_heads=dim // dim_head, bias=True, cs='channel', out_dim=dim)
        self.conv_spatial = DFFN(dim, ffn_expansion_factor, bias)
        self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.fuse = SKFF(dim, height=2, reduction=4, bias=True)

    def forward(self, left_down, right_up):  # left_down - right_up?
        x_cat = torch.cat([left_down, right_up], dim=1)
        x_in = self.conv1(x_cat)
        x = self.norm1(x_in)
        x = self.attn(x) * self.gamma + self.conv2(x_cat)

        x = self.conv_spatial(self.norm2(x)) * self.beta + x

        return x


class Fusextv2(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1.33, bias=True, LayerNorm_type='WithBias', dim_head=32):
        super(Fusextv2, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.norm1 = LayerNorm2d(dim * 2)
        self.conv_spatial = DFFN(dim * 2, ffn_expansion_factor, bias)
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((1, dim * 2, 1, 1)), requires_grad=True)
        # self.fuse = SKFF(dim, height=2, reduction=4, bias=True)

    def forward(self, left_down, right_up):  # left_down - right_up?
        x_cat = torch.cat([left_down, right_up], dim=1)
        x_in = self.conv1(x_cat)
        x = self.norm1(x_in)
        x = self.conv_spatial(x) * self.beta + x_in
        x = self.conv2(x)
        e, d = torch.chunk(x, 2, dim=1)
        output = e + d
        return output


class FuseBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0:
        self.fft_block1 = fft_bench_complex_mlp(c, DW_Expand, window_size=None, bias=True,
                                                act_method=nn.GELU)  # , act_method=nn.GELU

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(F.adaptive_avg_pool2d(x, [1, 1]))

        x = self.conv3(x)
        if self.sin:
            x = x + torch.sin(self.fft_block1(x_))
        else:
            x = x + self.fft_block1(x_)
        x = self.dropout1(x)

        return x * self.beta


class Fusextv3(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1.33, bias=True, LayerNorm_type='WithBias', dim_head=32):
        super(Fusextv3, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.norm1 = LayerNorm2d(dim * 2)
        self.conv_spatial = FuseBlock(dim * 2)
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros((1, dim*2, 1, 1)), requires_grad=True)
        # self.fuse = SKFF(dim, height=2, reduction=4, bias=True)

    def forward(self, left_down, right_up):  # left_down - right_up?
        x_cat = torch.cat([left_down, right_up], dim=1)
        x_in = self.conv1(x_cat)
        x = self.conv_spatial(x_in) + x_in
        x = self.conv2(x)
        e, d = torch.chunk(x, 2, dim=1)
        output = e + d
        return output


class FuseXBlock(nn.Module):
    def __init__(self, c, num_heads=1, window_size=8, window_size_fft=-1, shift_size=-1, DW_Expand=2, FFN_Expand=2,
                 drop_out_rate=0., sin=False, attn_type=None, fft_bench=True):
        super().__init__()
        dw_channel = c * DW_Expand
        self.sin = sin
        # print(sin)
        self.window_size = window_size  # window_size
        self.window_size_fft = window_size_fft
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        # self.attn = Attention_win(dw_channel, num_heads=num_heads)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )
        self.norm1 = LayerNorm2d(c)
        # SimpleGate
        self.sg = SimpleGate()
        # self.sg = SimpleGate_frelu()
        # if window_size_fft is None or window_size_fft >= 0: fft_bench_complex_conv
        if fft_bench:
            self.fft_block1 = fft_bench_complex_conv(c, DW_Expand, num_heads=1, bias=True,
                                                     act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_bench = fft_bench
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

    def forward(self, inp):
        x = inp
        x_ = self.norm1(x)
        x = self.conv1(x_)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)

        x = self.conv3(x)
        if self.fft_bench:
            if self.sin:
                x = x + torch.sin(self.fft_block1(x_))
            else:
                x = x + self.fft_block1(x_)
        x = self.dropout1(x)

        return x * self.beta


class FusNext(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=1.33, bias=True, LayerNorm_type='WithBias', dim_head=32,
                 fft_bench=True):
        super(FusNext, self).__init__()
        self.conv1 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.conv2 = nn.Conv2d(dim * 2, dim * 2, 1, 1, 0)
        # self.norm1 = LayerNorm2d(dim * 2)
        self.conv_spatial = FuseXBlock(dim * 2, fft_bench=fft_bench)
        # self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        # self.beta = nn.Parameter(torch.zeros((1, dim*2, 1, 1)), requires_grad=True)
        # self.fuse = SKFF(dim, height=2, reduction=4, bias=True)

    def forward(self, left_down, right_up):  # left_down - right_up?
        # print(left_down.shape, right_up.shape)
        x_cat = torch.cat([left_down, right_up], dim=1)
        x_in = self.conv1(x_cat)
        x = self.conv_spatial(x_in) + x_in
        x = self.conv2(x)
        e, d = torch.chunk(x, 2, dim=1)
        output = e + d
        return output


class FSAS(nn.Module):
    def __init__(self, dim, bias):
        super(FSAS, self).__init__()

        self.to_hidden = nn.Conv2d(dim, dim * 6, kernel_size=1, bias=bias)
        self.to_hidden_dw = nn.Conv2d(dim * 6, dim * 6, kernel_size=3, stride=1, padding=1, groups=dim * 6, bias=bias)

        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)

        self.norm = LayerNorm2d(dim * 2)

        self.patch_size = 8

    def forward(self, x):
        hidden = self.to_hidden(x)

        q, k, v = self.to_hidden_dw(hidden).chunk(3, dim=1)

        q_patch = rearrange(q, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        k_patch = rearrange(k, 'b c (h patch1) (w patch2) -> b c h w patch1 patch2', patch1=self.patch_size,
                            patch2=self.patch_size)
        q_fft = torch.fft.rfft2(q_patch.float())
        k_fft = torch.fft.rfft2(k_patch.float())

        out = q_fft * k_fft
        out = torch.fft.irfft2(out, s=(self.patch_size, self.patch_size))
        out = rearrange(out, 'b c h w patch1 patch2 -> b c (h patch1) (w patch2)', patch1=self.patch_size,
                        patch2=self.patch_size)

        out = self.norm(out)

        output = v * out
        output = self.project_out(output)

        return output


class Ffftformer(nn.Module):
    def __init__(self, dim, ffn_expansion_factor=3, bias=True, att=True):
        super(Ffftformer, self).__init__()

        self.att = att
        if self.att:
            self.norm1 = LayerNorm2d(dim)
            self.attn = FSAS(dim, bias)
            self.gamma = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)

            self.fft_block1 = fft_bench_complex_linear(dim, 1, window_size=None, bias=True,
                                                    act_method=nn.GELU)  # , act_method=nn.GELU
        self.fft_block2 = fft_bench_complex_linear(dim, 1, window_size=None, bias=True,
                                                act_method=nn.GELU)
        self.beta = nn.Parameter(torch.zeros((1, dim, 1, 1)), requires_grad=True)
        self.norm2 = LayerNorm2d(dim)
        self.ffn = DFFN(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        if self.att:
            z = self.norm1(x)
            x = x + self.gamma * (self.attn(z) + self.fft_block1(z))
        y = self.norm2(x)
        x = x + self.beta * (self.ffn(y) + self.fft_block2(y))

        return x


class AttentionX(nn.Module):
    def __init__(self, dim, num_heads, bias, window_size=8, grid_size=8, window_size_dct=9,
                 qk_norm=True, proj_out=True, temp_div=True, norm_dim=-1, cs='channel', padding_mode='zeros',
                 out_dim=None):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.dim = dim // num_heads
        self.out_dim = out_dim // num_heads
        self.qk_norm = qk_norm
        self.num_heads = num_heads
        self.norm_dim = norm_dim  # -2
        self.window_size = window_size
        self.window_size_dct = window_size_dct
        self.grid_size = grid_size
        self.cs = cs
        # print(self.qk_norm)
        self.add = True if 'mlp_add' in self.cs else False
        self.channel_mlp = True if 'clp' in self.cs else False
        self.block_mlp = True if 'mlp' in self.cs else False
        self.coarse_mlp = True if 'coarse' in self.cs else False
        self.block_graph = True if 'graph' in self.cs else False
        self.global_attn = True if 'global' in self.cs else False
        if not self.global_attn:
            if 'grid' in self.cs:
                N = grid_size ** 2
                self.k = grid_size
            else:
                N = window_size ** 2
                self.k = window_size
        if self.coarse_mlp:
            self.mlp_coarse = CoarseMLP(dim=1, window_size_dct=window_size_dct, num_heads=1, bias=bias)
        if self.block_mlp:
            self.mlp = nn.Sequential(
                nn.Linear(N, N, bias=True),
                nn.GELU(),
            )
        if self.channel_mlp:
            self.cmlp = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=1, bias=True),
                nn.GELU(),
            )
        # elif self.block_graph:
        #     self.graph = Grapher(dim, window_size=self.k)
        self.qkv = nn.Conv2d(dim, dim * 2 + out_dim, kernel_size=1, bias=bias)

        self.qkv_dwconv = nn.Conv2d(dim * 2 + out_dim, dim * 2 + out_dim, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 2 + out_dim, bias=bias, padding_mode=padding_mode)

        if temp_div:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1) / math.sqrt(out_dim))
        else:
            self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        if proj_out:
            self.project_out = nn.Conv2d(out_dim, out_dim, kernel_size=1, bias=True)
        else:
            self.project_out = nn.Identity()

    def get_attn(self, qkv):
        H, W = qkv.shape[-2:]
        # if self.window_size is not None:
        #     qkv, batch_list = self.winp(qkv)
        qkv = check_image_size(qkv, self.window_size)
        Hx, Wx = qkv.shape[-2:]
        if 'grid' in self.cs:
            qkv = rearrange(qkv, 'b (head c) (h h1) (w w1) -> (b h1 w1) head c (h w)', head=self.num_heads,
                            h=self.grid_size, w=self.grid_size)
        else:
            qkv = rearrange(qkv, 'b (head c) (h1 h) (w1 w) -> (b h1 w1) head c (h w)', head=self.num_heads,
                            h=self.window_size, w=self.window_size)
        # q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        # v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv.split([self.out_dim, self.dim, self.dim], dim=2)

        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)

        attn = (q @ k.transpose(-2, -1)) * self.temperature

        attn = attn.softmax(dim=-1)
        out = (attn @ v)
        if self.block_mlp:
            if self.add:
                out = out + self.mlp(v)
            else:
                out = out * self.mlp(v)

        if 'grid' in self.cs:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h h1) (w w1)', head=self.num_heads,
                            h1=Hx // self.grid_size,
                            w1=Wx // self.grid_size, h=self.grid_size, w=self.grid_size)
        else:
            out = rearrange(out, '(b h1 w1) head c (h w) -> b (head c) (h1 h) (w1 w)', head=self.num_heads,
                            h1=Hx // self.window_size,
                            w1=Wx // self.window_size, h=self.window_size, w=self.window_size)

        return out[:, :, :H, :W]

    def get_attn_global(self, qkv):
        H, W = qkv.shape[-2:]
        qkv = rearrange(qkv, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q, k, v = qkv.split([self.out_dim, self.dim, self.dim], dim=2)
        if self.qk_norm:
            q = torch.nn.functional.normalize(q, dim=self.norm_dim)
            k = torch.nn.functional.normalize(k, dim=self.norm_dim)
        if 'spatial' in self.cs:
            attn = (q.transpose(-2, -1) @ k) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v.transpose(-2, -1))  # .contiguous())
            # print(attn.shape, out.shape)
            out = out.transpose(-2, -1)
        else:
            attn = (q @ k.transpose(-2, -1)) * self.temperature

            attn = attn.softmax(dim=-1)
            out = (attn @ v)
        if self.block_mlp:
            out = out * self.mlp(v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=H, w=W)
        return out

    def forward(self, x):

        qkv = self.qkv_dwconv(self.qkv(x))
        # _, _, H, W = qkv.shape
        if not self.global_attn:
            out = self.get_attn(qkv)
        else:
            out = self.get_attn_global(qkv)
        out = self.project_out(out)
        return out

    def flops(self, inp_shape):
        C, H, W = inp_shape
        flops = 0
        # fc1
        flops += H * W * C * C * 3
        # dwconv
        flops += H * W * (C * 3) * 3 * 3
        # attn
        c_attn = C // self.num_heads
        if 'spatial' in self.cs:
            flops += self.num_heads * 2 * (c_attn * H * W * (self.window_size ** 2))
        else:
            flops += self.num_heads * 2 * ((c_attn ** 2) * H * W)
        if self.channel_mlp:
            flops += H * W * C * C
        if self.block_mlp:
            flops += H * W * C * (self.window_size ** 2)
        # fc2
        flops += H * W * C * C
        # print("Attn:{%.2f}" % (flops / 1e9))
        return flops


##########################################################################
##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=2, reduction=8, bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels / reduction), 4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1, bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, left, right):
        batch_size = left.shape[0]
        n_feats = left.shape[1]

        inp_feats = torch.cat([left, right], dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats * attention_vectors, dim=1)

        return feats_V


class fft_bench_complex_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False):
        super(fft_bench_complex_conv, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y

class fft_bench_concat_channl_conv(nn.Module):
    def __init__(self, dim, dw=1, window_size=None, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(fft_bench_concat_channl_conv, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)
        self.window_size = window_size
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            y = window_reversex(y, self.window_size, H, W, batch_list)
        return y
class fft_bench_concat_channl_dwconv(nn.Module):
    def __init__(self, dim, dw=1, window_size=None, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(fft_bench_concat_channl_dwconv, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_dwconv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=3, padding=1, groups=hid_dim * 2, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)
        self.window_size = window_size
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_dwconv1(self.complex_conv1(torch.cat([y.real, y.imag], dim=1)))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            y = window_reversex(y, self.window_size, H, W, batch_list)
        return y
class fft_bench_concat_batch_conv(nn.Module):
    def __init__(self, dim, dw=1, window_size=None, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(fft_bench_concat_batch_conv, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim, hid_dim, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, dim, kernel_size=1, bias=bias)
        self.window_size = window_size
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=0))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=0)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)


        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            y = torch.fft.irfft2(y, s=self.window_size, norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y
class fft_bench_concat_split_conv(nn.Module):
    def __init__(self, dim, dw=1, window_size=None, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(fft_bench_concat_split_conv, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias, groups=2)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias, groups=2)
        self.window_size = window_size
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        b, c, H, W = x.shape

        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            x, batch_list = window_partitionx(x, self.window_size)
            # print(x.shape, batch_list)
        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)


        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            y = torch.fft.irfft2(y, s=(self.window_size[0], self.window_size[1]), norm=self.norm)
            y = window_reversex(y, self.window_size, H, W, batch_list)
        else:
            y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        # print(y.shape, self.window_size, batch_list, b,c,H, W)
        return y


class fft_bench_concat_split_dwconv(nn.Module):
    def __init__(self, dim, dw=1, window_size=None, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(fft_bench_concat_split_dwconv, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias, groups=2)
        self.complex_dwconv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=3, padding=1, groups=hid_dim * 2, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias, groups=2)
        self.window_size = window_size
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape
        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            x, batch_list = window_partitionx(x, self.window_size)
        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_dwconv1(self.complex_conv1(torch.cat([y.real, y.imag], dim=1)))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        if self.window_size is not None and (H != self.window_size[0] or W != self.window_size[1]):
            y = window_reversex(y, self.window_size, H, W, batch_list)
        return y
class fft_bench_complex_conv_phase_tig(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False, train_size=[256, 256]):
        super(fft_bench_complex_conv_phase_tig, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)
        self.phase_para = nn.Parameter(torch.zeros(1, dim, train_size[0], train_size[1]))
        torch.nn.init.kaiming_uniform_(self.phase_para, a=math.sqrt(2))
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        # phase_proj = torch.exp(1j*self.phase_para)
        phase_proj = kornia.geometry.subpix.spatial_softmax2d(self.phase_para)
        phase_proj = torch.fft.rfft2(phase_proj, norm=self.norm)
        # phase_proj = phase_proj / (torch.abs(phase_proj)+1e-6)
        # print(y.shape, phase_proj.shape)
        y = y * phase_proj
        # print(torch.mean(torch.abs(self.phase_para)))
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class fft_bench_complex_large_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False, kernel_size=19):
        super(fft_bench_complex_large_conv, self).__init__()
        self.act_fft = act_method()
        # self.act_spatial = nn.GELU()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim, hid_dim, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, dim, kernel_size=1, bias=bias)
        # self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.bias = bias
        self.norm = norm
        # init_kers = torch.zeros([1, hid_dim, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        # self.kers = nn.Parameter(init_kers, requires_grad=True)
        # torch.nn.init.kaiming_uniform_(self.kers, a=math.sqrt(5))
        self.kernel_size = kernel_size
        self.pad = nn.ReflectionPad2d(kernel_size // 2)
        self.ker_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=hid_dim, out_channels=num_heads * kernel_size ** 2, kernel_size=1, padding=0,
                      stride=1,
                      groups=1, bias=True),
            nn.BatchNorm2d(num_heads * kernel_size ** 2),
            # nn.Softmax(1)
        )
        self.num_heads = num_heads
        # nn.init.kaiming_normal(self.kers)
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        y = self.complex_conv1(x)
        batch_size = y.shape[0]
        ker_x = self.ker_gen(y).view(batch_size, self.num_heads, self.kernel_size, self.kernel_size)
        ker_x = kornia.geometry.subpix.spatial_softmax2d(ker_x)
        y = self.pad(y)
        # print(y.shape)
        H, W = y.shape[-2:]
        y = rearrange(y, 'b (nh c1) h w -> b nh c1 h w', nh=self.num_heads)

        kers = convert_psf2otf(ker_x, (batch_size, self.num_heads, H, W))
        kers = kers.unsqueeze(2)
        y = torch.fft.rfft2(y, norm=self.norm)
        # y = y * kers
        y = y * torch.conj(kers) / (torch.abs(kers) + 1e-5)
        y = rearrange(y, 'b nh c1 h w -> b (nh c1) h w')
        y = torch.cat([y.real, y.imag], dim=1)
        # print(self.kers.shape, (1, y.shape[1]//2, H, W))

        # kers = torch.cat([kers.real, kers.imag], dim=1)
        # # y1, y2 = y.chunk(2, dim=1)
        # # ker1, ker2 = kers.chunk(2, dim=1)
        # # kers_mag = torch.abs(kers)
        # y = y * kers
        y = self.act_fft(y)

        y_real, y_imag = y.chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # ker_x = self.ker_norm(self.kers)
        # ker_x = kornia.geometry.subpix.spatial_softmax2d(ker_x)

        # y2 = y * ker2  # / (kers_mag ** 2 + 1e-3)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        # y = torch.concat([y, y1, y2], dim=1)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        # print(y.shape)
        y = y[:, :, self.kernel_size // 2 - 1:-self.kernel_size // 2,
            self.kernel_size // 2 - 1:-self.kernel_size // 2].contiguous()
        # print(y.shape, self.kernel_size//2)
        y = self.complex_conv2(y)

        # y, y1, y2 = y.chunk(3, dim=1)
        # y1 = torch.softmax(y1, dim=1)
        # y = y * torch.sum(y1 * y2, dim=1, keepdim=True)
        # y = self.act_spatial(y)
        return y


class fft_bench_complex_large_conv1(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False, kernel_size=19):
        super(fft_bench_complex_large_conv1, self).__init__()
        self.act_fft = act_method()
        # self.act_spatial = nn.GELU()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)
        self.project_out = nn.Conv2d(dim * 2, dim, kernel_size=1, bias=bias)
        self.bias = bias
        self.norm = norm
        init_kers = torch.zeros([1, dim, kernel_size, kernel_size])
        # init_kers[:, :, kernel_size // 2, kernel_size // 2] = 1.
        self.kers = nn.Parameter(init_kers, requires_grad=True)
        torch.nn.init.kaiming_uniform_(self.kers, a=math.sqrt(5))
        self.ker_norm = LayerNorm2d(dim)
        # nn.init.kaiming_normal(self.kers)
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        # print(self.kers.shape, (1, y.shape[1]//2, H, W))

        y = self.act_fft(y)
        y = self.complex_conv2(y)
        y_real, y_imag = y.chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        ker_x = self.ker_norm(self.kers)
        ker_x = kornia.geometry.subpix.spatial_softmax2d(ker_x)
        kers = convert_psf2otf(ker_x, (1, y.shape[1], H, W))
        # kers = torch.cat([kers.real, kers.imag], dim=1)
        # y1, y2 = y.chunk(2, dim=1)
        # ker1, ker2 = kers.chunk(2, dim=1)
        # kers_mag = torch.abs(kers)
        y1 = y * kers
        y2 = y * torch.conj(kers)  # / (kers_mag ** 2 + 1e-3)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = torch.concat([y, y1, y2], dim=1)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y, y1, y2 = y.chunk(3, dim=1)
        y = y + y1 * y2
        # y = self.act_spatial(y)
        return self.project_out(y)


class fft_bench_conv_multiact(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False):
        super(fft_bench_conv_multiact, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y_real, y_imag = y.chunk(2, dim=1)
        y_real1, y_real2 = y_real.chunk(2, dim=1)
        y_real1 = self.act_fft(y_real1)
        y_real2 = -self.act_fft(-y_real2)
        y_imag1, y_imag2, y_imag3, y_imag4 = y_imag.chunk(4, dim=1)
        y_imag1 = self.act_fft(y_imag1)
        y_imag2 = -self.act_fft(-y_imag2)
        y_imag3 = -self.act_fft(-y_imag3)
        y_imag4 = self.act_fft(y_imag4)

        y_real = torch.cat([y_real1, y_real2], dim=1)
        y_imag = torch.cat([y_imag1, y_imag2, y_imag3, y_imag4], dim=1)
        y = torch.cat([y_real, y_imag], dim=1)
        # y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class fft_bench_deform_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False):
        super(fft_bench_deform_conv, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_deform_conv = DCNv3_pytorch(hid_dim * 2, kernel_size=1, group=num_heads,
                                                 center_feature_scale=False, remove_center=False)
        self.complex_conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.complex_deform_conv(y)
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class fft_bench_deform_conv2(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False):
        super(fft_bench_deform_conv2, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_deform_conv = DCNv3_pytorch(hid_dim * 2, kernel_size=1, group=num_heads,
                                                 center_feature_scale=False, remove_center=False)
        self.complex_conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))

        y = self.act_fft(y)
        y = self.complex_deform_conv(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class fft_bench_mlp(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False,
                 spatial_size=[128, 128]):
        super(fft_bench_mlp, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        h, w = spatial_size
        self.spatial_size = spatial_size
        # print(dim, hid_dim)
        self.complex_mlp = nn.Sequential(*[nn.Sequential(
            nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias),
            act_method(),
            nn.Linear(w // 2 + 1, w // 2 + 1),
            act_method(),
            Rearrange('b c h w -> b c w h'),
            nn.Linear(h, h),
            act_method(),
            Rearrange('b c w h -> b c h w'),
            nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)
        )])

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape
        if H != self.spatial_size[0] or W != self.spatial_size[1]:
            x = F.interpolate(x, size=self.spatial_size, mode='bilinear', align_corners=True)
        y = torch.fft.rfft2(x, norm=self.norm)
        y_real, y_imag = self.complex_mlp(torch.cat([y.real, y.imag], dim=1)).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=self.spatial_size, norm=self.norm)
        if H != self.spatial_size[0] or W != self.spatial_size[1]:
            y = F.interpolate(y, size=[H, W], mode='bilinear', align_corners=True)

        return y


class fft_bench_complex_conv_with_scale_up(nn.Module):
    def __init__(self, in_dim, dim, dw=1, norm='backward', act_method=nn.ReLU, scale_up=2, bias=False):
        super(fft_bench_complex_conv_with_scale_up, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        if scale_up > 1:
            self.conv_scale_up = nn.Sequential(nn.Conv2d(in_dim, dim * (scale_up ** 2), 1, bias=False),
                                               nn.PixelShuffle(scale_up)
                                               )
            self.conv_scale_down = nn.Sequential(nn.PixelUnshuffle(scale_up),
                                                 nn.Conv2d(dim * (scale_up ** 2), in_dim, 1, bias=False),
                                                 )
        else:
            self.conv_scale_up = nn.Identity()
            self.conv_scale_down = nn.Identity()
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):

        x = self.conv_scale_up(x)
        _, _, H, W = x.shape
        # print(x.shape)
        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        y = self.conv_scale_down(y)
        return y


class fft_bench_complex_conv_dycalayer(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.ReLU, num_heads=1, bias=False):
        super(fft_bench_complex_conv_dycalayer, self).__init__()
        self.act_fft = act_method()
        # self.window_size = window_size
        # dim = out_channel
        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)
        self.ca = DCNv3_pytorch_calayer(dim, kernel_size=3)  # kernel_size=3 5
        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.act_fft(y)

        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        ca = self.ca(torch.abs(y))
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y * ca


class fft_bench_complex_conv2(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(fft_bench_complex_conv2, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_dconv = nn.Conv2d(hid_dim * 2, hid_dim * 2, groups=num_heads * 2, kernel_size=3, padding=1,
                                       bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.complex_dconv(y)
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class multihead_fft_bench_complex_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_complex_conv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = nn.Conv2d(dim, hid_dim, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(hid_dim, dim, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape
        x = self.conv1(x)
        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=0))
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=0)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return self.conv2(y)


class multihead_fft_bench_real_complex_3x3conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_real_complex_3x3conv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        # self.complex_conv2 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_real, x_imag = self.conv1(x_fft).chunk(2, dim=1)
        x_fft = torch.cat([x_real, x_imag], dim=0)
        y = self.complex_conv1(x_fft)
        y_real, y_imag = y.chunk(2, dim=0)
        y_real, y_imag = self.conv2(self.act_fft(torch.cat([y_real, y_imag], dim=1))).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class multihead_fft_bench_real_complex_resconv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_real_complex_resconv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_real, x_imag = self.conv1(x_fft).chunk(2, dim=1)
        x_fft = torch.cat([x_real, x_imag], dim=0)
        y = self.complex_conv1(x_fft)
        y = self.act_fft(y)
        y = self.complex_conv2(y) + x_fft
        y_real, y_imag = y.chunk(2, dim=0)
        y_real, y_imag = self.conv2(self.act_fft(torch.cat([y_real, y_imag], dim=1))).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class multihead_fft_bench_real_complex_noresconv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_real_complex_noresconv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_real, x_imag = self.conv1(x_fft).chunk(2, dim=1)
        x_fft = torch.cat([x_real, x_imag], dim=0)
        y = self.complex_conv1(x_fft)
        y = self.act_fft(y)
        y = self.complex_conv2(y)  # + x_fft
        y_real, y_imag = y.chunk(2, dim=0)
        y_real, y_imag = self.conv2(self.act_fft(torch.cat([y_real, y_imag], dim=1))).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class multihead_fft_bench_real_complex_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_real_complex_conv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        # self.conv1 = nn.Conv2d(dim*2, hid_dim*2, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        # self.conv2 = nn.Conv2d(hid_dim*2, dim*2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=0)
        # x_real, x_imag = self.conv1(x_fft).chunk(2, dim=1)
        # x_fft = torch.cat([x_real, x_imag], dim=0)
        y = self.complex_conv1(x_fft)
        y = self.act_fft(y)
        y = self.complex_conv2(y)
        y_real, y_imag = y.chunk(2, dim=0)
        # y_real, y_imag = self.conv2(self.act_fft(torch.cat([y_real, y_imag], dim=1))).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class multihead_fft_bench_real_complex_dconv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_real_complex_dconv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(hid_dim, hid_dim, groups=hid_dim, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim, hid_dim, groups=hid_dim, kernel_size=3, padding=1, bias=bias)
        self.conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_real, x_imag = self.conv1(x_fft).chunk(2, dim=1)
        x_fft = torch.cat([x_real, x_imag], dim=0)
        y = self.complex_conv1(x_fft)
        y = self.act_fft(y)
        y = self.complex_conv2(y) + x_fft
        y_real, y_imag = y.chunk(2, dim=0)
        y_real, y_imag = self.conv2(self.act_fft(torch.cat([y_real, y_imag], dim=1))).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class multihead_fft_bench_resconv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fft_bench_resconv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_conv1 = nn.Conv2d(hid_dim * 2, hid_dim * 2, groups=hid_dim, kernel_size=1, padding=0, bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim * 2, hid_dim * 2, groups=hid_dim, kernel_size=1, padding=0, bias=bias)
        self.conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=1)
        x_fft = self.conv1(x_fft)
        # x_fft = torch.cat([x_real, x_imag], dim=0)
        y = self.complex_conv1(x_fft)
        y = self.act_fft(y)
        y = self.complex_conv2(y) + x_fft
        # y_real, y_imag = y.chunk(2, dim=0)
        y_real, y_imag = self.conv2(self.act_fft(y)).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class multihead_fourier_complex_conv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(multihead_fourier_complex_conv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = Complex_Conv(dim, hid_dim, kernel_size=1, bias=bias)
        # self.norm1 = nn.Identity() # LayerNorm2d(hid_dim)
        self.complex_conv1 = Complex_Conv(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.complex_conv2 = Complex_Conv(hid_dim, hid_dim, groups=num_heads, kernel_size=3, padding=1, bias=bias)
        self.conv2 = Complex_Conv(hid_dim, dim, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=0)
        x_fft = self.conv1(x_fft)
        y = self.complex_conv1(x_fft)
        y = self.act_fft(y)
        y = self.complex_conv2(y) + x_fft

        y_real, y_imag = self.conv2(self.act_fft(y)).chunk(2, dim=0)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class fourier_complex_conv1x1(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(fourier_complex_conv1x1, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.conv1 = Complex_Conv(dim, hid_dim, kernel_size=1, bias=bias)

        self.conv2 = Complex_Conv(hid_dim, dim, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        x_fft = torch.fft.rfft2(x, norm=self.norm)
        x_fft = torch.cat([x_fft.real, x_fft.imag], dim=0)
        x_fft = self.conv1(x_fft)
        y_real, y_imag = self.conv2(self.act_fft(x_fft)).chunk(2, dim=0)
        y = torch.complex(y_real, y_imag)
        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class fft_bench_complex_depthconv(nn.Module):
    def __init__(self, dim, dw=1, norm='backward', act_method=nn.GELU, num_heads=1, bias=False):
        super(fft_bench_complex_depthconv, self).__init__()
        self.act_fft = act_method()

        hid_dim = int(dim * dw)
        # print(dim, hid_dim)
        self.complex_conv1 = nn.Conv2d(dim * 2, hid_dim * 2, kernel_size=1, bias=bias)
        self.complex_dconv = nn.Conv2d(hid_dim * 2, hid_dim * 2, groups=hid_dim * 2, kernel_size=3, padding=1,
                                       bias=bias)
        self.complex_conv2 = nn.Conv2d(hid_dim * 2, dim * 2, kernel_size=1, bias=bias)

        self.bias = bias
        self.norm = norm
        # self.min = inf
        # self.max = -inf

    def forward(self, x):
        _, _, H, W = x.shape

        y = torch.fft.rfft2(x, norm=self.norm)
        y = self.complex_conv1(torch.cat([y.real, y.imag], dim=1))
        y = self.complex_dconv(y)
        y = self.act_fft(y)
        y_real, y_imag = self.complex_conv2(y).chunk(2, dim=1)
        y = torch.complex(y_real, y_imag)
        # y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)

        y = torch.fft.irfft2(y, s=(H, W), norm=self.norm)
        return y


class Complex_Conv(nn.Module):
    def __init__(self, dim, dim_out, groups=1, kernel_size=1, padding=0, bias=True):
        super(Complex_Conv, self).__init__()
        # padding = kernel_size // 2 - 1
        self.complex_conv = nn.Conv2d(dim, dim_out * 2, kernel_size=kernel_size, bias=bias, padding=padding,
                                      groups=groups)

    def forward(self, x):
        a, b = self.complex_conv(x).chunk(2, dim=1)
        a_real, a_imag = a.chunk(2, dim=0)
        b_imag, b_real = b.chunk(2, dim=0)

        return torch.cat([a_real - b_real, b_imag + a_imag], dim=0)

        # bs_blur_idx = torch.where(torch.sum(mask_blur, dim=[1, 2]) > 0)[0]
        # # print(bs_blur_idx)
        # mask_blur = mask_blur.ge(0.5)
        # mam_list = []
        # for i in bs_blur_idx:
        #     # if torch.sum(mask[i, 1, :, :]) == 0:
        #     #     continue
        #     # print(i, mask_blur.shape)
        #     mask_ = mask_blur[i, :, :]
        #     # mask_ = torch.index_select(mask_blur, dim=0, index=i)
        #     y_z = y[i, ...]
        #     y_zero = torch.zeros_like(y_z)
        #     z = torch.masked_select(y_z, mask_)
        #     z = z.view(1, ch, -1)
        #     # print(z.shape)
        #     z = z.permute(0, 2, 1)
        #     z = self.mamba(z)
        #     z = z.permute(0, 2, 1)
        #     y_z = y_zero.masked_scatter(mask_, z.flatten())
        #     mam_list.append(y_z.unsqueeze(0))
        # if len(mam_list) > 0:
        #     mamba_out = torch.cat(mam_list, dim=0)
        #     y = y.index_add_(0, bs_blur_idx, mamba_out, alpha=1)