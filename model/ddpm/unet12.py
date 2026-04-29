import torch
import torch.nn as nn
import torch.nn.functional as f
from mamba_ssm import Mamba
from torch import einsum
from torch.nn import Module
import pywt
from einops import rearrange,repeat
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

from model.ddpm.unit import *
from model.ddpm.RestorMixer import WindowAttention
import model.ddpm.RestorMixer as RestorMixer

class DWT_2D(Module):
    def __init__(self, wavename):
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            - self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)

class IDWT_2D(Module):
    def __init__(self, wavename):
        super(IDWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.dec_lo
        self.band_low.reverse()
        self.band_high = wavelet.dec_hi
        self.band_high.reverse()
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros((L, L1 + self.band_length - 2))
        matrix_g = np.zeros((L1 - L, L1 + self.band_length - 2))
        end = None if self.band_length_half == 1 else (
            - self.band_length_half + 1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index + j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index + j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(
            self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(
            self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:, (self.band_length_half - 1):end]
        matrix_h_1 = matrix_h_1[:, (self.band_length_half - 1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:, (self.band_length_half - 1):end]
        matrix_g_1 = matrix_g_1[:, (self.band_length_half - 1):end]
        matrix_g_1 = np.transpose(matrix_g_1)
        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, LL, LH, HL, HH):
        assert len(LL.size()) == len(LH.size()) == len(
            HL.size()) == len(HH.size()) == 4
        self.input_height = LL.size()[-2] + HH.size()[-2]
        self.input_width = LL.size()[-1] + HH.size()[-1]
        self.get_matrix()
        return IDWTFunction_2D.apply(LL, LH, HL, HH, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)



class TimeEmbedding(nn.Module):
    def __init__(self, dim,device):
        super().__init__()
        self.dim = dim
        self.timeemd = torch.exp(
            torch.arange(0, dim, 2, dtype=torch.float32,device=device) *
            (-math.log(10000) / dim)
        )

    def forward(self, input):
        shape = input.shape
        sinusoid_in = torch.ger(input.view(-1).float(), self.timeemd)
        pos_emb = torch.cat([sinusoid_in.sin(), sinusoid_in.cos()], dim=-1)
        pos_emb = pos_emb.view(*shape, self.dim)
        return pos_emb


class AttnBlock(nn.Module):
    def __init__(self, dim, skip_rescale=True, init_scale=0.):
        super().__init__()
        self.NIN_0 = NIN(dim, dim)
        self.NIN_1 = NIN(dim, dim)
        self.NIN_2 = NIN(dim, dim)
        self.NIN_3 = NIN(dim, dim, init_scale=init_scale)
        self.skip_rescale = skip_rescale

        self.qnorm=nn.LayerNorm(dim)
        self.knorm=nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        h = x
        q = self.NIN_0(h)
        k = self.NIN_1(h)
        v = self.NIN_2(h)

        # ----- 新增：对 q 和 k 进行层归一化 -----
        # 将通道维放到最后，对每个位置的特征向量做 LayerNorm
        q = q.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        k = k.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        q = self.qnorm(q)  # 归一化最后一个维度（C）
        k = self.knorm(k)
        q = q.permute(0, 3, 1, 2).contiguous()  # 恢复为 (B, C, H, W)
        k = k.permute(0, 3, 1, 2).contiguous()
        # ------------------------------------


        w = torch.einsum('bchw,bcij->bhwij', q, k) * (int(C) ** (-0.5))
        w = torch.reshape(w, (B, H, W, H * W))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, H, W))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        h = self.NIN_3(h)
        if not self.skip_rescale:
            return x + h
        else:
            return (x + h) / np.sqrt(2.)


class AttnBlockWithChannel(nn.Module):
    def __init__(self, dim):
        super(AttnBlockWithChannel,self).__init__()
        self.NIN_0 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3,1,1,groups=dim),
        )
        self.NIN_1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3,1,1,groups=dim),
        )
        self.NIN_2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3,1,1,groups=dim),
        )

    def forward(self, x):
        B, C, H, W = x.shape
        q = self.NIN_0(x)
        k = self.NIN_1(x)
        v = self.NIN_2(x)

        scale = (H * W) ** -0.5
        w = torch.einsum('bchw,bdhw->bcd', q, k) * scale
        w = F.softmax(w, dim=-1)
        h = torch.einsum('bcd,behw->bchw', w, v)
        return h

class SCConv(nn.Module):
    def __init__(self, inplanes, norm_layer, planes=None, stride=1, padding=1, dilation=1, groups=1, pooling_r=4):
        super(SCConv, self).__init__()
        if planes is None:
            planes = inplanes
        self.k2 = nn.Sequential(
                    nn.AvgPool2d(kernel_size=pooling_r, stride=pooling_r),
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k3 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=1,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )
        self.k4 = nn.Sequential(
                    nn.Conv2d(inplanes, planes, kernel_size=3, stride=stride,
                                padding=padding, dilation=dilation,
                                groups=groups, bias=False),
                    norm_layer(planes),
                    )

    def forward(self, x):
        identity = x

        out = torch.sigmoid(torch.add(identity, F.interpolate(self.k2(x), identity.size()[2:],mode='bilinear'))) # sigmoid(identity + k2)
        out = torch.mul(self.k3(x), out) # k3 * sigmoid(identity + k2)
        out = self.k4(out) # k4

        return out

class SCA(nn.Module):
    def __init__(self,dim):
        super(SCA, self).__init__()
        self.maxpool=nn.AdaptiveMaxPool2d(1)
        self.avgpool=nn.AdaptiveAvgPool2d(1)
        self.maxconv=nn.Conv2d(dim,dim,1)
        self.avgconv=nn.Conv2d(dim,dim,1)
    def forward(self,x):
        x_max=self.maxpool(x)
        x_avg=self.avgpool(x)

        x_max=self.maxconv(x_max)
        x_avg=self.avgconv(x_avg)

        return (x*x_max*x_avg+x)/math.sqrt(2)

class UPOrDwConv2d(nn.Module):
    def __init__(self, dim, dimout, kernel=3, up=False, down=False,
                 resample_kernel=(1, 3, 3, 1),
                 use_bias=True):
        super().__init__()
        assert not (up and down)
        self.weight = nn.Parameter(torch.zeros(dimout, dim, kernel, kernel))
        self.weight.data = default_init()(self.weight.data.shape)
        if use_bias:
            self.bias = nn.Parameter(torch.zeros(dimout))

        self.up = up
        self.down = down
        self.resample_kernel = resample_kernel
        self.kernel = kernel
        self.use_bias = use_bias

    def forward(self, x):
        if self.up:
            x = upsample_conv_2d(x, self.weight, k=self.resample_kernel)
        elif self.down:
            x = downsample_conv_2d(x, self.weight, k=self.resample_kernel)
        else:
            x = F.conv2d(x, self.weight, stride=1, padding=self.kernel // 2)

        if self.use_bias:
            x = x + self.bias.reshape(1, -1, 1, 1)

        return x

# class Upsample(nn.Module):
#     def __init__(self,dim,dimout):
#         super(Upsample,self).__init__()
#         self.conv=UPOrDwConv2d(dim,dimout,up=True)
#     def forward(self,x):
#         return self.conv(x)
#
# class Downsample(nn.Module):
#     def __init__(self,dim,dimout):
#         super(Downsample,self).__init__()
#         self.conv=UPOrDwConv2d(dim,dimout,down=True)
#     def forward(self,x):
#         return self.conv(x)

def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, 1, 1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)




class AdaptiveGroupNorm(nn.Module):
    def __init__(self, num_groups, dim,dimout, style_dim):
        super().__init__()

        self.norm = nn.GroupNorm(
            num_groups, dim, affine=False, eps=1e-6)
        self.style = dense(style_dim, dim * 2)
        self.conv=nn.Conv2d(dim,dimout,1)

        self.style.bias.data[:dim] = 1
        self.style.bias.data[dim:] = 0

    def forward(self, input, style):
        style = self.style(style)
        gamma, beta = style.chunk(2, 1)

        out = self.norm(input)
        out = gamma * out + beta

        return self.conv(out)

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class RecBlock(nn.Module):
    def __init__(self,act,dim,dimout=None,temb_dim=None,zemb_dim=None,dropout=0.1,skip_rescale=True,device="cuda"):
        super(RecBlock,self).__init__()
        dimout=dimout if dimout else dim
        self.skip_rescale=skip_rescale
        if temb_dim is not None:
            Dense0=nn.Linear(dimout,dimout*4)
            Dense0.weight.data=default_init()(Dense0.weight.shape)
            nn.init.zeros_(Dense0.bias)
            Dense1=nn.Linear(dimout*4,dimout*2)
            Dense1.weight.data = default_init()(Dense1.weight.shape)
            nn.init.zeros_(Dense1.bias)
            self.temb_mlp=nn.Sequential(
                TimeEmbedding(dimout,device=device),
                Dense0,
                Swish(),
                Dense1
            )
        else:
            self.temb_mlp=None
        self.act=act
        # self.norm1=AdaptiveGroupNorm(
        #     min(dim // 4, 32),dim,dimout,zemb_dim
        # )
        self.norm1=nn.Sequential(
            nn.GroupNorm(
                min(dim // 4, 32), dim, eps=1e-6),
            nn.Conv2d(dim,dimout,1),
            nn.Conv2d(dimout,dimout,3,1,1,groups=dimout,bias=False)
        )
        # self.norm2=AdaptiveGroupNorm(
        #     min(dimout // 4, 32), dimout,dimout, zemb_dim
        # )
        self.norm2=nn.Sequential(
            nn.GroupNorm(
                min(dimout // 4, 32), dimout, eps=1e-6),
            nn.Conv2d(dimout,dimout,1),
            nn.Conv2d(dimout,dimout,3,1,1,groups=dimout,bias=False)
        )

        self.dropout = nn.Dropout(p=dropout)
        self.outconv=nn.Conv2d(dimout,dimout,1)

        self.recconv=nn.Conv2d(dim,dimout,1) if dim!=dimout else nn.Identity()
        # self.zrecconv=nn.Conv2d(zemb_dim,dimout,1) if dim!=dimout else nn.Identity()

    def forward(self,x,temb=None,zemb=None):
        # temp=self.act(self.norm1(x,zemb))
        temp = self.act(self.norm1(x))
        if self.temb_mlp is not None:
            temb=self.temb_mlp(temb)[:,:,None,None]
            scale, shift=temb.chunk(2, dim=1)
            temp=temp*(scale+1)+shift
        # temp=self.act(self.norm2(temp,zemb))
        temp = self.act(self.norm2(temp))
        temp=self.dropout(temp)
        temp=self.outconv(temp)
        x=self.recconv(x)
        # zemb=self.zrecconv(zemb)

        if self.skip_rescale:
            # return (x+temp)/math.sqrt(2.),zemb
            return (x + temp) / math.sqrt(2.)
        else:
            # return  x+temp,zemb
            return x + temp

class RecWithAtBlock(nn.Module):
    def __init__(self,at,dim,dimout,temb_dim=None,zemb_dim=None,dropout=0.1,skip_rescale=True,num_recblock=2,device="cuda"):
        super(RecWithAtBlock,self).__init__()
        act=nn.SiLU()
        model=[]
        for _ in range(num_recblock):
            model.append(RecBlock(act,dim,dimout,temb_dim, zemb_dim, dropout, skip_rescale,device=device))
            dim=dimout
            # zemb_dim=dimout
        self.block=nn.Sequential(*model)
        self.at =at
    def forward(self,x,temb,zemb=None):
        assert zemb is None
        for i,model in enumerate(self.block):
            x=model(x,temb,zemb)
        x = self.at(x)
        return x

class LHAttention(nn.Module):
    def __init__(self,dim,use_bias=False):
        super(LHAttention,self).__init__()
        self.Hconv=nn.Sequential(
            nn.GroupNorm(min(dim // 4, 32),dim),
            nn.Conv2d(dim,dim,1,bias=use_bias),
            nn.Conv2d(dim,dim,3,1,1,groups=dim,bias=use_bias)
        )
        self.Lconv=nn.Sequential(
            nn.GroupNorm(min(dim // 4, 32), dim),
            nn.Conv2d(dim, dim, 1,bias=use_bias),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim,bias=use_bias)
        )
        max_dim=min(dim*2*4,512)
        self.mlp=nn.Sequential(
            nn.Conv2d(dim*2,max_dim,1,bias=use_bias),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(max_dim,dim*2,1,bias=use_bias),
        )
    def forward(self,L,H):
        L=self.Lconv(L)
        H_temp=self.Hconv(H)
        x_temp=torch.cat([L,H_temp],dim=1)
        x_L_temp,x_H_temp=self.mlp(x_temp).chunk(2,dim=1)
        x_temp=x_L_temp*x_H_temp
        return (H+x_temp)/math.sqrt(2.)

def get_relative_position_index(win_h, win_w):
    coords = torch.stack(torch.meshgrid([torch.arange(win_h), torch.arange(win_w)], indexing='ij'))
    coords_flatten = torch.flatten(coords, 1)
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()
    relative_coords[:, :, 0] += win_h - 1
    relative_coords[:, :, 1] += win_w - 1
    relative_coords[:, :, 0] *= 2 * win_w - 1
    return relative_coords.sum(-1)



class TRA(nn.Module):
    def __init__(self, dim, num_head, window_size=4):
        super(TRA, self).__init__()
        self.dim = dim
        self.num_head = num_head
        self.window_size = window_size
        self.window_area = window_size ** 2
        self.qkv = nn.Conv2d(dim, 3 * dim, 1, bias=True)
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_head))
        self.register_buffer("relative_position_index", get_relative_position_index(window_size, window_size))
        self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_head, 1, 1))), requires_grad=True)
        self.proj = nn.Conv2d(dim, dim, 1)

    def _get_rel_pos_bias(self):
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_area, self.window_area, -1)
        return relative_position_bias.permute(2, 0, 1).contiguous().unsqueeze(0)

    def forward(self, x0):
        qkv_0 = self.qkv(x0)
        qkv = rearrange(qkv_0, 'b (l c) h w -> b l c h w', l=self.num_head)
        B, L, C, H, W = qkv.size()
        q, k, v = rearrange(
            qkv,
            'b l c (h wh) (w ww) -> (b h w) l (wh ww) c',
            wh=self.window_size, ww=self.window_size
        ).chunk(3, dim=-1)
        attn0 = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-2, -1)
        logit_scale = torch.clamp(self.logit_scale, max=math.log(1. / 0.01)).exp()
        attn1 = attn0 * logit_scale + self._get_rel_pos_bias()
        attn = F.softmax(attn1, dim=-1)
        x = attn @ v
        x = rearrange(
            x,
            '(b h w) l (wh ww) c -> b (l c) (h wh) (w ww)',
            h=H // self.window_size, w=W // self.window_size, wh=self.window_size
        )
        x_out = self.proj(x)
        return x_out

class CCA(nn.Module):
    def __init__(self, dim, num_heads, ifBox=True):
        super(CCA, self).__init__()
        self.factor = num_heads
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1)

    def Calculate_attn(self, q, k, v):
        b, c = q.shape[:2]
        hw = q.shape[-1] // self.factor
        shape_ori = "b (head c) (factor hw)"
        shape_tar = "b head (c factor) hw"
        q = rearrange(q, f'{shape_ori} -> {shape_tar}', factor=self.factor, hw=hw, head=self.num_heads)
        k = rearrange(k, f'{shape_ori} -> {shape_tar}', factor=self.factor, hw=hw, head=self.num_heads)
        v = rearrange(v, f'{shape_ori} -> {shape_tar}', factor=self.factor, hw=hw, head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = torch.softmax(attn, dim=-1)
        out = (attn @ v)
        out = rearrange(out, f'{shape_tar} -> {shape_ori}', factor=self.factor, hw=hw, head=self.num_heads)
        return out

    def forward(self, x):
        b, c, h, w = x.shape
        x_sort, idx_h = x.sort(-2)
        x_sort, idx_w = x_sort.sort(-1)
        x = x_sort
        qkv = self.qkv_dwconv(self.qkv(x))
        q1, k1, v = qkv.chunk(3, dim=1)
        v, idx = v.view(b, c, -1).sort(dim=-1)
        q1 = torch.gather(q1.view(b, c, -1), dim=2, index=idx)
        k1 = torch.gather(k1.view(b, c, -1), dim=2, index=idx)
        out1 = self.Calculate_attn(q1, k1, v)
        out1 = torch.scatter(out1, 2, idx, out1).view(b, c, h, w)
        out = out1
        out_replace = torch.scatter(out, -1, idx_w, out)
        out_replace = torch.scatter(out_replace, -2, idx_h, out_replace)
        return out_replace

class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            window_size,
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.window_size=window_size
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        x=hidden_states
        _, _, H, W = x.shape

        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        if pad_r > 0 or pad_b > 0:
            x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
            _, _, Hp, Wp = x.shape
        else:
            Hp, Wp = H, W
        x = window_partition(x, self.window_size)
        hidden_states=x

        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                            groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                            groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)

        x=out
        x = window_reverse(x, self.window_size, Hp, Wp)
        if pad_r > 0 or pad_b > 0:
            x = x[:, :, :H, :W].contiguous()
        out=x
        return out


class BSCMLP(nn.Module):
    def __init__(self, dim,use_bias=False):
        super(BSCMLP, self).__init__()
        self.norm = nn.GroupNorm(1, dim)
        self.conv1 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=use_bias),
            SCConv(dim, norm_layer=lambda c: nn.GroupNorm(min(32, c // 4), c))
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(dim, dim, 1, bias=use_bias),
            SCConv(dim, norm_layer=lambda c: nn.GroupNorm(min(32, c // 4), c))
        )
        self.act = nn.GELU()
        self.outconv2 = nn.Conv2d(dim, dim, 1, bias=use_bias)
    def forward(self, x):
        temp=self.norm(x)
        x1=self.conv1(temp)
        x2=self.conv2(temp)
        temp=x1*self.act(x2)
        return self.outconv2(temp)


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows

def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x

class MVWA(nn.Module):
    def __init__(self,dim,use_bias=False,window_size=8,num_heads=8,init_values=1e-4):
        super(MVWA, self).__init__()
        self.mamba=MambaVisionMixer(dim,window_size,bias=use_bias)
        self.norm1=nn.GroupNorm(1, dim)
        self.norm2=nn.GroupNorm(1,dim)
        self.attn=SwinBlock(dim,window_size=window_size,num_heads=num_heads,init_values=init_values)
        self.dw=nn.Conv2d(dim,dim,1,groups=dim,bias=use_bias)
        self.mlp=BSCMLP(dim,use_bias=use_bias)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)
    def forward(self,x):
        temp=x
        x=self.norm1(x)
        x=self.dw(x)
        x=temp+self.gamma_1*self.mamba(x)
        x = x + self.gamma_2 * self.mlp(x)
        x=self.attn(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self,dim,window_size=8,num_heads=8,use_bias=False,init_values=1e-4):
        super(SwinBlock, self).__init__()
        self.dim=dim
        self.window_size=window_size
        self.num_heads=num_heads

        self.norm1=nn.LayerNorm(dim)
        self.attn=WindowAttention(
            dim,window_size=(window_size,window_size),num_heads=num_heads,
        )
        # self.norm2=nn.LayerNorm(dim)
        self.mlp=BSCMLP(dim,use_bias=use_bias)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)
    def forward(self,x):
        B, C, H, W = x.shape
        pad_h = (self.window_size - H % self.window_size) % self.window_size
        pad_w = (self.window_size - W % self.window_size) % self.window_size
        x = F.pad(x, (0, pad_w, 0, pad_h))  # pad (left, right, top, bottom)
        _, _, Hp, Wp = x.shape
        shortcut = x
        x = x.permute(0, 2, 3, 1).contiguous()  # B, Hp, Wp, C
        x = self.norm1(x)

        x_windows = RestorMixer.window_partition(x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, self.dim)
        attn_windows = self.attn(x_windows)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, self.dim)
        x = RestorMixer.window_reverse(attn_windows, self.window_size, Hp, Wp)  # B, Hp, Wp, C
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, Hp, Wp

        x = shortcut + self.gamma_1*x
        x=x+self.gamma_2*self.mlp(x)

        if pad_h > 0 or pad_w > 0:
            x = x[:, :, :H, :W]
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

class STBlockWithLocalVision(nn.Module):
    def __init__(self,dim,use_bias=False,init_values=1e-4):
        super(STBlockWithLocalVision, self).__init__()
        self.Sconv1=nn.Conv2d(dim,dim,1,groups=dim,bias=use_bias)
        self.Sconv3=nn.Conv2d(dim,dim,3,1,1,groups=dim,bias=use_bias)
        self.Dconv3=nn.Conv2d(dim,dim,3,1,3,dilation=3,groups=dim,bias=use_bias)
        self.transconv=nn.Sequential(
            nn.Conv2d(4 * dim, 2*dim, 1, bias=use_bias),
            nn.Conv2d(2*dim, dim, 1, bias=use_bias),
        )

        self.norm1 = RMSNorm(dim)
        self.norm2 = RMSNorm(dim)
        self.conv1 =nn.Sequential(
            nn.Conv2d(dim, dim, 1),
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        )
        self.conv2 = nn.Conv2d(dim, dim, 1)
        self.conv3 = nn.Conv2d(dim, dim, 1)
        self.conv4 = nn.Conv2d(dim, dim, 1)
        self.transconv1 = nn.Conv2d(dim, dim * 2, 1)
        self.transconv2 = nn.Conv2d(dim, dim * 2, 1)

        self.sca = SCA(dim)

        self.gamma_1 = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((1, dim, 1, 1)), requires_grad=True)

    def forward(self,x):
        x_temp=x.permute(0, 2, 3, 1).contiguous()
        x_temp = self.norm1(x_temp)
        x_temp=x_temp.permute(0, 3, 1, 2).contiguous()


        # x_temp = self.norm1(x)
        x_temp = self.conv1(x_temp)
        x_temp1, x_temp2 = self.transconv1(x_temp).chunk(2, dim=1)
        x_temp = x_temp1 * x_temp2
        x_temp = self.conv2(self.sca(x_temp))

        x_conv1=self.Sconv1(x)
        x_conv3=self.Sconv3(x)
        x_Dconv3=self.Dconv3(x)

        x_temp=self.transconv(torch.cat([x_temp,x_conv1,x_conv3,x_Dconv3],dim=1))

        x = x_temp*self.gamma_1 + x

        x_temp=x.permute(0, 2, 3, 1).contiguous()
        x_temp = self.norm2(x_temp)
        x_temp=x_temp.permute(0, 3, 1, 2).contiguous()

        # x_temp = self.norm2(x)
        x_temp = self.conv3(x_temp)
        x_temp1, x_temp2 = self.transconv2(x_temp).chunk(2, dim=1)
        x_temp = self.conv4(x_temp1 * x_temp2)
        x = x_temp*self.gamma_2 + x
        return x


class AttnBlockWithPrior(nn.Module):
    def __init__(self,dim,prior_dim,ratic=2,use_bias=False):
        super(AttnBlockWithPrior, self).__init__()
        self.hidden_dim=dim*ratic
        self.norm=nn.GroupNorm(min(dim//4,32),dim)
        self.conv1=nn.Sequential(
            nn.Conv2d(dim, self.hidden_dim, 1,bias=use_bias),
            nn.Conv2d(self.hidden_dim, self.hidden_dim, 3, 1, 1, groups=self.hidden_dim, bias=use_bias),
        )
        self.conv2=nn.Sequential(
            nn.Conv2d(prior_dim, self.hidden_dim*2, 1,bias=use_bias),
            nn.Conv2d(self.hidden_dim*2, self.hidden_dim*2, 3, 1, 1, groups=self.hidden_dim*2, bias=use_bias),
        )

        self.qnorm=nn.LayerNorm(self.hidden_dim)
        self.knorm = nn.LayerNorm(self.hidden_dim)

        self.outconv=nn.Conv2d(self.hidden_dim, dim, 1)
        self.attn=AttnBlock(dim)
    def forward(self,x,prior):
        B,C,H,W=x.shape
        _,_,PH,PW=prior.shape
        temp=x
        x=self.norm(x)
        q=self.conv1(x)
        k,v=self.conv2(prior).chunk(2,dim=1)

        # ----- 新增：对 q 和 k 进行层归一化 -----
        # 将通道维放到最后，对每个位置的特征向量做 LayerNorm
        q = q.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        k = k.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        q = self.qnorm(q)  # 归一化最后一个维度（C）
        k = self.knorm(k)
        q = q.permute(0, 3, 1, 2).contiguous()  # 恢复为 (B, C, H, W)
        k = k.permute(0, 3, 1, 2).contiguous()
        # ------------------------------------

        w = torch.einsum('bchw,bcij->bhwij', q, k) * (self.hidden_dim ** (-0.5))
        w = torch.reshape(w, (B, H, W, PH * PW))
        w = F.softmax(w, dim=-1)
        w = torch.reshape(w, (B, H, W, PH, PW))
        h = torch.einsum('bhwij,bcij->bchw', w, v)
        x=(self.outconv(h)+temp)/math.sqrt(2)

        return self.attn(x)



class HFtoLFSE(nn.Module):
    def __init__(self, dim, r=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim*r, bias=True),
            nn.LeakyReLU(),
            nn.Linear(dim*r, dim, bias=True),
            nn.Sigmoid()
        )
    def forward(self, HF, LF):
        s = self.pool(HF).view(HF.size(0), -1)  # B,C
        g = self.mlp(s).view(HF.size(0), HF.size(1), 1, 1)
        return LF * (1.0 + g)

class LowRankChannelXCA(nn.Module):
    def __init__(self, dim,N, d=32, heads=1):
        super().__init__()
        self.d = d
        self.heads = heads
        self.q = nn.Conv2d(dim, d*heads, 1, bias=False)
        self.qnorm=RMSNorm(N)
        self.knorm=RMSNorm(N)
        self.k = nn.Conv2d(dim, d*heads, 1, bias=False)
        self.v = nn.Conv2d(dim, d*heads, 1, bias=False)
        self.out = nn.Conv2d(d*heads, dim, 1, bias=True)
    def forward(self, x):
        B, C, H, W = x.shape
        N = H * W
        q = self.q(x).view(B, self.heads, self.d, N)  # B,heads,d,N
        k = self.k(x).view(B, self.heads, self.d, N)
        v = self.v(x).view(B, self.heads, self.d, N)

        q=self.qnorm(q)
        k=self.knorm(k)

        scale = (N) ** -0.5
        cov = torch.einsum('bhdn,bhen->bhde', q, k) * scale  # B,heads,d,d
        attn = F.softmax(cov, dim=-1)  # along last dim
        out = torch.einsum('bhde,bhen->bhdn', attn, v)  # B,heads,d,N
        out = out.view(B, self.heads*self.d, H, W)
        return self.out(out)

class PooledSpatialCross(nn.Module):
    def __init__(self, dim, S=8):
        super().__init__()
        self.S = S
        self.q_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.k_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.v_proj = nn.Conv2d(dim, dim, 1, bias=False)
        self.qnorm=RMSNorm(dim)
        self.knorm=RMSNorm(dim)
        self.out = nn.Conv2d(dim, dim, 1, bias=True)
    def forward(self, x):
        # x: B,C,H,W
        B, C, H, W = x.shape
        N = H*W
        q = self.q_proj(x).view(B, C, N)  # B,C,N
        k = self.k_proj(x)
        v = self.v_proj(x)
        # pooled k,v to SxS
        k_small = F.adaptive_avg_pool2d(k, (self.S, self.S)).view(B, C, self.S*self.S)  # B,C,S2
        v_small = F.adaptive_avg_pool2d(v, (self.S, self.S)).view(B, C, self.S*self.S)

        q=q.permute(0,2,1).contiguous()
        k_small=k_small.permute(0,2,1).contiguous()
        q=self.qnorm(q)
        k_small=self.knorm(k_small)
        q=q.permute(0,2,1).contiguous()
        k_small=k_small.permute(0,2,1).contiguous()

        # q vs k_small -> (B, N, S2)
        scale = (C) ** -0.5
        w = torch.einsum('bcn,bcm->bnm', q, k_small)*scale  # B,N,S2
        w = F.softmax(w, dim=-1)
        out = torch.einsum('bnm,bcm->bcn', w, v_small)  # B,C,N
        out = out.view(B, C, H, W)
        return self.out(out)

class HF_LG_Block(nn.Module):
    def __init__(self, dim,N, d=32, heads=1, pool_S=8, use_local_conv=True):
        super().__init__()
        self.gate = HFtoLFSE(dim, r=4)
        self.local_conv = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, groups=dim),  # depthwise local
            nn.Conv2d(dim, dim, 1)
        ) if use_local_conv else nn.Identity()
        self.ch_xca = LowRankChannelXCA(dim,N=N, d=d, heads=heads)
        self.spat = PooledSpatialCross(dim, S=pool_S)
        self.merge = nn.Conv2d(dim*3, dim, 1)
        self.norm = RMSNorm(dim)
        self.act = nn.GELU()
        self.out = nn.Conv2d(dim, dim, 1)
    def forward(self, L, H):
        Lg = self.gate(H, L)
        loc = self.local_conv(Lg)
        xc = self.ch_xca(Lg)
        xs = self.spat(Lg)
        merged = torch.cat([loc, xc, xs], dim=1)
        merged = self.act(self.merge(merged)) #B C H W

        merged=merged.permute(0,2,3,1).contiguous()
        merged = self.norm(merged)
        merged =merged.permute(0,3,1,2).contiguous()

        out = (L + self.out(merged))/math.sqrt(2)
        return out

class Unet(nn.Module):
    def __init__(self,
                 channels=3,
                 dim=48,
                 layerout=[1,1,2,4,8],
                 Attnout=[1,2,2,4],
                 ker=[31,21,11],
                 window_size=[8,16,16],
                 num_heads=[8,4,4],
                 temb_dim=None,  #dim
                 zemb_dim=None,  #dim
                 dropout=0.1,
                 skip_rescale=True,
                 num_recblock=2,
                 ratic=16,
                 use_bias=False,
                 factor=2.68,
                 device="cuda"):
        super(Unet,self).__init__()
        self.channels=channels
        self.out_dim=channels
        zemb_dim=dim
        self.dwt,self.iwt=DWT_2D("haar"),IDWT_2D("haar")
        self.lin_conv=nn.Conv2d(channels,dim//2,1)
        self.hin_conv=nn.Conv2d(channels*3,dim//2,1)
        self.zlin_conv=nn.Conv2d(channels,zemb_dim//2,1)
        self.zhin_conv=nn.Conv2d(channels*3,zemb_dim//2,1)
        self.Colorfeature=CCA(dim//2,num_heads=8)
        self.Structurefeature=TRA(dim//2,num_head=8)
        self.l_conv=nn.Sequential(
            nn.Conv2d(dim,dim,1),
            nn.Conv2d(dim,dim,3,1,1,groups=dim)
        )
        self.h_conv=nn.Sequential(
            nn.Conv2d(dim,dim,1),
            nn.Conv2d(dim,dim,3,1,1,groups=dim)
        )
        l_downsmodel=[]
        l_upmodel=[]
        h_downsmodel=[]
        h_upmodel=[]
        layers=[(j,layerout[i+1]) for i,j in enumerate(layerout[:-1])]
        self.layer_len=len(layers)
        img_size=128
        for index,(i,j) in enumerate(layers):
            l_downsmodel.append(nn.Sequential(
                RecWithAtBlock(
                    STBlockWithLocalVision(dim * j,use_bias=use_bias),
                    dim * i,
                    dim * j,
                    temb_dim=temb_dim,
                    dropout=dropout,
                    skip_rescale=skip_rescale,
                    num_recblock=num_recblock,
                    device=device
                ),
                HF_LG_Block(dim * j,N=img_size*img_size,heads=j,pool_S=img_size//2),
                Downsample(dim * j, dim * j)
            ))
            h_downsmodel.append(nn.Sequential(
                RecWithAtBlock(
                    nn.Identity(),
                    dim * i,
                    dim * j,
                    temb_dim=temb_dim,
                    dropout=dropout,
                    skip_rescale=skip_rescale,
                    num_recblock=num_recblock,
                    device=device
                ),
                LHAttention(dim * j),
                Downsample(dim * j, dim * j)
            ))
            img_size=img_size//2

        self.l_downs=nn.Sequential(*l_downsmodel)
        self.h_downs=nn.Sequential(*h_downsmodel)

        self.l_mid_conv1=nn.Conv2d(dim*layerout[-1],dim*layerout[-1],1)
        self.l_at=AttnBlockWithPrior(dim*layerout[-1],zemb_dim//2)
        self.l_mid_conv2=nn.Conv2d(dim*layerout[-1],dim*layerout[-1],1)

        self.h_mid_conv1 = nn.Conv2d(dim * layerout[-1], dim * layerout[-1], 1)
        self.h_at=AttnBlockWithPrior(dim*layerout[-1],zemb_dim//2)
        self.h_mid_conv2 = nn.Conv2d(dim * layerout[-1], dim * layerout[-1], 1)

        for index,(i,j) in enumerate(reversed(layers)):
            img_size=img_size*2
            l_upmodel.append(nn.Sequential(
                RecWithAtBlock(
                    MVWA(dim * i, window_size=window_size[index], num_heads=num_heads[index]),
                    dim * j * 2,
                    dim * i,
                    temb_dim=temb_dim,
                    dropout=dropout,
                    skip_rescale=skip_rescale,
                    num_recblock=num_recblock,
                    device=device
                ),
                HF_LG_Block(dim * i, N=img_size * img_size, heads=i, pool_S=img_size // 2),
                Upsample(dim * j,dim * j)
            ))
            h_upmodel.append(nn.Sequential(
                RecWithAtBlock(
                    nn.Identity(),
                    dim * j * 2,
                    dim * i,
                    temb_dim=temb_dim,
                    dropout=dropout,
                    skip_rescale=skip_rescale,
                    num_recblock=num_recblock,
                    device=device
                ),
                LHAttention(dim * i),
                Upsample(dim * j, dim * j)
            ))

        self.l_ups=nn.Sequential(*l_upmodel)
        self.h_ups=nn.Sequential(*h_upmodel)

        self.l_out=nn.Conv2d(dim,channels,1)
        self.h_out=nn.Conv2d(dim,channels*3,1)


    def forward(self,x,zemb,temb):

        with torch.no_grad():
            if isinstance(temb, int) or isinstance(temb, float):
                temb = torch.tensor([temb]).to(x.device)

        txl,txlh,txhl,txhh=self.dwt(x)
        tzl,tzlh,tzhl,tzhh=self.dwt(zemb)

        txh=torch.cat([txlh,txhl,txhh],dim=1)
        tzh=torch.cat([tzlh,tzhl,tzhh],dim=1)
        txl=self.lin_conv(txl)
        txh=self.hin_conv(txh)
        tzl=self.zlin_conv(tzl)
        tzh=self.zhin_conv(tzh)
        tzl=self.Colorfeature(tzl)
        tzh=self.Structurefeature(tzh)
        txl=self.l_conv(torch.cat([txl,tzl],dim=1))
        txh=self.h_conv(torch.cat([txh,tzh],dim=1))

        lop=[]
        hop=[]

        for i in range(self.layer_len):

            l_block,l_attn,l_down=self.l_downs[i]
            txl = l_block(txl, temb)

            h_block,h_attn,h_down=self.h_downs[i]
            txh = h_block(txh, temb)

            txl_temp=l_attn(L=txl,H=txh)
            txh_temp=h_attn(L=txl,H=txh)
            lop.append(txl_temp)
            txl = l_down(txl_temp)
            hop.append(txh_temp)
            txh = h_down(txh_temp)



        txl=self.l_mid_conv2(self.l_at(self.l_mid_conv1(txl),tzl))
        txh=self.h_mid_conv2(self.h_at(self.h_mid_conv1(txh),tzh))

        for i in range(self.layer_len):

            l_block,l_attn,l_up=self.l_ups[i]
            txl=l_up(txl)
            txl=torch.cat([txl,lop.pop()],dim=1)
            txl = l_block(txl, temb)

            h_block, h_attn, h_up = self.h_ups[i]
            txh = h_up(txh)
            txh = torch.cat([txh, hop.pop()], dim=1)
            txh = h_block(txh, temb)

            txl_temp=l_attn(L=txl,H=txh)
            txh_temp = h_attn(L=txl,H=txh)
            txl=txl_temp
            txh=txh_temp

        txl=self.l_out(txl)
        txh=self.h_out(txh)
        txlh,txhl,txhh=torch.chunk(txh,chunks=3,dim=1)

        out=self.iwt(txl,txlh,txhl,txhh)
        return [out]