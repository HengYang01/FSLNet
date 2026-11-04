import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from mamba_ssm.modules.mamba_simple import Mamba
import torch.nn.init as init
import math
from einops import rearrange
import numbers
import torch
import torch.nn as nn

def _crop_flatten(tensor: torch.Tensor, patch_size: int):
    """
    [B, C, H, W] -> [B, C, (H/ps)*(W/ps), ps*ps]
    """
    B, C, H, W = tensor.shape
    ps = patch_size
    assert H % ps == 0 and W % ps == 0, "H, W must be divisible by patch_size."
    x = tensor.view(B, C, H // ps, ps, W // ps, ps) \
             .permute(0, 1, 2, 4, 3, 5) \
             .contiguous() \
             .view(B, C, (H // ps) * (W // ps), ps * ps)
    return x

def _unstack_reconstruct(tensor: torch.Tensor, patch_size: int):
    B, C, L = tensor.shape
    H = W = int(math.isqrt(L))
    assert H * W == L, "L must be a perfect square (H*W)."
    ps = patch_size
    assert H % ps == 0 and W % ps == 0, "H, W must be divisible by patch_size."

    num_patches_h = H // ps
    num_patches_w = W // ps

    # [B, C, H*W] -> [B, C, num_patches_h, num_patches_w, ps, ps]
    x = tensor.view(B, C, num_patches_h, num_patches_w, ps, ps)
    # -> [B, C, H, W]
    x = x.permute(0, 1, 2, 4, 3, 5).contiguous().view(B, C, H, W)
    return x


def crop_flatten_4(tensor, patch_size: int = 4):
    return _crop_flatten(tensor, patch_size=4)

def unstack_reconstruct_4(tensor, patch_size: int = 4):
    return _unstack_reconstruct(tensor, patch_size=4)

def crop_flatten_8(tensor, patch_size: int = 8):
    return _crop_flatten(tensor, patch_size=8)

def unstack_reconstruct_8(tensor, patch_size: int = 8):
    return _unstack_reconstruct(tensor, patch_size=8)


class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder1 = Mamba(dim)
        self.encoder2 = Mamba(dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, xin):
        B, C, H, W = xin.shape
        L = H * W
        # x_4 = crop_flatten_4(xin)

        x_4 = crop_flatten_4(xin).view(B, -1, L).permute(0, 2, 1)
        x_8 = crop_flatten_8(xin).view(B, -1, L).permute(0, 2, 1)

        x_4 = self.norm1(x_4)
        x_8 = self.norm2(x_8)

        y = (
            unstack_reconstruct_4(self.encoder1(x_4).permute(0, 2, 1))
            + unstack_reconstruct_8(self.encoder2(x_8).permute(0, 2, 1))
            + xin
        )
        return y


class Get_ltpe(nn.Module):
    def __init__(self):
        super(Get_ltpe, self).__init__()
        all_kernels = []
        center = (1, 1)
        offsets = [(-1, 0), (0, -1), (1, 0), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        for offset in offsets:
            kernel = [[0 for _ in range(3)] for _ in range(3)]
            i = center[0] + offset[0]
            j = center[1] + offset[1]
            if 0 <= i < 3 and 0 <= j < 3:
                kernel[i][j] = -1
            kernel[center[0]][center[1]] = 1
            all_kernels.append(kernel)
        all_kernels = torch.FloatTensor(all_kernels).unsqueeze(1)
        self.weights = nn.Parameter(data=all_kernels, requires_grad=False)
        self.learnable_weights = nn.Parameter(torch.rand(len(all_kernels)))

    def forward(self, x):
        x_ltpe = F.conv2d(x, self.weights, padding=1)
        out = torch.sum(x_ltpe * self.learnable_weights.view(1, -1, 1, 1), dim=1).unsqueeze(1)
        return out

class Freprocess(nn.Module):
    def __init__(self, channels):
        super(Freprocess, self).__init__()
        self.pre1 = nn.Conv2d(channels, channels, 1, 1, 0)
        
    def forward(self, msf):
        _, _, H, W = msf.shape
        msF = torch.fft.fft2(self.pre1(msf) + 1e-8, norm='backward')
        msF_amp = torch.abs(msF)
        msF_pha = torch.angle(msF)
        return msF_amp, msF_pha

class LearnableBoxMask(nn.Module):
    def __init__(self, img_size, init_box=None, tau=6.0, hard_eval=False, tau_min=2.0, tau_max=12.0):
        super(LearnableBoxMask, self).__init__()
        H, W = img_size
        self.H, self.W = H, W
        self.hard_eval = hard_eval

        self.log_tau = nn.Parameter(torch.log(torch.tensor(float(tau))))
        self.register_buffer("tau_min", torch.tensor(float(tau_min)))
        self.register_buffer("tau_max", torch.tensor(float(tau_max)))

        if init_box is None: 
            cx0 = torch.rand(1) * (W * 0.5) + W * 0.25
            cy0 = torch.rand(1) * (H * 0.5) + H * 0.25
            w0  = torch.tensor([W * 0.5])
            h0  = torch.tensor([H * 0.5])
        else:
            x1, y1, x2, y2 = init_box
            cx0 = torch.tensor([(x1 + x2) / 2.0])
            cy0 = torch.tensor([(y1 + y2) / 2.0])
            w0  = torch.tensor([max(1.0, x2 - x1)])
            h0  = torch.tensor([max(1.0, y2 - y1)])

        self.cx     = nn.Parameter(cx0.float())
        self.cy     = nn.Parameter(cy0.float())
        self.log_w  = nn.Parameter(w0.log().float())
        self.log_h  = nn.Parameter(h0.log().float())

        yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
        self.register_buffer("xx", xx.float())
        self.register_buffer("yy", yy.float())

    def forward(self):
        tau = self.log_tau.exp().clamp(self.tau_min.item(), self.tau_max.item())

        w = self.log_w.exp().clamp(min=1.0, max=float(self.W))
        h = self.log_h.exp().clamp(min=1.0, max=float(self.H))

        left   = self.cx - w / 2
        right  = self.cx + w / 2
        top    = self.cy - h / 2
        bottom = self.cy + h / 2

        mask_x = torch.sigmoid(tau * (self.xx - left)) * (1.0 - torch.sigmoid(tau * (self.xx - right)))
        mask_y = torch.sigmoid(tau * (self.yy - top))  * (1.0 - torch.sigmoid(tau * (self.yy - bottom)))
        mask = (mask_x * mask_y).unsqueeze(0).unsqueeze(0)  # [1,1,H,W]

        if not self.training and self.hard_eval:
            return (mask > 0.5).float()
        return mask


class GLfrepeocess(nn.Module):
    def __init__(self, channels, img_size):
        super(GLfrepeocess, self).__init__()
        self.Globalfreprocess = Freprocess(channels)
        self.amp_mask = LearnableBoxMask(img_size)

    def forward(self, msf):
        _, _, H, W = msf.shape
        Global_msf_amp, Global_msf_pha = self.Globalfreprocess(msf)
        amp_mask = self.amp_mask()               # [1,1,H,W]
        masked_amp = amp_mask * Global_msf_amp
        reconstructed_X = masked_amp * (torch.cos(Global_msf_pha) + 1j * torch.sin(Global_msf_pha))
        out_msf = torch.abs(torch.fft.ifft2(reconstructed_X, s=(H, W), norm='backward'))
        return out_msf

class FeatureProcess(nn.Module):
    def __init__(self, channels, img_size=(64,64)):
        super(FeatureProcess, self).__init__()
        self.layer1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.layer2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.block  = GLfrepeocess(channels, img_size)

    def forward(self, ms, pan_edge):
        fre_add = self.block(ms)
        fuse_2 = self.layer1(ms + fre_add)
        fuse_final = self.layer2(fuse_2 + pan_edge)
        return fuse_final

class Network(nn.Module):
    def __init__(self, num_bands=16, channels=12):
        super(Network, self).__init__()
        self.channels = channels
        self.msfa_size = [4, 4]
        self.conv1 = nn.Conv2d(num_bands + 1, channels, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, mosaic, pan):
        mosaic_reshape = torch.nn.functional.pixel_unshuffle(mosaic, downscale_factor=4)
        mosaic_interpolate = torch.nn.functional.interpolate(mosaic_reshape, scale_factor=8, mode="bilinear")
        x = torch.cat((mosaic_interpolate, pan), 1)
        y = self.conv1(x) + pan.repeat(1, self.channels, 1, 1)
        return y

class Net(nn.Module):
    def __init__(self, in_channels=1, out_channels=16, n_feat1=48, n_feat2=32, n_feat3=16):
        super(Net, self).__init__()
        self.process64   = FeatureProcess(n_feat1, img_size=(64, 64))
        self.process32   = FeatureProcess(n_feat1, img_size=(32, 32))
        self.process16   = FeatureProcess(n_feat1, img_size=(16, 16))
        self.process64_2 = FeatureProcess(n_feat1, img_size=(64, 64))

        self.down1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.down4 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.up1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.up2 = nn.Upsample(scale_factor=2, mode='nearest')

        self.post1 = nn.Conv2d(n_feat1 + n_feat2 + n_feat3, n_feat1, 1, 1, 0)
        self.post2 = nn.Conv2d(n_feat1, out_channels, 1, 1, 0)

        self.layer = Network(num_bands=16, channels=n_feat1)

        self.getltpe1 = Get_ltpe()
        self.getltpe2 = Get_ltpe()
        self.getltpe3 = Get_ltpe()
        self.getltpe4 = Get_ltpe()

        self.mamba_64   = SingleMambaBlock(n_feat1)
        self.mamba_32   = SingleMambaBlock(n_feat1)
        self.mamba_16   = SingleMambaBlock(n_feat1)
        self.mamba_64_2 = SingleMambaBlock(n_feat1)

    def forward(self, ms, pan):
        if not isinstance(pan, torch.Tensor):
            raise Exception('User does not provide pan image!')
        _, _, m, n = ms.shape
        _, _, M, N = pan.shape

        msf = self.layer(ms, pan)

        ms64 = msf
        pan64 = pan
        pan64_edge = self.getltpe1(pan)

        ms32 = self.down1(msf)
        pan32 = self.down2(pan)
        pan32_edge = self.getltpe2(pan32)
        # pan32 = pan32.repeat(1, ms32.shape[1], 1, 1)

        ms16 = self.down3(ms32)
        pan16 = self.down4(pan32)
        pan16_edge = self.getltpe3(pan16)
        # pan16 = pan16.repeat(1, ms16.shape[1], 1, 1)

        HR64 = self.process64(ms64, pan64_edge)
        HR64 = self.mamba_64(HR64)

        HR32 = self.process32(ms32, pan32_edge)
        HR32 = self.mamba_32(HR32)

        HR16 = self.process16(ms16, pan16_edge)
        HR16 = self.mamba_16(HR16)

        HR32  = self.up1(HR16) + HR32
        HR_CAT = self.up2(HR32) + HR64

        # pan_last = pan
        pan_last_edge = self.getltpe4(pan)

        HR = self.process64_2(HR_CAT, pan_last_edge)
        HR = self.mamba_64_2(HR)

        HR = self.post2(HR + msf)
        return HR

from thop import profile
a = torch.rand(1, 1, 32, 32).cuda()
b = torch.rand(1, 1, 64, 64).cuda()
model = Net().cuda()
print(model)
flops, params = profile(model, inputs=(a, b))
print(f"FLOPs: {flops / 1e9:.4f} G")
print(f"Params: {params / 1e6:.4f} M")


total = sum([param.nelement() for param in model.parameters()])
print('Number of parameter: % .4fM' % (total / 1e6))

