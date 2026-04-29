import torch
from torchvision import models,transforms
import torch.nn as nn
import torch.nn.functional as F
import math
import piq
import pywt
import numpy as np
from util.SWT import SWTForward
from kornia.color import rgb_to_hsv

def hsv_loss_kornia(img1, img2,
                    weight_h=1.0, weight_s=1.0, weight_v=1.0,
                    eps=1e-6):
    """
    img1, img2: tensors in range [-1, 1], shape (B,3,H,W)
    return: scalar loss
    Uses kornia.color.rgb_to_hsv
    """
    # map to [0,1] and clamp (keep dtype/device)
    img1_pos = (img1 + 1.0) / 2.0
    img2_pos = (img2 + 1.0) / 2.0

    # kornia expects (B,3,H,W) and returns (B,3,H,W) with H,S,V in [0,1]
    hsv1 = rgb_to_hsv(img1_pos)
    hsv2 = rgb_to_hsv(img2_pos)

    h1, s1, v1 = hsv1[:, 0, ...], hsv1[:, 1, ...], hsv1[:, 2, ...]
    h2, s2, v2 = hsv2[:, 0, ...], hsv2[:, 1, ...], hsv2[:, 2, ...]

    # hue -> unit vector (handles wrap smoothly)
    theta1 = h1
    theta2 = h2
    x1, y1 = torch.cos(theta1), torch.sin(theta1)
    x2, y2 = torch.cos(theta2), torch.sin(theta2)

    chord = torch.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + eps)
    h_norm = chord / 2.0  # normalized hue difference in [0,1]

    # weight hue by saturation to ignore grayish pixels
    sat_weight = (s1 + s2) / 2.0
    h_term = (h_norm * sat_weight).mean()

    s_term = torch.abs(s1 - s2).mean()
    v_term = torch.abs(v1 - v2).mean()

    loss = weight_h * h_term + weight_s * s_term + weight_v * v_term
    return loss


class VGG19Features(nn.Module):
    def __init__(self,device,num=36):
        super(VGG19Features,self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.vgg19model = vgg.features[:num]
        self.vgg19model = self.vgg19model.to(device)

        self.vgg19model.eval()

        for param in self.vgg19model.parameters():
            param.requires_grad = False
        self.device = device

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        return self.vgg19model(x)

class SWTLoss(nn.Module):
    def __init__(self,device):
        super(SWTLoss,self).__init__()
        self.device = device
        wavelet=pywt.Wavelet("sym7")
        dlo = wavelet.dec_lo
        an_lo = np.divide(dlo, sum(dlo))
        an_hi = wavelet.dec_hi
        rlo = wavelet.rec_lo
        syn_lo = 2 * np.divide(rlo, sum(rlo))
        syn_hi = wavelet.rec_hi
        filters = pywt.Wavelet('wavelet_normalized', [an_lo, an_hi, syn_lo, syn_hi])
        self.sfm = SWTForward(1, filters, 'periodic').to(self.device)

        self.l_pix_w = 0.1
        self.l_pix_w_lh = 0.01
        self.l_pix_w_hl = 0.01
        self.l_pix_w_hh = 0.05
        self.l_fea_w = 1

        self.cri_pix=nn.L1Loss().to(self.device)
        self.cri_fea = nn.L1Loss().to(self.device)
        self.netF = piq.DISTS()


    def forward(self,pred,gt):
        """
            pred: The RGB image predicted by the model, shape [N, 3, H, W], values should be in [-1, 1]
            gt:   The ground truth HR RGB image, shape [N, 3, H, W], values should be in [-1, 1]
            Returns: Combined loss value
        """
        pred=(pred+1)/2
        gt=(gt+1)/2
        sr_img_y = 16.0 + (pred[:, 0:1, :, :] * 65.481 + pred[:, 1:2, :, :] * 128.553 + pred[:, 2:, :,:] * 24.966)
        wavelet_sr = self.sfm(sr_img_y)[0]

        LL_band = wavelet_sr[:, 0:1, :, :]
        LH_band = wavelet_sr[:, 1:2, :, :]
        HL_band = wavelet_sr[:, 2:3, :, :]
        HH_band = wavelet_sr[:, 3:, :, :]

        hr_img_y = 16.0 + (gt[:, 0:1, :, :] * 65.481 + gt[:, 1:2, :, :] * 128.553 + gt[:, 2:, :,:] * 24.966)
        wavelet_hr = self.sfm(hr_img_y)[0]

        LL_band_hr   = wavelet_hr[:,0:1, :, :]
        LH_band_hr   = wavelet_hr[:,1:2, :, :]
        HL_band_hr   = wavelet_hr[:,2:3, :, :]
        HH_band_hr   = wavelet_hr[:,3:, :, :]

        l_g_pix = self.l_pix_w * self.cri_pix(LL_band, LL_band_hr)
        l_g_pix_lh = self.l_pix_w_lh * self.cri_pix(LH_band, LH_band_hr)
        l_g_pix_hl = self.l_pix_w_hl * self.cri_pix(HL_band, HL_band_hr)
        l_g_pix_hh = self.l_pix_w_hh * self.cri_pix(HH_band, HH_band_hr)
        l_g_total = l_g_pix + l_g_pix_lh + l_g_pix_hl + l_g_pix_hh

        l_g_fea = self.l_fea_w * self.netF(gt, pred)
        l_g_total += l_g_fea

        return l_g_total



# vgg19 = models.vgg19(pretrained=False)
# print(vgg19)
# for i, layer in enumerate(vgg19.features[:36]):
#     print(f"layer {i}: {layer}")