import logging
import torch
import torch.nn as nn

logger = logging.getLogger('base')

####################
# define network
####################

def define_G(opt):
    model_opt = opt['model']
    from model.ddpm import diffusionWithResidualPro,unet12

    model = unet12.Unet(
        channels=model_opt['unet']['channel'],
        dim=model_opt['unet']['dim'],
        window_size=model_opt['unet']['window_size'],
        layerout=model_opt['unet']['layerout'],
        temb_dim=model_opt['unet']['temb_dim'],
        zemb_dim=model_opt['unet']['zemb_dim'],
        dropout=model_opt['unet']['dropout'],
        skip_rescale=model_opt['unet']['skip_rescale'],
        num_recblock=model_opt['unet']['num_recblock'],
        ratic=model_opt['unet']['ratic'],
        use_bias=model_opt['unet']['use_bias'],
        factor=model_opt['unet']['factor']
    )
    netG=diffusionWithResidualPro.ResidualDiffusion(
        model
    )
    if opt['gpu_ids'] and opt['distributed']:
        assert torch.cuda.is_available()
        netG = nn.DataParallel(netG)
    return netG
