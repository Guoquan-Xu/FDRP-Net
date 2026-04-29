import torch
import data as Data
import model as Model
import argparse
import logging
import util.logger as Logger
from util.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
from model.SimplestColorBalance import simplest_color_balance
import time
import Norm.metrics as metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default='config/ModelWithColor.json',
                        help='JSON file for configuration')
    parser.add_argument('-p', '--phase', type=str, choices=['val'], help='val(generation)', default='val')
    parser.add_argument('-gpu', '--gpu_ids', type=str, default=None)
    parser.add_argument('-debug', '-d', action='store_true',default=False)
    parser.add_argument('-enable_wandb', action='store_true',default=False)
    parser.add_argument('-log_infer', action='store_true',default=False)
    parser.add_argument('-outImg', action='store_true', default="./result/unet12/UIEB")
    Dataname='unet12UIEB'
    # parse configs
    args = parser.parse_args()
    opt = Logger.parse(args)
    opt = Logger.dict_to_nonedict(opt)

    # logging
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])

    # Initialize WandbLogger
    if opt['enable_wandb']:
        wandb_logger = WandbLogger(opt)
    else:
        wandb_logger = None

    # dataset
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'val':
            val_set = Data.create_dataset(dataset_opt, phase)
            val_loader = Data.create_dataLoader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model
    diffusion = Model.create_model(opt,device)
    logger.info('Initial Model Finished')

    logger.info('Begin Model Inference.')


    idx = 0
    sumpsnr, sumssim, sumLpips, sumuiqm, sumuciqe, sumuranker = 0., 0., 0., 0., 0., 0.
    N=0

    result_path = '{}'.format(args.outImg)
    os.makedirs(result_path, exist_ok=True)

    if hasattr(diffusion, 'ema') and diffusion.ema is not None:
        diffusion.ema.copy_params_from_ema_to_model()

    for _, (val_data,trainName) in enumerate(val_loader):
        idx += 1
        diffusion.feed_data(val_data)
        diffusion.test(continous=True)
        visuals = diffusion.get_current_visuals()

        target_img = metrics.tensor2img(visuals['targetImg'],min_max=(-1,1))
        restore_img = metrics.tensor2img(visuals['restored'],min_max=(-1,1))

        restore_img=simplest_color_balance(restore_img)

        psnr,ssim,CLIPIPS,uiqm,uciqe,uranker = metrics.getEvaluationIndex(restore_img, target_img, 'Ref')

        sumpsnr += psnr
        sumssim += ssim
        sumLpips += CLIPIPS
        sumuiqm += uiqm
        sumuciqe += uciqe
        sumuranker+=uranker
        N += 1

        metrics.save_img(restore_img, '{}/{}.png'.format(result_path, trainName[0]))

        with open(os.path.join(f'{Dataname}_metrics.txt'), 'a') as f:
            f.write('{}: psnr={} ssim={} lpips={} uiqm={} uciqe={} uranker={}\n'.format(f'{trainName[0]}.png', psnr, ssim,CLIPIPS, uiqm, uciqe,uranker))

    mpsnr = sumpsnr / N
    mssim = sumssim / N
    mlpips = sumLpips / N
    muiqm = sumuiqm / N
    muciqe = sumuciqe / N
    muranker = sumuranker / N

    print("PSNR: {}, SSIM: {},LPIPS: {} UIQM: {}, UCIQE: {} URanker: {}".
                 format(mpsnr, mssim, mlpips, muiqm, muciqe, muranker))
    with open(os.path.join(f'{Dataname}_metrics.txt'), 'a') as f:
        f.write(
            'Average: psnr={} ssim={} lpips={} uiqm={} uciqe={} uranker={}\n'.format(mpsnr, mssim, mlpips,
                                                                                             muiqm, muciqe,
                                                                                             muranker))




