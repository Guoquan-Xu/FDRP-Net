import torch
from tqdm import tqdm
import data
import model
import argparse
import logging
import util.logger as Logger
import util.metrics as Metrics
from util.wandb_logger import WandbLogger
from tensorboardX import SummaryWriter
import os
import numpy as np
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')

os.environ['WANDB_API_KEY'] = 'your WANDB_API_KEY'


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument('-c','--config',type=str,default='config/ModelWithColor.json',
                        help='JSON file for configuration')
    parser.add_argument('-p','--phase',type=str,choices=['train','val'],
                        help='Run either train(training) or val(generation)',default='train')
    parser.add_argument('-gpu','--gpu_ids',type=str,default=None)
    parser.add_argument('-debug','-d',type=bool,default=False)
    parser.add_argument('-enable_wandb',default=True)
    parser.add_argument('-log_wandb_ckpt', action='store_true',default=False)
    parser.add_argument('-log_eval', action='store_true',default=True)


    args=parser.parse_args()
    opt=Logger.parse(args)
    opt=Logger.dict_to_nonedict(opt)

    torch.backends.cudnn.enabled=True
    torch.backends.cudnn.benchmark=True

    Logger.setup_logger(None, opt['path']['log'],
                        'train', level=logging.INFO, screen=True)
    Logger.setup_logger('val', opt['path']['log'], 'val', level=logging.INFO)
    logger = logging.getLogger('base')
    logger.info(Logger.dict2str(opt))
    tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
    device='cuda' if torch.cuda.is_available() else 'cpu'

    if opt['enable_wandb']:
        import wandb
        wandb_logger = WandbLogger(opt)
        wandb.define_metric('validation/val_step')
        wandb.define_metric('epoch')
        wandb.define_metric("validation/*", step_metric="val_step")
        val_step = 0
    else:
        wandb_logger = None

    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train' and args.phase != 'val':
            train_set = data.create_dataset(dataset_opt, phase)
            train_loader = data.create_dataLoader(
                train_set, dataset_opt, phase)
        elif phase == 'val':
            val_set = data.create_dataset(dataset_opt, phase)
            val_loader = data.create_dataLoader(
                val_set, dataset_opt, phase)
    logger.info('Initial Dataset Finished')

    diffusion = model.create_model(opt,device)
    logger.info('Initial Model Finished')

    current_step = diffusion.begin_step
    current_epoch = diffusion.begin_epoch
    n_iter = opt['train']['n_iter']

    if opt['path']['resume_state']:
        logger.info('Resuming training from epoch: {}, iter: {}.'.format(
            current_epoch, current_step))

    if opt['phase'] == 'train':
        while current_step < n_iter:
            current_epoch += 1
            avg_loss=0
            temp_loss=0
            loader_data = iter(train_loader)
            tbar = range(len(train_loader))
            tbar = tqdm(tbar, ncols=130, leave=True,total=len(train_loader),position=0)
            for i in tbar:
                train_data = next(loader_data)
                current_step += 1
                if current_step > n_iter:
                    break
                diffusion.feed_data(train_data)
                temp_loss=diffusion.optimize_parameters()
                # log
                if current_step % opt['train']['print_freq'] == 0:
                    logs = diffusion.get_current_log()
                    message = '<epoch:{:3d}, iter:{:8d}>, lr:{:.4e}> '.format(
                        current_epoch, current_step, diffusion.get_current_learning_rate())
                    for k, v in logs.items():
                        message += '{:s}: {:.4e} '.format(k, v)
                        tb_logger.add_scalar(k, v, current_step)
                    logger.info(message)

                    if wandb_logger:
                        wandb_logger.log_metrics(logs)

                if current_step % opt['train']['save_checkpoint_freq'] == 0:
                    logger.info('Saving models and training states.')
                    diffusion.save_network(current_epoch, current_step)

                    if wandb_logger and opt['log_wandb_ckpt']:
                        wandb_logger.log_checkpoint(current_epoch, current_step)

                # validation
                if current_step % opt['train']['val_freq'] == 0:
                    avg_psnr = 0.0
                    avg_ssim = 0.0
                    avg_LPIPS = 0.0
                    idx = 0
                    result_path = '{}/{}'.format(opt['path']
                                                 ['results'], current_epoch)
                    os.makedirs(result_path, exist_ok=True)

                    original_state = None
                    ema_swapped=False
                    if hasattr(diffusion, 'ema') and diffusion.ema is not None:
                        original_state = {k: v.clone() for k, v in diffusion.netG.state_dict().items()}
                        diffusion.ema.copy_params_from_ema_to_model()
                        ema_swapped = True

                    for _,  val_data in enumerate(val_loader):
                        idx += 1
                        diffusion.feed_data(val_data)
                        diffusion.test(continous=False)
                        visuals = diffusion.get_current_visuals()
                        restore_img = Metrics.tensor2img(visuals['restored'])  # uint8
                        target_img = Metrics.tensor2img(visuals['targetImg'])  # uint8
                        input_img = Metrics.tensor2img(visuals['valImg'])  # uint8

                        # generation
                        Metrics.save_img(
                            target_img, '{}/{}_{}_target.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            restore_img, '{}/{}_{}_restore.png'.format(result_path, current_step, idx))
                        Metrics.save_img(
                            input_img, '{}/{}_{}_input.png'.format(result_path, current_step, idx))
                        tb_logger.add_image('Iter_{}'.format(current_step),
                            np.transpose(np.concatenate((input_img, restore_img, target_img),
                                                        axis=1), [2, 0, 1]),idx)
                        avg_psnr += Metrics.calculate_psnr(
                            restore_img, target_img)
                        avg_ssim += Metrics.calculate_ssim(restore_img, target_img)
                        avg_LPIPS+=Metrics.getLPIPS(visuals['restored'], visuals['targetImg'])

                        if wandb_logger:
                            wandb_logger.log_image(
                                f'validation_{idx}',
                                np.concatenate((input_img, restore_img, target_img), axis=1)
                            )

                    avg_psnr = avg_psnr / idx
                    avg_ssim= avg_ssim / idx
                    avg_LPIPS= avg_LPIPS / idx
                    # log
                    logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
                    logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
                    logger.info('# Validation # LPIPS: {:.4e}'.format(avg_LPIPS))
                    logger_val = logging.getLogger('val')  # validation logger
                    logger_val.info('<epoch:{:3d}, iter:{:8d}, lr:{:.4e}> psnr: {:.4e}, ssim: {:.4e}, LPIPS: {:.4e}'.format(
                        current_epoch, current_step,diffusion.get_current_learning_rate(), avg_psnr, avg_ssim, avg_LPIPS))
                    # tensorboard logger
                    tb_logger.add_scalar('psnr', avg_psnr, current_step)
                    tb_logger.add_scalar('ssim', avg_ssim, current_step)
                    tb_logger.add_scalar('LPIPS', avg_LPIPS, current_step)

                    if wandb_logger:
                        wandb_logger.log_metrics({
                            'validation/val_psnr': avg_psnr,
                            'validation/val_ssim': avg_ssim,
                            'validation/val_LPIPS': avg_LPIPS,
                            'validation/val_step': val_step,

                        })
                        val_step += 1
                    if ema_swapped:
                        diffusion.netG.load_state_dict(original_state)

            if wandb_logger:
                wandb_logger.log_metrics({'epoch': current_epoch-1})

        # save model
        logger.info('End of training.')
    else:
        logger.info('Begin Model Evaluation.')
        avg_psnr = 0.0
        avg_ssim = 0.0
        avg_LPIPS = 0.0
        idx = 0
        result_path = '{}'.format(opt['path']['results'])
        os.makedirs(result_path, exist_ok=True)

        if hasattr(diffusion, 'ema') and diffusion.ema is not None:
            diffusion.ema.copy_params_from_ema_to_model()

        for _,  val_data in enumerate(val_loader):
            idx += 1
            diffusion.feed_data(val_data)
            diffusion.test(continous=False)
            visuals = diffusion.get_current_visuals()

            target_img = Metrics.tensor2img(visuals['targetImg'])  # uint8
            input_img = Metrics.tensor2img(visuals['valImg'])  # uint8
            restore_img = Metrics.tensor2img(visuals['restored'])  # uint8
            Metrics.save_img(
                restore_img, '{}/{}_{}_sr.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                target_img, '{}/{}_{}_target.png'.format(result_path, current_step, idx))
            Metrics.save_img(
                input_img, '{}/{}_{}_input.png'.format(result_path, current_step, idx))

            # generation
            eval_psnr = Metrics.calculate_psnr(restore_img, target_img)
            eval_ssim = Metrics.calculate_ssim(restore_img, target_img)
            eval_LPIPS = Metrics.getLPIPS(visuals['restored'], visuals['targetImg'])

            avg_psnr += eval_psnr
            avg_ssim += eval_ssim
            avg_LPIPS+=eval_LPIPS

            if wandb_logger and opt['log_eval']:
                wandb_logger.log_eval_data(input_img, restore_img, target_img, psnr=eval_psnr, ssim=eval_ssim,LPIPS=eval_LPIPS)

        avg_psnr = avg_psnr / idx
        avg_ssim = avg_ssim / idx
        avg_LPIPS = avg_LPIPS / idx

        # log
        logger.info('# Validation # PSNR: {:.4e}'.format(avg_psnr))
        logger.info('# Validation # SSIM: {:.4e}'.format(avg_ssim))
        logger.info('# Validation # LPIPS: {:.4e}'.format(avg_LPIPS))
        logger_val = logging.getLogger('val')  # validation logger
        logger_val.info('<epoch:{:3d}, iter:{:8,d}> psnr: {:.4e}, ssim: {:.4e}, LPIPS: {:.4e}'.format(
            current_epoch, current_step, avg_psnr, avg_ssim,avg_LPIPS))

        if wandb_logger:
            if opt['log_eval']:
                wandb_logger.log_eval_table()
                print("finish")
            wandb_logger.log_metrics({
                'PSNR': float(avg_psnr),
                'SSIM': float(avg_ssim),
                'LPIPS': float(avg_LPIPS)
            })
