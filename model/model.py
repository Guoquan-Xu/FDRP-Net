import logging
from collections import OrderedDict
import torch.nn.functional as F
import torch
import torch.nn as nn
import os
import model.networks as networks
from .base_model import BaseModel
from datetime import datetime
from ema_pytorch import EMA
logger = logging.getLogger('base')



class DDPM(BaseModel):
    def __init__(self, opt,device):
        super(DDPM, self).__init__(opt)

        self.netG = self.set_device(networks.define_G(opt))
        self.schedule_phase = None
        if self.opt['phase'] == 'train':
            self.netG.train()
            optim_params = list(self.netG.parameters())
            self.optG=torch.optim.AdamW(
                optim_params,
                lr=opt['train']['optimizer']['lr'],
                betas=(opt['train']['optimizer']['betas'][0],opt['train']['optimizer']['betas'][1]),
            )
            self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optG,
                T_max=opt['train']['n_iter'],
                eta_min=opt['train']['optimizer']['eta_min']
            )

            self.log_dict = OrderedDict()

        ema_decay = opt['train'].get('ema_decay', 0.999)
        ema_update_after = opt['train'].get('ema_update_after_step', 0)
        ema_update_every = opt['train'].get('ema_update_every', 1)

        self.ema = EMA(
            self.netG,
            beta=ema_decay,
            update_after_step=ema_update_after,
            update_every=ema_update_every
        )

        self.load_network()
        self.print_network()


    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad


    def get_current_learning_rate(self):
        return self.optG.param_groups[0]['lr']

    def feed_data(self, data):
        self.data = self.set_device(data)

    def check_gradients(self,model):
        NI=False
        nan_inf_params = []
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    print(f"NaN/Inf gradient in {name}")
                    nan_inf_params.append(name)
                    NI=True

        if NI:
            log_dir = None
            if hasattr(self, 'opt') and 'path' in self.opt and 'log' in self.opt['path']:
                log_dir = self.opt['path']['log']
            else:
                # 若 self 中有 log_dir 属性也可用
                log_dir = getattr(self, 'log_dir', './logs')

            log_file = os.path.join(log_dir, 'gradient_errors.log')
            os.makedirs(log_file, exist_ok=True)

            with open(log_file, 'a') as f:
                f.write(f"\n--- NaN/Inf gradients detected at {datetime.now()} ---\n")
                for p in nan_inf_params:
                    f.write(f"NaN/Inf gradient in {p}\n")
                f.write("--- End of error list ---\n")

        assert NI==False

    def optimize_parameters(self, flag=None):
        self.optG.zero_grad()
        loss = self.netG(self.data['trainImg'],self.data['referenceImg'])
        loss.backward()

        self.check_gradients(self.netG.model)

        self.optG.step()
        self.scheduler.step()

        if hasattr(self, 'ema') and self.ema is not None:
            self.ema.update()

        self.log_dict['loss'] = loss.item()

        return self.log_dict['loss']


    def optimize_parameters2(self, flag=None):
        if flag is None:
            self.optG.zero_grad()
            loss = self.netG(self.data['trainImg'],self.data['referenceImg'])
            b, c, h, w = self.data['trainImg'].shape
            loss = loss.sum() / int(b * c * h * w)
            loss.backward()
            self.optG.step()
            self.scheduler.step()


            self.log_dict['loss'] = loss.item()



    def test(self, cand=None, continous=False):
        self.netG.eval()
        with torch.no_grad():
            if isinstance(self.netG, nn.DataParallel):
                self.SR = self.netG.module.super_resolution(
                    self.data['trainImg'], continous)
            else:
                self.SR = self.netG.super_resolution(
                    self.data['trainImg'],continous)

        self.netG.train()

    def set_new_noise_schedule(self, schedule_opt, schedule_phase='train'):
        if self.schedule_phase is None or self.schedule_phase != schedule_phase:
            self.schedule_phase = schedule_phase
            if isinstance(self.netG, nn.DataParallel):
                self.netG.module.set_noise_schedule(schedule_opt)
            else:
                self.netG.set_noise_schedule(schedule_opt)


    def get_current_log(self):
        return self.log_dict

    def get_current_visuals(self,sample=False):
        out_dict = OrderedDict()
        if sample:
            out_dict['restored'] = self.SR.detach().float().cpu()
            out_dict['valImg'] = self.data['trainImg'].detach().float().cpu()
        else:
            out_dict['restored'] = self.SR.detach().float().cpu()
            out_dict['targetImg']=self.data['referenceImg'].detach().float().cpu()
            out_dict['valImg'] = self.data['trainImg'].detach().float().cpu()
        return out_dict

    def print_network(self):
        s, n = self.get_network_description(self.netG)
        if isinstance(self.netG, nn.DataParallel):
            net_struc_str = '{} - {}'.format(self.netG.__class__.__name__,
                                             self.netG.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netG.__class__.__name__)

        logger.info(
            'Network G structure: {}, with parameters: {:,d}'.format(net_struc_str, n))
        logger.info(s)

    def save_network(self, epoch, iter_step):
        gen_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_gen.pth'.format(iter_step, epoch))
        opt_path = os.path.join(
            self.opt['path']['checkpoint'], 'I{}_E{}_opt.pth'.format(iter_step, epoch))
        # gen
        network = self.netG
        if isinstance(self.netG, nn.DataParallel):
            network = network.module
        state_dict = network.state_dict()
        for key, param in state_dict.items():
            state_dict[key] = param.cpu()
        torch.save(state_dict, gen_path)
        # opt
        opt_state = {
            'epoch': epoch,
            'iter': iter_step,
            'current_iter': iter_step,  # 保存当前迭代次数
            'scheduler': self.scheduler.state_dict() if hasattr(self, 'scheduler') else None,
            'optimizer': self.optG.state_dict(),
            'EMA':self.ema.state_dict(),
        }
        torch.save(opt_state, opt_path)

        logger.info(
            'Saved model in [{:s}] ...'.format(gen_path))

    def load_network(self):
        load_path = self.opt['path']['resume_state']
        if load_path is not None:
            logger.info(
                'Loading pretrained model for G [{:s}] ...'.format(load_path))
            gen_path = '{}_gen.pth'.format(load_path)
            opt_path = '{}_opt.pth'.format(load_path)
            # gen
            network = self.netG
            if isinstance(self.netG, nn.DataParallel):
                network = network.module
            network.load_state_dict(torch.load(
                gen_path), strict=(not self.opt['model']['finetune_norm']))

            opt = torch.load(opt_path)
            if self.opt['phase'] == 'train':
                # optimizer
                self.optG.load_state_dict(opt['optimizer'])
                self.begin_step = opt['iter']
                self.begin_epoch = opt['epoch']

                if 'current_iter' in opt:
                    self.current_iter = opt['current_iter']
                else:
                    self.current_iter = opt['iter']

                if 'scheduler' in opt and False:
                    self.scheduler.load_state_dict(opt['scheduler'])
                else:
                    self._reset_scheduler()

            if 'EMA' in opt:
                self.ema.load_state_dict(opt['EMA'])

    def _reset_scheduler(self):
        if hasattr(self.scheduler, 'T_max'):  # CosineAnnealingLR 特有的属性
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optG,
                T_max=self.scheduler.T_max,
                eta_min=self.scheduler.eta_min,
                last_epoch=self.current_iter
            )
        else:
            lr_lambda = self.scheduler.lr_lambdas[0] if hasattr(self.scheduler, 'lr_lambdas') else None
            if lr_lambda is not None:
                self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                    self.optG,
                    lr_lambda=lr_lambda,
                    last_epoch=self.current_iter
                )
