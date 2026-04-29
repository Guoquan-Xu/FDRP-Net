import logging
import torch.utils.data as data
from data.prepare_data import prepareDataset
from torchsampler import ImbalancedDatasetSampler

def create_dataLoader(dataset,opt,phase):
    if phase == 'train':
        return data.DataLoader(
            dataset,
            batch_size=opt['batch_size'],
            # shuffle=opt['use_shuffle'],
            num_workers=opt['num_workers'],
            pin_memory=True,
            sampler=ImbalancedDatasetSampler(dataset)
        )
    elif phase == 'val':
        return data.DataLoader(
            dataset,
            batch_size=opt['batch_size'],
            shuffle=False,
            num_workers=opt['num_workers'],
            pin_memory=True
        )
    else:
        raise NotImplemented(
            'Dataloader [{:s}] is not found.'.format(phase)
        )

def create_dataset(opt,phase):
    dataroot={'train':opt['trainroot'],'reference':opt['referenceroot']}
    dataset=prepareDataset(
        dataroot=dataroot,
        img_size=opt['resolution'],
        datatype=opt['datatype'],
        phase=phase,
        data_len=opt['data_len']
    )
    logger = logging.getLogger('base')
    logger.info('Dataset [{:s} - {:s}] is created.'.format(dataset.__class__.__name__,
                                                           opt['name']))
    return dataset