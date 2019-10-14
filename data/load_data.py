from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import torch

from transforms.spatial_transforms import Normalize
from torch.utils.data import DataLoader

from data.mri_data import MRI

def getData(opt, dataType):
    assert opt.dataset in ['MriPost', 'MriPre']
    if dataType == 'train':
        if opt.dataset == 'MriPost':
            data = MRI(opt)
    elif dataType == 'test':
        if opt.dataset == 'MriPost':
            data = MRI(opt)
    elif dataType == 'validation':
        if opt.dataset == 'MriPost':
            data = MRI(opt)
    return data

##########################################################################################
##########################################################################################

def get_normalization_method(config):
    if config.no_mean_norm and not config.std_norm:
        return Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    elif not config.std_norm:
        return Normalize(config.mean, [0.5, 0.5, 0.5])
    else:
        return Normalize(config.mean, config.std)

##########################################################################################
##########################################################################################

def getDataloader(opt):

    print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))

    dataLoader = dict()

    # Training Dataset
    datasetTrain = getData(opt, 'train')
    dataLoader['train'] = DataLoader(
        datasetTrain, opt.batchSize, shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads))

    # Testing Dataset
    datasetTest = getData(opt, 'test')
    dataLoader['test'] = DataLoader(
        datasetTest, opt.batchSize, shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads))

    if not opt.no_eval:
        # Validation Dataset
        datasetVal = getData(opt, 'validation')
        dataLoader['validation'] = DataLoader(
            datasetVal, opt.batchSize, shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))
        print('Found {} validation examples'.format(len(datasetVal)))

    print('Found {} training examples'.format(len(datasetTrain)))
    print('Found {} testing examples'.format(len(datasetTest)))

    return dataLoader