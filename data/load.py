from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import torch

from transforms.spatial_transforms import Normalize
from torch.utils.data import DataLoader


from data.mri_dataset import MRI

##########################################################################################
##########################################################################################

def get_training_set(opt, spatial_transform, temporal_transform, target_transform):

    assert opt.dataset in ['kinetics', 'activitynet', 'ucf101', 'blender','MRI_Post']


    if opt.dataset == 'MRI_Post':

        training_data = MRI(
            opt,
            spatial_transform=None,
            temporal_transform=None,
            target_transform=None)


    return training_data


##########################################################################################
##########################################################################################

def get_validation_set(opt, spatial_transform, temporal_transform, target_transform):

    assert opt.dataset in ['MRI_Post','kinetics', 'activitynet', 'ucf101', 'blender']

    # Disable evaluation
    if opt.no_eval:
        return None

    if opt.dataset == 'MRI_Post':

        validation_data = MRI(
            opt,
            spatial_transform=None,
            temporal_transform=None,
            target_transform=None)


    return validation_data

##########################################################################################
##########################################################################################

def get_test_set(config, spatial_transform, temporal_transform, target_transform):

    assert config.dataset in ['kinetics', 'activitynet', 'ucf101', 'blender']
    assert config.test_subset in ['val', 'test']

    if config.test_subset == 'val':
        subset = 'validation'
    elif config.test_subset == 'test':
        subset = 'testing'

    if config.dataset == 'kinetics':

        test_data = Kinetics(
            config.video_path,
            config.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    elif config.dataset == 'activitynet':

        test_data = ActivityNet(
            config.video_path,
            config.annotation_path,
            subset,
            True,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    elif config.dataset == 'ucf101':

        test_data = UCF101(
            config.video_path,
            config.annotation_path,
            subset,
            0,
            spatial_transform,
            temporal_transform,
            target_transform,
            sample_duration=config.sample_duration)

    return test_data


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

def get_data_loaders(opt, train_transforms, validation_transforms=None):

    print('[{}] Preparing datasets...'.format(datetime.now().strftime("%A %H:%M")))

    data_loaders = dict()

    # Define the data pipeline
    dataset_train = get_training_set(
        opt, train_transforms['spatial'],
        train_transforms['temporal'], train_transforms['target'])

    # dataset_train = get_training_set(
    #     opt, train_transforms['spatial'],
    #     train_transforms['temporal'], train_transforms['target'])

    data_loaders['train'] = DataLoader(
        dataset_train, opt.batchSize, shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads))

    dataset_test = get_validation_set(
            opt, validation_transforms['spatial'],
            validation_transforms['temporal'], validation_transforms['target'])

    # dataset_train = get_training_set(
    #     opt, train_transforms['spatial'],
    #     train_transforms['temporal'], train_transforms['target'])

    data_loaders['test'] = DataLoader(
        dataset_test, opt.batchSize, shuffle=not opt.serial_batches,
        num_workers=int(opt.num_threads))



    print('Found {} training examples'.format(len(dataset_train)))

    if not opt.no_eval and validation_transforms:

        dataset_validation = get_validation_set(
            opt, validation_transforms['spatial'],
            validation_transforms['temporal'], validation_transforms['target'])

        print('Found {} validation examples'.format(len(dataset_validation)))

        data_loaders['validation'] = DataLoader(
            dataset_validation, opt.batchSize, shuffle=True,
            num_workers=opt.num_workers, pin_memory=True)

    return data_loaders