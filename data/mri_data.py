import torch
import torch.utils.data as data
import matplotlib.pyplot as plt
import os
from torchvision import transforms
import numpy as np
import nibabel as nib
from skimage.transform import rotate
import random
import math
import functools
import json
import copy
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset

import scipy
from scipy.ndimage import affine_transform, map_coordinates

class ToTensor(object):
    """Convert ndarrays in sample to Tensors. Modifiy code from pytorch offical website"""
    def __call__(self, sample):

        return  torch.from_numpy(sample).type(torch.FloatTensor)

class MRI(data.Dataset):

    def __init__(self, opt):
        BaseDataset.__init__(self, opt)
        self.transform  = transforms.Compose([ToTensor()])
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.data_paths = [o for o in os.listdir(self.dir)  if os.path.isdir(os.path.join(self.dir, o))]
        self.length = len(self.data_paths)
        self.phase = opt.phase

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # A to B
        running_instance = os.path.join(self.dir,self.data_paths[index])
        groundTruth = 'T1c.nii'
        # inputSeq = ['apt.nii','T1.nii','T2.nii','FLAIR.nii']
        inputSeq = ['T1.nii']
        n = len(inputSeq)
        inputArray = np.zeros([n,256,256]).astype(np.float32)
        gtArray= np.zeros([1,256,256]).astype(np.float32)
        
        degree = random.random()*360

        # Ground Truth
        gtPath = os.path.join(running_instance,self.data_paths[index] + '_' + groundTruth) 
        target = nib.load(sequence_path)
        array = target.get_fdata().astype(np.float32)

        array = array * 2 - 1
        gtArray[0,:,:] = array
        filename = self.data_paths[index] + '_' + 'gt'
        B_paths = os.path.join(running_instance, filename)
        np.save(B_paths, gtArray) 

        # Input Sequences
        if n == 1:
            sequence_path = os.path.join(running_instance,self.data_paths[index]+'_'+sequence) 
            img = nib.load(sequence_path)
            data = img.get_fdata().astype(np.float32)
            data = data*2-1
            inputArray[0,:,:] = data
        else:
            for sequence in inputSeq:
                if sequence == 'apt.nii':
                    count = 0
                elif sequence == 'T1.nii':
                    count = 1
                elif sequence == 'T2.nii':
                    count = 2
                elif sequence == 'FLAIR.nii':
                    count = 3
                sequence_path = os.path.join(running_instance,self.data_paths[index]+'_'+sequence) 
                img = nib.load(sequence_path)
                data = img.get_fdata().astype(np.float32)
                data = data*2-1
                inputArray[count,:,:] = data

        filename = self.data_paths[index] + '_' + 'input'
        A_paths = os.path.join(running_instance, filename)
        np.save(A_paths, inputArray) 

        # apply the same transform to both A and B
        # TODO: change transformation for input=4
        A = inputArray
        B = gtArray
        transform_params = get_params(self.opt, B.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(A)
        B = B_transform(B)

        data = {'A':A, 'B':B, 'A_paths': A_paths ,'B_paths': B_paths}

        return data