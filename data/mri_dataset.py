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
from data.base_dataset import BaseDataset, get_transform
#from data.image_folder import make_dataset

#from datasets.rotations import x_rotmat, y_rotmat, z_rotmat
#from datasets.transformations import rotation_matrix, translation_matrix
import scipy
from scipy.ndimage import affine_transform, map_coordinates

class ToTensor(object):
    """Convert ndarrays in sample to Tensors. Modifiy code from pytorch offical website"""

    def __call__(self, sample):

        return  torch.from_numpy(sample).type(torch.FloatTensor)

class MRI(data.Dataset):
    """
    Args:
        root (string): Root directory path. 'C:\\Users\\hkwon28\\Desktop\\MRI_Project\\mri-gan\\datasets\\data\\training_raw'
        spatial_transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        temporal_transform (callable, optional): A function/transform that  takes in a list of frame indices
            and returns a transformed version
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an video given its path and frame indices.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    # def __init__(self,
    #              dataroot,
    #              phase,
    #              spatial_transform=None,
    #              temporal_transform=None,
    #              target_transform=None):
    def __init__(self, opt,     spatial_transform=None, temporal_transform=None,target_transform=None):
        BaseDataset.__init__(self, opt)

        self.transform  = transforms.Compose([ToTensor()])
        self.dir = os.path.join(opt.dataroot, opt.phase)
        self.data_paths = [o for o in os.listdir(self.dir)  if os.path.isdir(os.path.join(self.dir, o))]
        self.length = len(self.data_paths)
        self.phase = opt.phase


        # # Get all subfolders
        # self.subset = subset
        # self.root = os.path.join(root_path, subset)
        # self.label_folders = [o for o in os.listdir(self.root) if os.path.isdir(os.path.join(self.root, o))]
        # class1 = [(o,self.label_folders[0])
        #           for o in os.listdir(os.path.join(self.root, self.label_folders[0]))
        #           if o.endswith('T2.nii')]
        # class2 = [(o, self.label_folders[1])
        #           for o in os.listdir(os.path.join(self.root, self.label_folders[1]))
        #           if o.endswith('T2.nii')]

        # self.image_folders = class1+class2
        # self.length = len(self.image_folders)


    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        # (img,label) = self.image_folders[index]
        # img_parts = img.split('_')
        # running_instance = os.path.join(self.root,label,img_parts[0]+'_'+img_parts[1]+'_')
        running_instance = os.path.join(self.dir,self.data_paths[index])
        array_3d = np.zeros([4,256,256]).astype(np.float32)
        target_array= np.zeros([1,256,256]).astype(np.float32)
        sequences = ['apt.nii','T1.nii','T1c.nii','T2.nii','FLAIR.nii']
        degree = random.random()*360

        for sequence in sequences:
            if sequence == 'T1c.nii':
                sequence_path = os.path.join(running_instance,self.data_paths[index]+'_'+sequence) 
                target = nib.load(sequence_path)
                array = target.get_fdata().astype(np.float32)
                #if self.phase == 'train':
                 #   array = rotate(array, degree, resize=False)
                array = array * 2 - 1
                target_array[0,:,:] = array
                filename = self.data_paths[index] + '_' + 'gt'
                B_paths = os.path.join(running_instance, filename)
                np.save(B_paths, target_array) 
                
            else:
                if sequence == 'apt.nii':
                    count = 0
                elif sequence == 'T1.nii':
                    count = 1
                elif sequence == 'T2.nii':
                    count = 2
                elif sequence == 'FLAIR.nii':
                    count = 3
                    #print('INSIDE')
                sequence_path = os.path.join(running_instance,self.data_paths[index]+'_'+sequence) 
                img = nib.load(sequence_path)
                data = img.get_fdata().astype(np.float32)

                #if self.phase == 'train':
                   # data = rotate(data, degree, resize=False)
                data = data*2-1
                array_3d[count,:,:] = data
        filename = self.data_paths[index] + '_' + 'input'
        A_paths = os.path.join(running_instance, filename)
        #print('array_3d: ' + str(array_3d.shape))
        np.save(A_paths, array_3d) 
                
                #print('A_paths: ' + str(A_paths))

        #print('arra_3d shape: ' + str(array_3d.shape))
        A = self.transform(array_3d)
        #B = self.transform(array)
        B = self.transform(target_array)

        data = {'A':A, 'B':B, 'A_paths': A_paths ,'B_paths': B_paths}

        return data
