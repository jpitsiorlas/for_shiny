# -*- coding: utf-8 -*-
import os
import time
import torch
import numpy as np
import pandas as pd
import nibabel as nib
from torch.utils.data import Dataset

from helpers import helpers

import torch.nn as nn
import torch.nn.functional as F


class Medical_data(Dataset):
    def __init__(self, train, paths, patch_size, im_path=None,
                 val=False, pgd=False):
        super(Medical_data, self).__init__()

        self.filenames = paths 
        self.train = train
        self.val = val
        self.patch_size = np.asarray(patch_size)
        self.fg = 0
        self.pgd = pgd
        self.images_path = im_path
        self.folder = ''

    def __len__(self):
        if self.train or self.pgd:
            return len(self.filenames)
        else:
            return len(self.voxel)

    def __getitem__(self, idx):
        if self.train or self.pgd:
            patient = self.filenames[int(idx)]
            image, affine = helpers.load_image(patient, self.train)
            im_shape, multimodal = helpers.image_shape(image)

            # If the image is smaller than the patch_size in any dimension, we
            # have to pad it to extract a patch
            if any(im_shape <= self.patch_size):
                dif = (self.patch_size - im_shape) // 2 + 3
                pad = np.maximum(dif, [0, 0, 0])

                if multimodal:
                    pad_im = [0] + pad.tolist()
                else:
                    pad_im = pad
                pad_im = tuple(zip(pad_im, pad_im))
                image = np.pad(image, pad_im, 'reflect')

        patches = helpers.extract_patch(
            self.image, self.voxel[idx], self.patch_size)
        label = torch.Tensor(self.voxel[idx])
        patches = torch.from_numpy(patches)
        info = 0

        return {'data': patches,'target':label, 'info': info}

    def change_epoch(self):
        self.fg = 1 - self.fg

    def update(self, im_idx):
        # This is only for testing
        patient = self.filenames[im_idx]
        name = patient.split('/')[-1].split('.')[0]
        print('Loading data of patient {} ---> {}'.format(
            name, time.strftime("%H:%M:%S")))

        image, affine = helpers.load_image(
            patient, self.train)
        self.image, pad = helpers.verify_size(image, self.patch_size)
        im_shape, multimodal = helpers.image_shape(self.image)
        if multimodal and pad is not None:
            pad = pad[1:]

        self.voxel = helpers.test_voxels(self.patch_size, im_shape)
        return im_shape, name, affine, pad

