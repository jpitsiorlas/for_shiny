# -*- coding: utf-8 -*-
import os
import itertools
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage


def load_image(patient,train):
    im = nib.load(patient)
    affine = im.affine
    im = im.get_fdata()
    if len(im.shape) > 3:
        im = np.transpose(im, (3, 0, 1, 2))
    
    return im, affine


def image_shape(im):
    im_shape = im.shape
    multimodal = False
    if len(im_shape) == 4:
        im_shape = im_shape[1:]
        multimodal = True
    return im_shape, multimodal


def test_voxels(patch_size, im_shape):
    """ Select the central voxels of the patches for testing """
    center = patch_size // 2
    dims = []
    for i, j in zip(im_shape, center):
        end = i - j
        num = np.ceil((end - j) / j)
        if num == 1:
            num += 1
        if num == 0:
            dims.append([i // 2])
            continue
        voxels = np.linspace(j, end, int(num))
        dims.append(voxels)
    voxels = list(itertools.product(*dims))
    return voxels


def val_voxels(im_shape, patch_size, label):
    low = patch_size // 2 - 1
    high = np.asarray(im_shape) - low
    pad = tuple(zip(low, low))
    mask = np.pad(np.ones(high - low), pad, 'constant')

    if (mask * label).sum() > 0:
        voxel = np.asarray(ndimage.measurements.center_of_mass(mask * label))
    else:
        voxel = np.asarray(ndimage.measurements.center_of_mass(label))
        nonzero = np.argwhere(mask == 1)
        distances = np.sqrt((nonzero[:, 0] - voxel[0]) ** 2 +
                            (nonzero[:, 1] - voxel[1]) ** 2 +
                            (nonzero[:, 2] - voxel[2]) ** 2)
        nearest_index = np.argmin(distances)
        voxel = nonzero[nearest_index]
    return voxel.astype(int)


def train_voxels(image, patch_size, label, foreground):
    """ Select the central voxels of the patches for testing """
    # Lower and upper bound to sample the central voxel (to avoid padding if
    # it is too close to the borders)
    im_shape, _ = image_shape(image)

    low = patch_size // 2
    high = np.asarray(im_shape) - low

    if foreground:
        # Force the center voxel to belong to a foreground category
        pad = tuple(zip(low, low))
        mask = np.pad(np.zeros(high - low), pad, 'constant',
                      constant_values=-1)

        np.copyto(mask, label, where=(mask == 0))
        fg = np.unique(mask)[2:]  # [ignore, bg, fg...]
        if fg.size > 0:
            cat = np.random.choice(fg)
            selected = np.argwhere(mask == cat)
            coords = selected[np.random.choice(len(selected))]
        else:
            x = np.random.randint(low[0], high[0])
            y = np.random.randint(low[1], high[1])
            z = np.random.randint(low[2], high[2])
            coords = (x, y, z)
    else:
        x = np.random.randint(low[0], high[0])
        y = np.random.randint(low[1], high[1])
        z = np.random.randint(low[2], high[2])
        coords = (x, y, z)
    return coords


def extract_patch(image, voxel, patch_size):
    im_shape, multimodal = image_shape(image)

    v1 = np.maximum(np.asarray(voxel) - patch_size // 2, 0)
    v1 = v1.astype(int)
    v2 = np.minimum(v1 + patch_size, im_shape)

    if multimodal:
        patch = image[:, v1[0]:v2[0], v1[1]:v2[1], v1[2]:v2[2]]
    else:
        patch = np.expand_dims(image[v1[0]:v2[0], v1[1]:v2[1], v1[2]:v2[2]], 0)
    patch, _ = verify_size(patch, patch_size)
    return patch


def verify_size(im, size):
    """ Verify if the patches have the correct size (if they are extracted
    from the borders they may be smaller) """
    im_shape, multimodal = image_shape(im)

    dif = np.asarray(size) - im_shape
    pad = None
    if any(dif > 0):
        dif = np.maximum(dif, [0, 0, 0])
        mod = dif % 2
        pad_1 = dif // 2
        pad_2 = pad_1 + mod
        if multimodal:
            pad_1 = [0] + pad_1.tolist()
            pad_2 = [0] + pad_2.tolist()
        pad = tuple(zip(pad_1, pad_2))
        im = np.pad(im, pad, 'reflect')
    return im, pad


def save_image(prediction, outpath, affine):
    case_path = '/'.join(outpath.split('/')[:-1])
    if not os.path.exists(case_path):
        os.makedirs(case_path)
    if isinstance(prediction, np.ndarray):
        new_pred = nib.Nifti1Image(prediction, affine)
    else:
        new_pred = nib.Nifti1Image(prediction.numpy(), affine)
    new_pred.set_data_dtype(np.uint8)
    nib.save(new_pred, outpath)


def padding(prediction,pad, shape):
    return prediction[pad[0][0]:shape[0] - pad[0][1],
                      pad[1][0]:shape[1] - pad[1][1],
                      pad[2][0]:shape[2] - pad[2][1]]