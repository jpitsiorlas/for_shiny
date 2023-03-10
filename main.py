

import os
import time
import argparse
import numpy as np

from model.model import ROG
from torch.utils.data import DataLoader
import dataloader
from helpers.helpers import padding,save_image
from helpers.post_processing import run_post_processing


import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from scipy.ndimage import gaussian_filter
import nibabel as nib
import torch.nn.functional as F
import torch.distributed as dist

import warnings
warnings.filterwarnings('ignore')

in_path = "/input"
out_path = "/output"

img_path = in_path
outpath = out_path
weights = "/model/best_dice_shiny.pth.tar"


# initialize the process group
port = str(2)

os.environ['MASTER_ADDR'] = 'localhost'
os.environ['MASTER_PORT'] = '1234' + port

rank = 0
world_size = 1
dist.init_process_group("nccl", rank=rank, world_size=world_size)
torch.cuda.set_device(rank)

#-----------------MODEL-----------------------#
model_params = {"classes": 2, "modalities": 1, "strides": [[2, 2, 1], [2, 2, 2], [2, 2, 2]], "img_size": [192, 192, 96], "in_channels": 1, "out_channels": 2, "feature_size": 48, "use_checkpoint": True}
patch_size = [212,212,96]
val_size = [256,256,96]

model = ROG(model_params).to(rank)
ddp_model = DDP(model, device_ids=[rank])

map_location = {"cuda:%d" % 0: "cuda:%d" % rank}
checkpoint = torch.load(
            os.path.join(weights),
            map_location=map_location)
ddp_model.load_state_dict(
            checkpoint["state_dict"])
ddp_model.eval()
w_patch = np.zeros(val_size)
sigmas = np.asarray(val_size) // 8
center = torch.Tensor(val_size) // 2   #warning
w_patch[tuple(center.long())] = 1
w_patch = gaussian_filter(w_patch, sigmas, 0, mode="constant", cval=0)
w_patch = torch.Tensor(w_patch / w_patch.max()).to(rank).half()

#-----------------Dataloader-----------------------#
paths= [os.path.join(root, filename) for root, _, files in os.walk(img_path) for filename in files]
test_dataset = dataloader.Medical_data(False, paths, val_size)
test_loader = DataLoader(test_dataset, sampler=None, shuffle=False, batch_size=1, num_workers=0)

#-----------------INFERENCE-----------------------#
for idx in range(len(paths)):
    shape, name, affine, pad = test_loader.dataset.update(idx)
    prediction = torch.zeros((model_params['classes'],) + shape).to(rank).half()
    weights = torch.zeros(shape).to(rank).half()

    for sample in test_loader:
        data = sample['data'].float()  # .squeeze_(0)
        with torch.no_grad():
            output = model(data.to(rank))
        output[0][0] *= w_patch
        output[1][0] *= w_patch

        low = (sample['target'][0] - center).long() 
        up = (sample['target'][0] + center).long()
        prediction[:, low[0]:up[0], low[1]:up[1], low[2]:up[2]] += output[0][0]
        weights[low[0]:up[0], low[1]:up[1], low[2]:up[2]] += w_patch

    # Vessels Prediction
    prediction /= weights
    prediction = F.softmax(prediction, dim=0)
    prediction_logits = torch.nan_to_num(prediction[1]).cpu()
    prediction = torch.argmax(prediction, dim=0).cpu()

    
    if pad is not None:
        prediction = padding(prediction, pad, shape)
        prediction_logits = padding(prediction_logits, pad, shape)

    # Save argmax predictions
    #save_image(prediction, os.path.join(outpath, name +'.nii.gz'), affine)
    post_pred = run_post_processing(prediction,kernel_size=2,conn_comp=1)

    # Save post-processed prediction
    save_image(post_pred, os.path.join(outpath, name +'_post.nii.gz'), affine)

    print('Prediction {} saved'.format(name))


