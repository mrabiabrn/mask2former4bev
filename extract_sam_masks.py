import os
import sys
import pickle 

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np

import matplotlib.pyplot as plt

sys.path.append('..')

from PIL import Image
from lang_sam import LangSAM

class Arguments:

    def __init__(self):
        
        self.project = 'mask2former4bev'
        self.model_name = 'mask2former4bev'
        self.dataset_path = '/datasets/nuscenes'
        self.version = 'trainval'

        # write all the parameters like above
        self.res_scale = 1
        self.H = 1600
        self.W = 900
        self.rand_crop_and_resize = 0
        
        self.resize_to = [224,400]
        self.crop_offset = 0
        self.random_flip = 0
        self.resize_lim = [1.0, 1.0]
        self.cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        self.ncams = 6

        self.do_shuffle_cams = 0
        self.refcam_id = 1

        self.voxel_size = [200, 8, 200]
        self.bounds = [-50, 50, -5, 5, -50, 50]
        
        self.num_queries = 100
        self.get_sem_masks = 0
        
        
args = Arguments()

langsam = LangSAM()

print('Device:', langsam.device)

from dataset import NuScenesDatasetWrapper

datamodule = NuScenesDatasetWrapper(args)

valset = datamodule.val()
trainset = datamodule.train()

from torchvision.transforms.functional import to_pil_image

text_prompt = "vehicle"

print('There are', len(trainset), 'training samples and', len(valset), 'validation samples')


for i in range(20670,len(trainset)):
    sample = trainset[i]

    # if mask is in the dir, continue
    #if os.path.exists(f'sam_masks/train/{i}.npz'):
    #    continue
    
    cam_masks = []
    for c in range(args.ncams):
        image_pil = to_pil_image(sample['images'][c], 'RGB')
        masks = langsam.predict(image_pil, text_prompt)[0]      # M, H, W

        if masks == [] or masks.shape[0] == 0:
            sum_masks = torch.zeros((args.resize_to[0], args.resize_to[1]))
        else:
            sum_masks = torch.sum(masks, dim=0) > 0               # H, W

        cam_masks.append(sum_masks)

    cam_masks = torch.stack(cam_masks, dim=0)                # C, H, W

    # save the masks as npz
    np.savez(f'sam_masks/train/{i}.npz', masks=cam_masks)

    if i % 100 == 0:
        print(f'Processed {i} samples')


for i in range(1300,len(valset)):
    sample = valset[i]

    # if mask is in the dir, continue
    # if os.path.exists(f'sam_masks/val/{i}.npz'):
    #     continue
    
    cam_masks = []
    for c in range(args.ncams):
        image_pil = to_pil_image(sample['images'][c], 'RGB')
        masks = langsam.predict(image_pil, text_prompt)[0]
        
        if masks == [] or masks.shape[0] == 0:
            sum_masks = torch.zeros((args.resize_to[0], args.resize_to[1]))
        else:
            sum_masks = torch.sum(masks, dim=0) > 0               # H, W

        cam_masks.append(sum_masks)

    cam_masks = torch.stack(cam_masks, dim=0)                # C, H, W

    # save the masks as npz
    np.savez(f'sam_masks/val/{i}.npz', masks=cam_masks)

    if i % 100 == 0:
        print(f'Processed {i} samples')

print('Done!')