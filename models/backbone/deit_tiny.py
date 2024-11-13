# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
import torch
import torch.nn as nn
from functools import partial
from torchvision import transforms



import timm
class DeiTTiny(nn.Module):
    def __init__(self,
                 args):
        super(DeiTTiny, self).__init__()

        model_name = "deit_tiny_patch16_224.fb_in1k"
        self.model = timm.create_model(
                                       model_name, 
                                       img_size=args.resize_to, 
                                       pretrained=True
                                       )  
        
        self.model.head = nn.Identity()

        self.model.train()

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

        self.feat_proj = nn.Linear(192, args.bev_latent_dim)
        self.patch_size = 16



    def forward(self, x):
        '''
            x: (B, NCAMS, 3, H, W)
        '''
        
        B, NCAM, C, H, W = x.shape
        
        x = x.view(-1, C, H, W)

        #x = (x - self.mean.to(x.device)) / self.std.to(x.device)

        x = self.model.forward_features(x)[:, 1:]           # (B*NCAMS, P, 192)

        x = self.feat_proj(x)                                # (B*NCAMS, P, featdim)
        #x = x.view(B, NCAM, -1, x.shape[-1])                  # B, NCAMS, P, featdim

        h = int(H / self.patch_size)
        w = int(W / self.patch_size)
        x = x.view(-1, h, w, x.shape[-1])                     # B*NCAMS, h, w, feat_dim
        x = x.permute(0, 3, 1, 2)                             # B*NCAMS, feat_dim, h, w

        return {'features': x,  'rgb_flip_index': None}
