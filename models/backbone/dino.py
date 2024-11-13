import torch
import torch.nn as nn

import timm
from .lora_utils import LoRA_ViT_timm




class DINOEncoder(nn.Module):
    def __init__(self, args):
        super(DINOEncoder, self).__init__()


        self.dino = timm.create_model(
                                      "vit_base_patch14_dinov2", 
                                      img_size=args.resize_to, 
                                      pretrained=True, 
                                      num_classes=0
                                      )
        #self.dino = LoRA_ViT_timm(self.dino, r=32) 

        self.patch_size = 14

        self.feat_proj = nn.Linear(768, args.bev_latent_dim)

    
    def forward(self, x):

        B,N,C,H,W = x.shape        

        x = x.reshape(B*N,C,H,W)                  # B*N, C, H, W
        
        x = self.dino(x).to(x.device)             # B*N, P, D

        x = self.feat_proj(x)                      # B*N, P, featdim

        h = int(H / self.patch_size)
        w = int(W / self.patch_size)
        x = x.view(-1, h, w, x.shape[-1])                     # B*NCAMS, h, w, feat_dim
        x = x.permute(0, 3, 1, 2)                             # B*NCAMS, feat_dim, h, w

        return {'features': x, 'rgb_flip_index': None}