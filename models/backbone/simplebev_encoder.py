
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

import torchvision


class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)



class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2, padding=0):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=padding, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip



class Encoder_res101(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.C = args.bev_latent_dim
        resnet = torchvision.models.resnet101(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-4])
        self.layer3 = resnet.layer3

        self.depth_layer = nn.Conv2d(512, self.C, kernel_size=1, padding=0)
        self.upsampling_layer = UpsamplingConcat(1536, 512)

    def forward(self, x):
        x1 = self.backbone(x)
        x2 = self.layer3(x1)
        x = self.upsampling_layer(x2, x1)
        x = self.depth_layer(x)

        return x
    

class SimpleBEVEncoder(nn.Module):
    def __init__(self, args):
        super().__init__()
        
        self.encoder = Encoder_res101(args)

        if args.use_frozen_bev_feats:
            # load encoder weights from frozen_bev_feats model's encoder
            print("Loading encoder weights from frozen_bev_feats model's encoder")
            param_keys = torch.load(args.frozen_bev_feats_path)['model_state_dict']
            # get the encoder weights in a dictionary
            encoder_keys = {k[8:]: v for k, v in param_keys.items() if 'encoder' in k}
            self.encoder.load_state_dict(encoder_keys, strict=False)

        self.freeze_backbone = args.freeze_backbone

        if args.freeze_backbone:
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.encoder.eval()

        self.rand_flip = args.random_flip

        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            

    def forward(self, x, mask):
        '''
            x: (B, NCAMS, 3, H, W)
        '''
        
        _, _, C, H, W = x.shape
        
        x = x.view(-1, C, H, W)
        if mask is not None:
            mask = mask.view(-1, 3, H, W)
            x = x * mask

        x = (x + 0.5 - self.mean.to(x.device)) / self.std.to(x.device) #+ 0.5

        rgb_flip_index = None
        if self.rand_flip:
            B0, _, _, _ = x.shape
            rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
            x[rgb_flip_index] = torch.flip(x[rgb_flip_index], [-1])
 
        if self.freeze_backbone:
            with torch.no_grad():
                return {'features' : self.encoder(x), 'rgb_flip_index': rgb_flip_index}

        return {'features' : self.encoder(x), 'rgb_flip_index': rgb_flip_index}
    



    