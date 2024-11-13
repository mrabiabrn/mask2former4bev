import torch.nn as nn
import timm


class SwinEncoder(nn.Module):
    def __init__(self, args):
        super(SwinEncoder, self).__init__()
        self.args = args

        #self.model = timm.create_model('swin_base_patch4_window7_224.ms_in22k', pretrained=True)
        self.model = timm.create_model('swinv2_base_window12_192.ms_in22k', pretrained=True)

        self.feat_proj = nn.Conv2d(768, args.bev_latent_dim, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        _, _, C, H, W = x.shape
        
        x = x.view(-1, C, H, W)

        x = self.model.forward_features(x)  # (B, C, H, W)

        print(x.shape)
        exit()

        return {'features': x, 'rgb_flip_index': None}
