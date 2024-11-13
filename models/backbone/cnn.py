import torch.nn as nn

class CNNEncoder(nn.Module):
    def __init__(self, args):
        super(CNNEncoder, self).__init__()
        self.args = args

        self.model = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32*4, 32*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32*4, 64*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64*4, 64*4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64*4, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, args.bev_latent_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        #self.feat_proj = nn.Conv2d(768, args.bev_latent_dim, kernel_size=1, stride=1, padding=0)


    def forward(self, x):

        _, _, C, H, W = x.shape
        
        x = x.view(-1, C, H, W)

        x = self.model(x)  # (B, C, H, W)

        return {'features': x, 'rgb_flip_index': None}