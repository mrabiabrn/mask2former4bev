import numpy as np

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models.resnet import resnet18

from .utils import geom, basic, vox

import fvcore.nn.weight_init as weight_init


EPS = 1e-4


def get_norm(norm, dim):
    if norm == "group":
        return nn.GroupNorm(32, dim)
    elif norm == "batch":
        return nn.BatchNorm2d(dim)
    elif norm == "instance":
        return nn.InstanceNorm2d(dim)
    elif norm == "layer":
        return nn.LayerNorm(dim)
    elif norm == "":
        return None
    else:
        raise ValueError(f"Unknown norm type {norm}")


class SimpleBEVModule(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ms_feat_channels = args.multiscale_feature_channels
        norm = args.multiscale_feature_norm

        conv_dim = args.multiscale_conv_dim
        self.mask_dim = args.mask_dim

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""

        self.use_multiscale_features = args.use_multiscale_features
        if self.use_multiscale_features:
            for idx, in_channels in enumerate(ms_feat_channels):
                print(f'idx: {idx}, in_channels: {in_channels}')
                if idx == len(ms_feat_channels) - 1:                # self.in_features
                    output_norm = get_norm(norm, conv_dim)
                    output_conv = nn.Conv2d(
                        in_channels,
                        conv_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                    )
                    
                    weight_init.c2_xavier_fill(output_conv)
                    # Add normalization and activation
                    output_conv = nn.Sequential(
                        output_conv,
                        output_norm,
                        nn.ReLU(inplace=True)
                    )
                    # set device for output_conv
                    output_conv = output_conv.to(device)

                    self.add_module("layer_{}".format(idx + 1), output_conv)

                    lateral_convs.append(None)
                    output_convs.append(output_conv)
                else:
                    lateral_norm = get_norm(norm, conv_dim)
                    output_norm = get_norm(norm, conv_dim)

                    lateral_conv = nn.Conv2d(
                        in_channels, conv_dim, kernel_size=1, bias=use_bias#, norm=lateral_norm
                    )
                    weight_init.c2_xavier_fill(lateral_conv)
                    lateral_conv = nn.Sequential(
                        lateral_conv,
                        lateral_norm,
                    )
                    lateral_conv = lateral_conv.to(device)
                    output_conv = nn.Conv2d(
                        conv_dim,
                        conv_dim,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                        bias=use_bias,
                    )
                    weight_init.c2_xavier_fill(output_conv)
                    

                    output_conv = nn.Sequential(
                        output_conv,
                        output_norm,
                        nn.ReLU(inplace=True)
                    )

                    # weight_init.c2_xavier_fill(lateral_conv)
                    # weight_init.c2_xavier_fill(output_conv)
                    self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                    self.add_module("layer_{}".format(idx + 1), output_conv)


                    output_conv = output_conv.to(device)
                    
                    lateral_convs.append(lateral_conv)
                    output_convs.append(output_conv)

            # Place convs into top-down order (from low to high resolution)
            # to make the top-down computation in forward clearer.
            self.lateral_convs = lateral_convs[::-1]
            self.output_convs = output_convs[::-1]

            for i in range(len(self.lateral_convs)):
                print(f'lateral_convs {i}: {self.lateral_convs[i]}')
            for i in range(len(self.output_convs)):
                print(f'output_convs {i}: {self.output_convs[i]}')

        self.X, self.Y, self.Z = args.voxel_size
        self.bounds = args.bounds

        scene_centroid_x = 0.0
        scene_centroid_y = 1.0  # down 1 meter
        scene_centroid_z = 0.0

        scene_centroid_py = np.array([scene_centroid_x,
                                    scene_centroid_y,
                                    scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid_py).float()

        self.vox_util = vox.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds,
            assert_cube=False)

        self.xyz_memA = basic.gridcloud3d(1, self.Z, self.Y, self.X, norm=False)
        self.xyz_camA = self.vox_util.Mem2Ref(self.xyz_memA, self.Z, self.Y, self.X, assert_cube=False)

        self.feat_dim = args.bev_latent_dim
        self.do_rgbcompress = args.do_rgb_compress
        self.rand_flip = args.random_flip

        self.segnet = Segnet(
                             Z=self.Z, 
                             Y=self.Y, 
                             X=self.X, 
                             vox_util=self.vox_util, 
                             do_rgbcompress=self.do_rgbcompress, 
                             latent_dim=self.feat_dim,
                             rand_flip=self.rand_flip,
                             )
        
        
        if args.use_frozen_bev_feats :
            param_keys = torch.load(args.frozen_bev_feats_path)['model_state_dict']
            # load segnet weights
            self.segnet.load_state_dict(param_keys, strict=False)
            # for param in self.segnet.parameters():
            #     param.requires_grad = False
            # self.segnet.eval()
        self.use_frozen_bev_feats = args.freeze_backbone

        self.mask_features = nn.Conv2d(
                                        self.feat_dim,
                                        self.mask_dim,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                    )
        weight_init.c2_xavier_fill(self.mask_features)


         # Weights
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)

        

    def forward(self, features_dict, data_dict):

        if self.use_frozen_bev_feats:
            self.segnet.eval()

        features = features_dict['features']
        rgb_flip_index = features_dict['rgb_flip_index']

        images = data_dict['images']    # B, N, C, H, W
        rots = data_dict['rots']        # B, N, 3, 3
        trans = data_dict['trans']      # B, N, 3
        intrins = data_dict['intrins']  # B, N, 4, 4
        
        B, N, _, H, W = images.shape

        device = images.device

        intrins_ = intrins.reshape(B*N, 4, 4)
    
        pix_T_cams_ = geom.merge_intrinsics(*geom.split_intrinsics(intrins_)).to(device) # B*N, 4, 4

        velo_T_cams = geom.merge_rtlist(rots, trans).to(device)         # B, N, 4, 4
        velo_T_cams_ = velo_T_cams.reshape(B*N, 4, 4)
        cams_T_velo_ = geom.safe_inverse(velo_T_cams_)
        
        cam0_T_camXs = geom.get_camM_T_camXs(velo_T_cams, ind=0)        # B, N, 4, 4

        vox_util = vox.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=self.scene_centroid.to(device),
            bounds=self.bounds,
            assert_cube=False)
        
        if self.rand_flip:
            features[rgb_flip_index] = torch.flip(features[rgb_flip_index], [-1])
        
        feat_camXs_ = features      # B*N, feat_dim, h, w
        
        _, C, Hf, Wf = feat_camXs_.shape

        assert  C == self.feat_dim, f'feat_camXs_ shape: {feat_camXs_.shape}, self.feat2d_dim: {self.feat_dim}'

        sy = Hf/float(H)
        sx = Wf/float(W)

        # unproject image feature to 3d grid
        featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy)

        if self.use_frozen_bev_feats:

            with torch.no_grad():
                out_dict = self.segnet( 
                                feat_camXs_, 
                                featpix_T_cams_, 
                                cam0_T_camXs,
                                vox_util
                                )
        else:

            out_dict = self.segnet( 
                                    feat_camXs_, 
                                    featpix_T_cams_, 
                                    cam0_T_camXs,
                                    vox_util
                                    )
            
        bev_feats = out_dict['raw_feat']  
        
        multi_scale_features = out_dict['multi_scale_features'] # (from low to high resolution)

        if self.use_multiscale_features:
       
            # Reverse feature maps into top-down order (from low to high resolution)
            for idx, f in enumerate(multi_scale_features): 
                x = multi_scale_features[idx] 
                lateral_conv = self.lateral_convs[idx]
                output_conv = self.output_convs[idx]
                
                if lateral_conv is None:
                    y = output_conv(x)
                else:
                    cur_fpn = lateral_conv(x)
                    # Following FPN implementation, we use nearest upsampling here
                    y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                    y = output_conv(y)

                multi_scale_features[idx] = y


        return self.mask_features(bev_feats), out_dict, multi_scale_features



class Segnet(nn.Module):
    def __init__(self, Z, Y, X, vox_util=None, 
                 use_radar=False,
                 use_lidar=False,
                 use_metaradar=False,
                 do_rgbcompress=True,
                 rand_flip=False,
                 latent_dim=128,
                 encoder_type="res101"):
        super(Segnet, self).__init__()
        assert (encoder_type in ["res101", "res50", "effb0", "effb4"])

        self.Z, self.Y, self.X = Z, Y, X
        self.use_radar = use_radar
        self.use_lidar = use_lidar
        self.use_metaradar = use_metaradar
        self.do_rgbcompress = do_rgbcompress   
        self.rand_flip = rand_flip
        self.latent_dim = latent_dim
        self.encoder_type = encoder_type

        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim

        # BEV compressor
        if self.use_radar:
            if self.use_metaradar:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y + 16*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y+1, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
        elif self.use_lidar:
            self.bev_compressor = nn.Sequential(
                nn.Conv2d(feat2d_dim*Y+Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                nn.InstanceNorm2d(latent_dim),
                nn.GELU(),
            )
        else:
            if self.do_rgbcompress:
                self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat2d_dim*Y, feat2d_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(latent_dim),
                    nn.GELU(),
                )
            else:
                # use simple sum
                pass

        # Decoder
        self.decoder = BEVDecoder(
            in_channels=latent_dim,
            n_classes=1,
            predict_future_flow=False
        )

        # Weights
        self.ce_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.center_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
        self.offset_weight = nn.Parameter(torch.tensor(0.0), requires_grad=True)
            
        # set_bn_momentum(self, 0.1)

        if vox_util is not None:
            self.xyz_memA = basic.gridcloud3d(1, Z, Y, X, norm=False)
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA, Z, Y, X, assert_cube=False)
        else:
            self.xyz_camA = None
        
    def forward(self, 
                feat_camXs_, 
                featpix_T_cams_, 
                cam0_T_camXs,
                vox_util, 
                rad_occ_mem0=None
                ):

        '''
            Modify SegNet such that it takes features from backbone instead of images
        
        '''
        B, N = cam0_T_camXs.shape[:2]
        Z, Y, X = self.Z, self.Y, self.X

        cam0_T_camXs_ = cam0_T_camXs.reshape(B*N, 4, 4)
        camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs_)

        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B*N,1,1)
        else:
            xyz_camA = None

        feat_mems_ = vox_util.unproject_image_to_mem(
                            feat_camXs_,
                            basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
                            camXs_T_cam0_, Z, Y, X,
                            xyz_camA=xyz_camA
                            )  # B*N, C, Z, Y, X
        
        feat_mems = feat_mems_.reshape(B, N, self.feat2d_dim, Z, Y, X)   # B, N, C, Z, Y, X
        
        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = basic.reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X

        if self.rand_flip:
            self.bev_flip1_index = np.random.choice([0,1], B).astype(bool)
            self.bev_flip2_index = np.random.choice([0,1], B).astype(bool)
            feat_mem[self.bev_flip1_index] = torch.flip(feat_mem[self.bev_flip1_index], [-1])
            feat_mem[self.bev_flip2_index] = torch.flip(feat_mem[self.bev_flip2_index], [-3])

            if rad_occ_mem0 is not None:
                rad_occ_mem0[self.bev_flip1_index] = torch.flip(rad_occ_mem0[self.bev_flip1_index], [-1])
                rad_occ_mem0[self.bev_flip2_index] = torch.flip(rad_occ_mem0[self.bev_flip2_index], [-3])

        # bev compressing
        if self.use_radar:
            assert(rad_occ_mem0 is not None)
            if not self.use_metaradar:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                rad_bev = torch.sum(rad_occ_mem0, 3).clamp(0,1) # squish the vertical dim
                feat_bev_ = torch.cat([feat_bev_, rad_bev], dim=1)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, 16*Y, Z, X)
                feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
                feat_bev = self.bev_compressor(feat_bev_)
        elif self.use_lidar:
            assert(rad_occ_mem0 is not None)
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
            rad_bev_ = rad_occ_mem0.permute(0, 1, 3, 2, 4).reshape(B, Y, Z, X)
            feat_bev_ = torch.cat([feat_bev_, rad_bev_], dim=1)
            feat_bev = self.bev_compressor(feat_bev_)
        else: # rgb only
            if self.do_rgbcompress:
                feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
                feat_bev = self.bev_compressor(feat_bev_)
            else:
                feat_bev = torch.sum(feat_mem, dim=3)

        # bev decoder
        out_dict = self.decoder(feat_bev, (self.bev_flip1_index, self.bev_flip2_index) if self.rand_flip else None)

        return out_dict



class UpsamplingConcat(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(), #inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU() #inplace=True),
        )

    def forward(self, x_to_upsample, x):
        x_to_upsample = self.upsample(x_to_upsample)
        x_to_upsample = torch.cat([x, x_to_upsample], dim=1)
        return self.conv(x_to_upsample)

class UpsamplingAdd(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.upsample_layer = nn.Sequential(
            nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            nn.InstanceNorm2d(out_channels),
        )

    def forward(self, x, x_skip):
        x = self.upsample_layer(x)
        return x + x_skip



class BEVDecoder(nn.Module):
    def __init__(self, in_channels, n_classes, predict_future_flow):
        super().__init__()
        backbone = resnet18(pretrained=False, zero_init_residual=True)
        self.first_conv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = backbone.bn1
        self.relu = backbone.relu

        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.predict_future_flow = predict_future_flow

        shared_out_channels = in_channels
        self.up3_skip = UpsamplingAdd(256, 128, scale_factor=2)
        self.up2_skip = UpsamplingAdd(128, 64, scale_factor=2)
        self.up1_skip = UpsamplingAdd(64, shared_out_channels, scale_factor=2)

        self.feat_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(), #inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=1, padding=0),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(), #inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(), #inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(), #inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(), #inplace=True),
                nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
            )


    def forward(self, x, bev_flip_indices=None):
        b, c, h, w = x.shape

        multi_scale_features = []

        # (H, W)
        skip_x = {'1': x}
        x = self.first_conv(x)
        x = self.bn1(x)
        x = self.relu(x)

        # (H/4, W/4)
        x = self.layer1(x)
        skip_x['2'] = x
        x = self.layer2(x)
        skip_x['3'] = x

        # (H/8, W/8)
        x = self.layer3(x)
        multi_scale_features.append(x)

        # First upsample to (H/4, W/4)
        x = self.up3_skip(x, skip_x['3'])
        multi_scale_features.append(x)

        # Second upsample to (H/2, W/2)
        x = self.up2_skip(x, skip_x['2'])
        multi_scale_features.append(x)

        # Third upsample to (H, W)
        x = self.up1_skip(x, skip_x['1'])

        if bev_flip_indices is not None:
            bev_flip1_index, bev_flip2_index = bev_flip_indices
            x[bev_flip2_index] = torch.flip(x[bev_flip2_index], [-2]) # note [-2] instead of [-3], since Y is gone now
            x[bev_flip1_index] = torch.flip(x[bev_flip1_index], [-1])

        feat_output = self.feat_head(x)
        segmentation_output = self.segmentation_head(x)
        instance_center_output = self.instance_center_head(x)
        instance_offset_output = self.instance_offset_head(x)
        instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None

        return {
            'raw_feat': x,
            'multi_scale_features': multi_scale_features,
            #'feat': feat_output.view(b, *feat_output.shape[1:]),
            'segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),          # B, 1, H, W
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]), # B, 1, H, W
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            # 'instance_flow': instance_future_output.view(b, *instance_future_output.shape[1:])
            # if instance_future_output is not None else None,
        }