import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys

from .utils import geom, basic, vox, misc

# import utils.geom
# import utils.vox
# import utils.misc
# import utils.basic

# from ..utils import basic
# from ..utils import geom
# from ..utils import misc

from torchvision.models.resnet import resnet18
#from efficientnet_pytorch import EfficientNet

import fvcore.nn.weight_init as weight_init
EPS = 1e-4

from functools import partial

def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.InstanceNorm1d, nn.InstanceNorm2d, nn.InstanceNorm3d)):
            m.momentum = momentum

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

class Decoder(nn.Module):
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
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=1, padding=0),
        )
        self.segmentation_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        )
        self.instance_offset_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        )
        self.instance_center_head = nn.Sequential(
            nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(shared_out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

        if self.predict_future_flow:
            self.instance_future_head = nn.Sequential(
                nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
                nn.InstanceNorm2d(shared_out_channels),
                nn.ReLU(inplace=True),
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
            'feat': feat_output.view(b, *feat_output.shape[1:]),
            'segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),
            'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            'instance_flow': instance_future_output.view(b, *instance_future_output.shape[1:])
            if instance_future_output is not None else None,
        }

import torchvision
class Encoder_res101(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
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

class Encoder_res50(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.C = C
        resnet = torchvision.models.resnet50(pretrained=True)
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

# class Encoder_eff(nn.Module):
#     def __init__(self, C, version='b4'):
#         super().__init__()
#         self.C = C
#         self.downsample = 8
#         self.version = version

#         if self.version == 'b0':
#             self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
#         elif self.version == 'b4':
#             self.backbone = EfficientNet.from_pretrained('efficientnet-b4')
#         self.delete_unused_layers()

#         if self.downsample == 16:
#             if self.version == 'b0':
#                 upsampling_in_channels = 320 + 112
#             elif self.version == 'b4':
#                 upsampling_in_channels = 448 + 160
#             upsampling_out_channels = 512
#         elif self.downsample == 8:
#             if self.version == 'b0':
#                 upsampling_in_channels = 112 + 40
#             elif self.version == 'b4':
#                 upsampling_in_channels = 160 + 56
#             upsampling_out_channels = 128
#         else:
#             raise ValueError(f'Downsample factor {self.downsample} not handled.')

#         self.upsampling_layer = UpsamplingConcat(upsampling_in_channels, upsampling_out_channels)
#         self.depth_layer = nn.Conv2d(upsampling_out_channels, self.C, kernel_size=1, padding=0)

#     def delete_unused_layers(self):
#         indices_to_delete = []
#         for idx in range(len(self.backbone._blocks)):
#             if self.downsample == 8:
#                 if self.version == 'b0' and idx > 10:
#                     indices_to_delete.append(idx)
#                 if self.version == 'b4' and idx > 21:
#                     indices_to_delete.append(idx)

#         for idx in reversed(indices_to_delete):
#             del self.backbone._blocks[idx]

#         del self.backbone._conv_head
#         del self.backbone._bn1
#         del self.backbone._avg_pooling
#         del self.backbone._dropout
#         del self.backbone._fc

#     def get_features(self, x):
#         # Adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
#         endpoints = dict()

#         # Stem
#         x = self.backbone._swish(self.backbone._bn0(self.backbone._conv_stem(x)))
#         prev_x = x

#         # Blocks
#         for idx, block in enumerate(self.backbone._blocks):
#             drop_connect_rate = self.backbone._global_params.drop_connect_rate
#             if drop_connect_rate:
#                 drop_connect_rate *= float(idx) / len(self.backbone._blocks)
#             x = block(x, drop_connect_rate=drop_connect_rate)
#             if prev_x.size(2) > x.size(2):
#                 endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
#             prev_x = x

#             if self.downsample == 8:
#                 if self.version == 'b0' and idx == 10:
#                     break
#                 if self.version == 'b4' and idx == 21:
#                     break

#         # Head
#         endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

#         if self.downsample == 16:
#             input_1, input_2 = endpoints['reduction_5'], endpoints['reduction_4']
#         elif self.downsample == 8:
#             input_1, input_2 = endpoints['reduction_4'], endpoints['reduction_3']
#         # print('input_1', input_1.shape)
#         # print('input_2', input_2.shape)
#         x = self.upsampling_layer(input_1, input_2)
#         # print('x', x.shape)
#         return x

#     def forward(self, x):
#         x = self.get_features(x)  # get feature vector
#         x = self.depth_layer(x)  # feature and depth head
#         return x



def batch_inputs(key, batched_inputs):
    inp = [x[key] for x in batched_inputs]
    inp = torch.stack(inp, dim=0)
    return inp


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

class SegnetWrapper(nn.Module):
    def __init__(self, args):
        super().__init__()

        Z, Y, X = args.voxel_size
        self.Z, self.Y, self.X = Z, Y, X

        # the scene centroid is defined wrt a reference camera,
        # which is usually random
        scene_centroid_x = 0.0
        scene_centroid_y = 1.0 # down 1 meter
        scene_centroid_z = 0.0

        scene_centroid_py = np.array([scene_centroid_x,
                                    scene_centroid_y,
                                    scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid_py).float()

        self.bounds = args.bounds

        vox_util = vox.Vox_util(
                                    Z, Y, X,
                                    scene_centroid=self.scene_centroid.cuda(),
                                    bounds=self.bounds,
                                    assert_cube=False)

        self.model = Segnet(
                            Z=Z, Y=Y, X=X,
                            vox_util=vox_util,
                            )
        
        lateral_convs = []
        output_convs = []

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ms_feat_channels = args.multiscale_feature_channels
        norm = args.multiscale_feature_norm

        conv_dim = args.multiscale_conv_dim
        self.mask_dim = args.mask_dim

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

        self.feat_dim = args.bev_latent_dim
        self.mask_features = nn.Conv2d(
                                        self.feat_dim,
                                        self.mask_dim,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1,
                                    )
        weight_init.c2_xavier_fill(self.mask_features)




    def forward(self, features, data_dict, training=True):
        """
        Args:
            batch list(dict): input data
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:
        """

        imgs = data_dict['images']      # B, N, C, H, W
        rots = data_dict['rots']        # B, N, 3, 3
        trans = data_dict['trans']      # B, N, 3
        intrins = data_dict['intrins']  # B, N, 4, 4

        B, S, _, H, W = imgs.shape

        device = imgs.device

        #origin_T_velo0t = egopose.to(device) # B,T,4,4
        # lrtlist_velo = lrtlist_velo.to(device)
        # scorelist = scorelist.to(device)

        rgb_camXs = imgs.float().cuda() #.to(device)
        rgb_camXs = rgb_camXs - 0.5 # go to -0.5, 0.5

        # xyz_velo0 = pts.to(device).permute(0, 2, 1)
        # rad_data = radar_data.to(device).permute(0, 2, 1) # B, R, 19
        # xyz_rad = rad_data[:,:,:3]
        # meta_rad = rad_data[:,:,3:]

        B, S, C, H, W = rgb_camXs.shape
        #B, V, D = xyz_velo0.shape

        __p = lambda x: basic.pack_seqdim(x, B)
        __u = lambda x: basic.unpack_seqdim(x, B)

        # mag = torch.norm(xyz_velo0, dim=2)
        # xyz_velo0 = xyz_velo0[:,mag[0]>1]
        # xyz_velo0_bak = xyz_velo0.clone()

        intrins_ = __p(intrins)
        pix_T_cams_ = geom.merge_intrinsics(*geom.split_intrinsics(intrins_)).cuda() #.to(device)
        pix_T_cams = __u(pix_T_cams_)

        velo_T_cams = geom.merge_rtlist(rots, trans).to(device)
        cams_T_velo = __u(geom.safe_inverse(__p(velo_T_cams)))

        cam0_T_camXs = geom.get_camM_T_camXs(velo_T_cams, ind=0)
        camXs_T_cam0 = __u(geom.safe_inverse(__p(cam0_T_camXs)))
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = __p(camXs_T_cam0)

        # xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:,0], xyz_velo0)
        # rad_xyz_cam0 = utils.geom.apply_4x4(cams_T_velo[:,0], xyz_rad)

        # lrtlist_cam0 = utils.geom.apply_4x4_to_lrtlist(cams_T_velo[:,0], lrtlist_velo)

        vox_util = vox.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=self.scene_centroid.to(device),
            bounds=self.bounds,
            assert_cube=False)

        #V = xyz_velo0.shape[1]

        # occ_mem0 = vox_util.voxelize_xyz(xyz_cam0, Z, Y, X, assert_cube=False)
        # rad_occ_mem0 = vox_util.voxelize_xyz(rad_xyz_cam0, Z, Y, X, assert_cube=False)
        # metarad_occ_mem0 = vox_util.voxelize_xyz_and_feats(rad_xyz_cam0, meta_rad, Z, Y, X, assert_cube=False)

        #if not (model.module.use_radar or model.module.use_lidar):

        in_occ_mem0 = None
        # elif model.module.use_lidar:
        #     assert(model.module.use_radar==False) # either lidar or radar, not both
        #     assert(model.module.use_metaradar==False) # either lidar or radar, not both
        #     in_occ_mem0 = occ_mem0
        # elif model.module.use_radar and model.module.use_metaradar:
        #     in_occ_mem0 = metarad_occ_mem0
        # elif model.module.use_radar:
        #     in_occ_mem0 = rad_occ_mem0
        # elif model.module.use_metaradar:
        #     assert(False) # cannot use_metaradar without use_radar

        cam0_T_camXs = cam0_T_camXs

        # lrtlist_cam0_g = lrtlist_cam0

        out_dict = self.model(
                        rgb_camXs=rgb_camXs,
                        pix_T_cams=pix_T_cams,
                        cam0_T_camXs=cam0_T_camXs,
                        vox_util=vox_util,
                        rad_occ_mem0=in_occ_mem0
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


        return self.mask_features(bev_feats), None, multi_scale_features





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

        self.mean = torch.as_tensor([0.485, 0.456, 0.406]).reshape(1,3,1,1).float().cuda()
        self.std = torch.as_tensor([0.229, 0.224, 0.225]).reshape(1,3,1,1).float().cuda()

        # Encoder
        self.feat2d_dim = feat2d_dim = latent_dim
        if encoder_type == "res101":
            self.encoder = Encoder_res101(feat2d_dim)
        elif encoder_type == "res50":
            self.encoder = Encoder_res50(feat2d_dim)
        # elif encoder_type == "effb0":
        #     self.encoder = Encoder_eff(feat2d_dim, version='b0')
        # else:
        #     # effb4
        #     self.encoder = Encoder_eff(feat2d_dim, version='b4')

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
        self.decoder = Decoder(
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

        #self.loss_fn = SimpleLoss(pos_weight=2.13)

    def forward(self, rgb_camXs, pix_T_cams, cam0_T_camXs, vox_util, d=None, rad_occ_mem0=None):
        '''
        B = batch size, S = number of cameras, C = 3, H = img height, W = img width
        rgb_camXs: (B,S,C,H,W)
        pix_T_cams: (B,S,4,4)
        cam0_T_camXs: (B,S,4,4)
        vox_util: vox util object
        rad_occ_mem0:
            - None when use_radar = False, use_lidar = False
            - (B, 1, Z, Y, X) when use_radar = True, use_metaradar = False
            - (B, 16, Z, Y, X) when use_radar = True, use_metaradar = True
            - (B, 1, Z, Y, X) when use_lidar = True
        '''
        B, S, C, H, W = rgb_camXs.shape
        assert(C==3)
        # reshape tensors
        __p = lambda x: basic.pack_seqdim(x, B)
        __u = lambda x: basic.unpack_seqdim(x, B)
        rgb_camXs_ = __p(rgb_camXs)
        pix_T_cams_ = __p(pix_T_cams)
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs_)

        # rgb encoder
        device = rgb_camXs_.device
        rgb_camXs_ = (rgb_camXs_ + 0.5 - self.mean.to(device)) / self.std.to(device)
        if self.rand_flip:
            B0, _, _, _ = rgb_camXs_.shape
            self.rgb_flip_index = np.random.choice([0,1], B0).astype(bool)
            rgb_camXs_[self.rgb_flip_index] = torch.flip(rgb_camXs_[self.rgb_flip_index], [-1])
        feat_camXs_ = self.encoder(rgb_camXs_)
        if self.rand_flip:
            feat_camXs_[self.rgb_flip_index] = torch.flip(feat_camXs_[self.rgb_flip_index], [-1])
        _, C, Hf, Wf = feat_camXs_.shape

        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X

        # unproject image feature to 3d grid
        featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy)
        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B*S,1,1)
        else:
            xyz_camA = None
        feat_mems_ = vox_util.unproject_image_to_mem(
            feat_camXs_,
            basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, Z, Y, X,
            xyz_camA=xyz_camA)
        feat_mems = __u(feat_mems_) # B, S, C, Z, Y, X

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

        # raw_e = out_dict['raw_feat']
        # feat_e = out_dict['feat']
        # seg_e = out_dict['segmentation']
        # center_e = out_dict['instance_center']
        # offset_e = out_dict['instance_offset']

        # seg_bev_g = d['seg_bev_g']
        # center_bev_g = d['center_bev_g']
        # offset_bev_g = d['offset_bev_g']
        # valid_bev_g = d['valid_bev_g']


        return out_dict


class SimpleLoss(torch.nn.Module):
    def __init__(self, pos_weight):
        super(SimpleLoss, self).__init__()
        self.loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.Tensor([pos_weight]), reduction='none')

    def forward(self, ypred, ytgt, valid):
        loss = self.loss_fn(ypred, ytgt)
        loss = basic.reduce_masked_mean(loss, valid)
        return loss

def balanced_mse_loss(pred, gt, valid=None):
    pos_mask = gt.gt(0.5).float()
    neg_mask = gt.lt(0.5).float()
    if valid is None:
        valid = torch.ones_like(pos_mask)
    mse_loss = F.mse_loss(pred, gt, reduction='none')
    pos_loss = basic.reduce_masked_mean(mse_loss, pos_mask*valid)
    neg_loss = basic.reduce_masked_mean(mse_loss, neg_mask*valid)
    loss = (pos_loss + neg_loss)*0.5
    return loss