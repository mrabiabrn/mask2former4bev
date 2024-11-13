# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, DeformConv, ShapeSpec, get_norm
from detectron2.modeling import SEM_SEG_HEADS_REGISTRY

from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from ..transformer_decoder.transformer import TransformerEncoder, TransformerEncoderLayer, _get_clones, _get_activation_fn

from .utils import geom, basic, vox

from torchvision.models.resnet import resnet18

EPS = 1e-4

from functools import partial

def build_pixel_decoder(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    name = cfg.MODEL.SEM_SEG_HEAD.PIXEL_DECODER_NAME
    model = SEM_SEG_HEADS_REGISTRY.get(name)(cfg, input_shape)
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model


# This is a modified FPN decoder.
@SEM_SEG_HEADS_REGISTRY.register()
class BasePixelDecoder(nn.Module):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(feature_channels) - 1:                # self.in_features
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_dim = mask_dim
        self.mask_features = Conv2d(
            conv_dim,
            mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3  # always use 3 scales

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        }
        ret["conv_dim"] = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        ret["mask_dim"] = cfg.MODEL.SEM_SEG_HEAD.MASK_DIM
        ret["norm"] = cfg.MODEL.SEM_SEG_HEAD.NORM
        return ret

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return self.mask_features(y), None, multi_scale_features

    def forward_features_old(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return self.mask_features(y), None, multi_scale_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)


class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        bs, c, h, w = src.shape
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory.permute(1, 2, 0).view(bs, c, h, w)


# This is a modified FPN decoder with extra Transformer encoder that processes the lowest-resolution feature map.
@SEM_SEG_HEADS_REGISTRY.register()
class TransformerEncoderPixelDecoder(BasePixelDecoder):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        transformer_pre_norm: bool,
        conv_dim: int,
        mask_dim: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            transformer_pre_norm: whether to use pre-layernorm or not
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim, mask_dim=mask_dim, norm=norm)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        in_channels = feature_channels[len(self.in_features) - 1]
        self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
        weight_init.c2_xavier_fill(self.input_proj)
        self.transformer = TransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            normalize_before=transformer_pre_norm,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        # update layer
        use_bias = norm == ""
        output_norm = get_norm(norm, conv_dim)
        output_conv = Conv2d(
            conv_dim,
            conv_dim,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=use_bias,
            norm=output_norm,
            activation=F.relu,
        )
        weight_init.c2_xavier_fill(output_conv)
        delattr(self, "layer_{}".format(len(self.in_features)))
        self.add_module("layer_{}".format(len(self.in_features)), output_conv)
        self.output_convs[0] = output_conv

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        ret["transformer_dropout"] = cfg.MODEL.MASK_FORMER.DROPOUT
        ret["transformer_nheads"] = cfg.MODEL.MASK_FORMER.NHEADS
        ret["transformer_dim_feedforward"] = cfg.MODEL.MASK_FORMER.DIM_FEEDFORWARD
        ret[
            "transformer_enc_layers"
        ] = cfg.MODEL.SEM_SEG_HEAD.TRANSFORMER_ENC_LAYERS  # a separate config
        ret["transformer_pre_norm"] = cfg.MODEL.MASK_FORMER.PRE_NORM
        return ret

    def forward_features(self, features):
        multi_scale_features = []
        num_cur_levels = 0
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(self.in_features[::-1]):
            x = features[f]
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            if lateral_conv is None:
                transformer = self.input_proj(x)
                pos = self.pe_layer(x)
                transformer = self.transformer(transformer, None, pos)
                y = output_conv(transformer)
                # save intermediate feature as input to Transformer decoder
                transformer_encoder_features = transformer
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            if num_cur_levels < self.maskformer_num_feature_levels:
                multi_scale_features.append(y)
                num_cur_levels += 1
        return self.mask_features(y), transformer_encoder_features, multi_scale_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)








@SEM_SEG_HEADS_REGISTRY.register()
class BEVPixelDecoder(BasePixelDecoder):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        voxel_size: Tuple[float, float, float],
        bounds: List[float],
        feat_dim: int,
        do_rgb_compress: bool,
        rand_flip: bool,
        last_resnet_layer: str,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            transformer_pre_norm: whether to use pre-layernorm or not
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(input_shape, conv_dim=conv_dim, mask_dim=mask_dim, norm=norm)

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        print(f'input_shape: {input_shape}')
        # self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        # feature_channels = [v.channels for k, v in input_shape]

        self.last_resnet_layer = last_resnet_layer
        self.last_resnet_layer_ch = input_shape[-1][1].channels

        feature_channels = [64, 128, 256] # these are for bev decoder

        print(f'in features: {self.in_features}')
        print(f'feature_channels: {feature_channels}')

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            print(f'idx: {idx}, in_channels: {in_channels}')
            if idx == len(feature_channels) - 1:                # self.in_features
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv2d(
                    in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)

        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        # for i in range(len(self.lateral_convs)):
        #     print(f'lateral_convs {i}: {self.lateral_convs[i]}')
        # for i in range(len(self.output_convs)):
        #     print(f'output_convs {i}: {self.output_convs[i]}')

        self.X, self.Y, self.Z = voxel_size
        self.bounds = bounds

        scene_centroid_x = 0.0
        scene_centroid_y = 1.0 # down 1 meter
        scene_centroid_z = 0.0

        scene_centroid_py = np.array([scene_centroid_x,
                                    scene_centroid_y,
                                    scene_centroid_z]).reshape([1, 3])
        self.scene_centroid = torch.from_numpy(scene_centroid_py).float()

        self.vox_util = vox.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=self.scene_centroid,
            bounds=bounds,
            assert_cube=False)

        self.xyz_memA = basic.gridcloud3d(1, self.Z, self.Y, self.X, norm=False)
        self.xyz_camA = self.vox_util.Mem2Ref(self.xyz_memA, self.Z, self.Y, self.X, assert_cube=False)

        self.feat_proj = nn.Conv2d(self.last_resnet_layer_ch, feat_dim, kernel_size=1, padding=0)
        self.feat2d_dim = feat_dim
        self.do_rgbcompress = do_rgb_compress

        # self.segnet =
        self.bev_compressor = nn.Sequential(
                    nn.Conv2d(feat_dim*self.Y, feat_dim, kernel_size=3, padding=1, stride=1, bias=False),
                    nn.InstanceNorm2d(feat_dim),
                    nn.GELU(),
                )

        self.decoder = BEVDecoder(
            in_channels=feat_dim,
            n_classes=1,
            predict_future_flow=False
        )

        self.mask_features = Conv2d(
            feat_dim,
            self.mask_dim,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        weight_init.c2_xavier_fill(self.mask_features)

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        ret = super().from_config(cfg, input_shape)
        
        ret["voxel_size"] = cfg.INPUT.VOXEL_SIZE
        ret["bounds"] = cfg.INPUT.BOUNDS
        ret["feat_dim"] = cfg.MODEL.BEV.FEAT_DIM
        ret["do_rgb_compress"] = cfg.MODEL.BEV.DO_RGB_COMPRESS
        ret["rand_flip"] = cfg.MODEL.BEV.RAND_FLIP
        ret["last_resnet_layer"] = cfg.MODEL.RESNETS.OUT_FEATURES[-1]
        return ret

    def forward_features(self, features, data_dict):

        images_ = data_dict['images']    # B*N, C, H, W
        rots = data_dict['rots']        # B, N, 3, 3
        trans = data_dict['trans']      # B, N, 3
        intrins = data_dict['intrins']  # B, N, 4, 4
        
        B, N = rots.shape[:2]
        H, W = images_.shape[2:]

        device = images_.device
        images = images_.reshape(B*N, 3, H, W)

        intrins_ = intrins.reshape(B*N, 4, 4)
    
        pix_T_cams_ = geom.merge_intrinsics(*geom.split_intrinsics(intrins_)).to(device) # B*N, 4, 4
        pix_T_cams = pix_T_cams_.reshape(B, N, 4, 4)

        velo_T_cams = geom.merge_rtlist(rots, trans).to(device)         # B, N, 4, 4
        velo_T_cams_ = velo_T_cams.reshape(B*N, 4, 4)
        cams_T_velo_ = geom.safe_inverse(velo_T_cams_)
        cams_T_velo = cams_T_velo_.reshape(B, N, 4, 4) 
        
        cam0_T_camXs = geom.get_camM_T_camXs(velo_T_cams, ind=0)        # B, N, 4, 4
        cam0_T_camXs_ = cam0_T_camXs.reshape(B*N, 4, 4)
        camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs_)
        camXs_T_cam0 = cam0_T_camXs_.reshape(B, N, 4, 4)

        vox_util = vox.Vox_util(
            self.Z, self.Y, self.X,
            scene_centroid=self.scene_centroid.to(device),
            bounds=self.bounds,
            assert_cube=False)
        
        in_occ_mem0 = None

        feat_camXs_ = features[self.last_resnet_layer]      # B*N, feat_dim, h, w. res3 --> 512, 28, 56

        feat_camXs_ = self.feat_proj(feat_camXs_) # B*N, feat2d_dim, h, w

        
        _, C, Hf, Wf = feat_camXs_.shape
        assert  C == self.feat2d_dim, f'feat_camXs_ shape: {feat_camXs_.shape}, self.feat2d_dim: {self.feat2d_dim}'

        sy = Hf/float(H)
        sx = Wf/float(W)
        Z, Y, X = self.Z, self.Y, self.X

        # unproject image feature to 3d grid
        featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy)
        
        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B*N,1,1)
        else:
            xyz_camA = None
        
        feat_mems_ = vox_util.unproject_image_to_mem(
            feat_camXs_,
            basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, Z, Y, X,
            xyz_camA=xyz_camA)

        # print(f'feat_mems_ shape: {feat_mems_.shape}')

        feat_mems = feat_mems_.reshape(B, N, C, Z, Y, X) 
        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = basic.reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X

        # bev compressing
        if self.do_rgbcompress:
            feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
            feat_bev = self.bev_compressor(feat_bev_)
        else:
            feat_bev = torch.sum(feat_mem, dim=3)

        # bev decoder
        out_dict = self.decoder(feat_bev)

        bev_feats = out_dict['raw_feat']
        multi_scale_features = out_dict['multi_scale_features'] # (from low to high resolution)
        #multi_scale_features = multi_scale_features[::-1]


        # print('reverse in features ' , self.in_features[::-1])
        # Reverse feature maps into top-down order (from low to high resolution)
        for idx, f in enumerate(multi_scale_features): #self.in_features[::-1]):
            x = multi_scale_features[idx] #[f]
            # print('x shape: ', x.shape)
            lateral_conv = self.lateral_convs[idx]
            output_conv = self.output_convs[idx]
            # print('lateral_conv: ', lateral_conv)
            #print('output_conv: ', output_conv)
            if lateral_conv is None:
                y = output_conv(x)
            else:
                cur_fpn = lateral_conv(x)
                # Following FPN implementation, we use nearest upsampling here
                y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
                y = output_conv(y)
            #print('y shape: ', y.shape)
            if idx < self.maskformer_num_feature_levels:
                multi_scale_features[idx] = y


        return self.mask_features(bev_feats), None, multi_scale_features

    def forward(self, features, targets=None):
        logger = logging.getLogger(__name__)
        logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
        return self.forward_features(features)


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
            self.xyz_memA = utils.basic.gridcloud3d(1, Z, Y, X, norm=False)
            self.xyz_camA = vox_util.Mem2Ref(self.xyz_memA, Z, Y, X, assert_cube=False)
        else:
            self.xyz_camA = None
        
    def forward(self, feat_camXs_, pix_T_cams, cam0_T_camXs, vox_util, rad_occ_mem0=None):
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
        __p = lambda x: utils.basic.pack_seqdim(x, B)
        __u = lambda x: utils.basic.unpack_seqdim(x, B)
        rgb_camXs_ = __p(rgb_camXs)
        pix_T_cams_ = __p(pix_T_cams)
        cam0_T_camXs_ = __p(cam0_T_camXs)
        camXs_T_cam0_ = utils.geom.safe_inverse(cam0_T_camXs_)

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
        featpix_T_cams_ = utils.geom.scale_intrinsics(pix_T_cams_, sx, sy)
        if self.xyz_camA is not None:
            xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B*S,1,1)
        else:
            xyz_camA = None
        feat_mems_ = vox_util.unproject_image_to_mem(
            feat_camXs_,
            utils.basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
            camXs_T_cam0_, Z, Y, X,
            xyz_camA=xyz_camA)
        feat_mems = __u(feat_mems_) # B, S, C, Z, Y, X

        mask_mems = (torch.abs(feat_mems) > 0).float()
        feat_mem = utils.basic.reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X

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

        raw_e = out_dict['raw_feat']
        feat_e = out_dict['feat']
        seg_e = out_dict['segmentation']
        center_e = out_dict['instance_center']
        offset_e = out_dict['instance_offset']

        return raw_e, feat_e, seg_e, center_e, offset_e










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

        # self.feat_head = nn.Sequential(
        #     nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.InstanceNorm2d(shared_out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=1, padding=0),
        # )
        # self.segmentation_head = nn.Sequential(
        #     nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.InstanceNorm2d(shared_out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(shared_out_channels, n_classes, kernel_size=1, padding=0),
        # )
        # self.instance_offset_head = nn.Sequential(
        #     nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.InstanceNorm2d(shared_out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        # )
        # self.instance_center_head = nn.Sequential(
        #     nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
        #     nn.InstanceNorm2d(shared_out_channels),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(shared_out_channels, 1, kernel_size=1, padding=0),
        #     nn.Sigmoid(),
        # )

        # if self.predict_future_flow:
        #     self.instance_future_head = nn.Sequential(
        #         nn.Conv2d(shared_out_channels, shared_out_channels, kernel_size=3, padding=1, bias=False),
        #         nn.InstanceNorm2d(shared_out_channels),
        #         nn.ReLU(inplace=True),
        #         nn.Conv2d(shared_out_channels, 2, kernel_size=1, padding=0),
        #     )

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

        # feat_output = self.feat_head(x)
        # segmentation_output = self.segmentation_head(x)
        # instance_center_output = self.instance_center_head(x)
        # instance_offset_output = self.instance_offset_head(x)
        # instance_future_output = self.instance_future_head(x) if self.predict_future_flow else None

        return {
            'raw_feat': x,
            'multi_scale_features': multi_scale_features,
            # 'feat': feat_output.view(b, *feat_output.shape[1:]),
            # 'segmentation': segmentation_output.view(b, *segmentation_output.shape[1:]),
            # 'instance_center': instance_center_output.view(b, *instance_center_output.shape[1:]),
            # 'instance_offset': instance_offset_output.view(b, *instance_offset_output.shape[1:]),
            # 'instance_flow': instance_future_output.view(b, *instance_future_output.shape[1:])
            # if instance_future_output is not None else None,
        }





# @SEM_SEG_HEADS_REGISTRY.register()
# class BEVPixelDecoder(BasePixelDecoder):
#     @configurable
#     def __init__(
#         self,
#         input_shape: Dict[str, ShapeSpec],
#         *,
#         conv_dim: int,
#         mask_dim: int,
#         voxel_size: Tuple[float, float, float],
#         bounds: List[float],
#         feat_dim: int,
#         do_rgb_compress: bool,
#         rand_flip: bool,
#         norm: Optional[Union[str, Callable]] = None,
#     ):
#         """
#         NOTE: this interface is experimental.
#         Args:
#             input_shape: shapes (channels and stride) of the input features
#             transformer_dropout: dropout probability in transformer
#             transformer_nheads: number of heads in transformer
#             transformer_dim_feedforward: dimension of feedforward network
#             transformer_enc_layers: number of transformer encoder layers
#             transformer_pre_norm: whether to use pre-layernorm or not
#             conv_dims: number of output channels for the intermediate conv layers.
#             mask_dim: number of output channels for the final conv layer.
#             norm (str or callable): normalization for all conv layers
#         """
#         super().__init__(input_shape, conv_dim=conv_dim, mask_dim=mask_dim, norm=norm)

#         input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
#         self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
#         feature_channels = [v.channels for k, v in input_shape]

#         feature_channels = [64, 128, 256] #[256, 128, 64]

#         print(f'in features: {self.in_features}')
#         print(f'feature_channels: {feature_channels}')

#         lateral_convs = []
#         output_convs = []

#         use_bias = norm == ""
#         for idx, in_channels in enumerate(feature_channels):
#             print(f'idx: {idx}, in_channels: {in_channels}')
#             if idx == len(feature_channels) - 1:                # self.in_features
#                 output_norm = get_norm(norm, conv_dim)
#                 output_conv = Conv2d(
#                     in_channels,
#                     conv_dim,
#                     kernel_size=3,
#                     stride=1,
#                     padding=1,
#                     bias=use_bias,
#                     norm=output_norm,
#                     activation=F.relu,
#                 )
#                 weight_init.c2_xavier_fill(output_conv)
#                 self.add_module("layer_{}".format(idx + 1), output_conv)

#                 lateral_convs.append(None)
#                 output_convs.append(output_conv)
#             else:
#                 lateral_norm = get_norm(norm, conv_dim)
#                 output_norm = get_norm(norm, conv_dim)

#                 lateral_conv = Conv2d(
#                     in_channels, conv_dim, kernel_size=1, bias=use_bias, norm=lateral_norm
#                 )
#                 output_conv = Conv2d(
#                     conv_dim,
#                     conv_dim,
#                     kernel_size=3,
#                     stride=1,
#                     padding=1,
#                     bias=use_bias,
#                     norm=output_norm,
#                     activation=F.relu,
#                 )
#                 weight_init.c2_xavier_fill(lateral_conv)
#                 weight_init.c2_xavier_fill(output_conv)
#                 self.add_module("adapter_{}".format(idx + 1), lateral_conv)
#                 self.add_module("layer_{}".format(idx + 1), output_conv)

#                 lateral_convs.append(lateral_conv)
#                 output_convs.append(output_conv)

#         # Place convs into top-down order (from low to high resolution)
#         # to make the top-down computation in forward clearer.
#         self.lateral_convs = lateral_convs[::-1]
#         self.output_convs = output_convs[::-1]

#         # for i in range(len(self.lateral_convs)):
#         #     print(f'lateral_convs {i}: {self.lateral_convs[i]}')
#         # for i in range(len(self.output_convs)):
#         #     print(f'output_convs {i}: {self.output_convs[i]}')

#         self.X, self.Y, self.Z = voxel_size
#         self.bounds = bounds

#         scene_centroid_x = 0.0
#         scene_centroid_y = 1.0 # down 1 meter
#         scene_centroid_z = 0.0

#         scene_centroid_py = np.array([scene_centroid_x,
#                                     scene_centroid_y,
#                                     scene_centroid_z]).reshape([1, 3])
#         self.scene_centroid = torch.from_numpy(scene_centroid_py).float()

#         self.vox_util = vox.Vox_util(
#             self.Z, self.Y, self.X,
#             scene_centroid=self.scene_centroid,
#             bounds=bounds,
#             assert_cube=False)

#         self.xyz_memA = basic.gridcloud3d(1, self.Z, self.Y, self.X, norm=False)
#         self.xyz_camA = self.vox_util.Mem2Ref(self.xyz_memA, self.Z, self.Y, self.X, assert_cube=False)

#         self.feat_proj = nn.Conv2d(512, feat_dim, kernel_size=1, padding=0)
#         self.feat2d_dim = feat_dim
#         self.do_rgbcompress = do_rgb_compress
#         self.bev_compressor = nn.Sequential(
#                     nn.Conv2d(feat_dim*self.Y, feat_dim, kernel_size=3, padding=1, stride=1, bias=False),
#                     nn.InstanceNorm2d(feat_dim),
#                     nn.GELU(),
#                 )

#         self.decoder = BEVDecoder(
#             in_channels=feat_dim,
#             n_classes=1,
#             predict_future_flow=False
#         )

#         self.mask_features = Conv2d(
#             feat_dim,
#             self.mask_dim,
#             kernel_size=3,
#             stride=1,
#             padding=1,
#         )
#         weight_init.c2_xavier_fill(self.mask_features)

#     @classmethod
#     def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
#         ret = super().from_config(cfg, input_shape)
        
#         ret["voxel_size"] = cfg.INPUT.VOXEL_SIZE
#         ret["bounds"] = cfg.INPUT.BOUNDS
#         ret["feat_dim"] = cfg.MODEL.BEV.FEAT_DIM
#         ret["do_rgb_compress"] = cfg.MODEL.BEV.DO_RGB_COMPRESS
#         ret["rand_flip"] = cfg.MODEL.BEV.RAND_FLIP
#         return ret

#     def forward_features(self, features, data_dict):

#         images_ = data_dict['images']    # B*N, C, H, W
#         rots = data_dict['rots']        # B, N, 3, 3
#         trans = data_dict['trans']      # B, N, 3
#         intrins = data_dict['intrins']  # B, N, 4, 4
        
#         B, N = rots.shape[:2]
#         H, W = images_.shape[2:]

#         device = images_.device
#         images = images_.reshape(B*N, 3, H, W)

#         intrins_ = intrins.reshape(B*N, 4, 4)
    
#         pix_T_cams_ = geom.merge_intrinsics(*geom.split_intrinsics(intrins_)).to(device) # B*N, 4, 4
#         pix_T_cams = pix_T_cams_.reshape(B, N, 4, 4)

#         velo_T_cams = geom.merge_rtlist(rots, trans).to(device)         # B, N, 4, 4
#         velo_T_cams_ = velo_T_cams.reshape(B*N, 4, 4)
#         cams_T_velo_ = geom.safe_inverse(velo_T_cams_)
#         cams_T_velo = cams_T_velo_.reshape(B, N, 4, 4) 
        
#         cam0_T_camXs = geom.get_camM_T_camXs(velo_T_cams, ind=0)        # B, N, 4, 4
#         cam0_T_camXs_ = cam0_T_camXs.reshape(B*N, 4, 4)
#         camXs_T_cam0_ = geom.safe_inverse(cam0_T_camXs_)
#         camXs_T_cam0 = cam0_T_camXs_.reshape(B, N, 4, 4)

#         vox_util = vox.Vox_util(
#             self.Z, self.Y, self.X,
#             scene_centroid=self.scene_centroid.to(device),
#             bounds=self.bounds,
#             assert_cube=False)
        
#         in_occ_mem0 = None

#         feat_camXs_ = features['res3']      # B*N, feat_dim, h, w. res3 --> 512, 28, 56
#         feat_camXs_ = self.feat_proj(feat_camXs_) # B*N, feat2d_dim, h, w

        
#         _, C, Hf, Wf = feat_camXs_.shape
#         assert  C == self.feat2d_dim, f'feat_camXs_ shape: {feat_camXs_.shape}, self.feat2d_dim: {self.feat2d_dim}'

#         sy = Hf/float(H)
#         sx = Wf/float(W)
#         Z, Y, X = self.Z, self.Y, self.X

#         # unproject image feature to 3d grid
#         featpix_T_cams_ = geom.scale_intrinsics(pix_T_cams_, sx, sy)
        
#         if self.xyz_camA is not None:
#             xyz_camA = self.xyz_camA.to(feat_camXs_.device).repeat(B*N,1,1)
#         else:
#             xyz_camA = None
        
#         feat_mems_ = vox_util.unproject_image_to_mem(
#             feat_camXs_,
#             basic.matmul2(featpix_T_cams_, camXs_T_cam0_),
#             camXs_T_cam0_, Z, Y, X,
#             xyz_camA=xyz_camA)

#         # print(f'feat_mems_ shape: {feat_mems_.shape}')

#         feat_mems = feat_mems_.reshape(B, N, C, Z, Y, X) 
#         mask_mems = (torch.abs(feat_mems) > 0).float()
#         feat_mem = basic.reduce_masked_mean(feat_mems, mask_mems, dim=1) # B, C, Z, Y, X

#         # bev compressing
#         if self.do_rgbcompress:
#             feat_bev_ = feat_mem.permute(0, 1, 3, 2, 4).reshape(B, self.feat2d_dim*Y, Z, X)
#             feat_bev = self.bev_compressor(feat_bev_)
#         else:
#             feat_bev = torch.sum(feat_mem, dim=3)

#         # bev decoder
#         out_dict = self.decoder(feat_bev)

#         bev_feats = out_dict['raw_feat']
#         multi_scale_features = out_dict['multi_scale_features'] # (from low to high resolution)
#         #multi_scale_features = multi_scale_features[::-1]


#         # print('reverse in features ' , self.in_features[::-1])
#         # Reverse feature maps into top-down order (from low to high resolution)
#         for idx, f in enumerate(multi_scale_features): #self.in_features[::-1]):
#             x = multi_scale_features[idx] #[f]
#             # print('x shape: ', x.shape)
#             lateral_conv = self.lateral_convs[idx]
#             output_conv = self.output_convs[idx]
#             # print('lateral_conv: ', lateral_conv)
#             #print('output_conv: ', output_conv)
#             if lateral_conv is None:
#                 y = output_conv(x)
#             else:
#                 cur_fpn = lateral_conv(x)
#                 # Following FPN implementation, we use nearest upsampling here
#                 y = cur_fpn + F.interpolate(y, size=cur_fpn.shape[-2:], mode="nearest")
#                 y = output_conv(y)
#             #print('y shape: ', y.shape)
#             if idx < self.maskformer_num_feature_levels:
#                 multi_scale_features[idx] = y


#         return self.mask_features(bev_feats), None, multi_scale_features

#     def forward(self, features, targets=None):
#         logger = logging.getLogger(__name__)
#         logger.warning("Calling forward() may cause unpredicted behavior of PixelDecoder module.")
#         return self.forward_features(features)
