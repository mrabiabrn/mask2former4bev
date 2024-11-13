'''
    This file is adapted from Mask2Former
'''

import torch.nn as nn
import torch.nn.functional as F

from ..bev_module.simplebev import SimpleBEVModule
from ..bev_module.simplebev4vit import SimpleBEV4ViT
from ..bev_module.segnet import SegnetWrapper
from ..transformer_decoder.mask2former_transformer_decoder import MultiScaleMaskedTransformerDecoder
from ..transformer_decoder.simple_transformer_decoder import SimpleTransformerDecoder


def build_bev_module(args):

    if args.bev_module_name == 'SimpleBEV':
        return SimpleBEVModule(args)
    elif args.bev_module_name == 'segnet':
        return SegnetWrapper(args)
    elif args.bev_module_name == 'SimpleBEV4VIT':
        return SimpleBEV4ViT(args)
    else:
        raise NotImplementedError
    

def build_predictor(args):
    '''
        TransformerPredictor
    '''
    if args.predictor_type == 'SimpleTransformerPredictor':
        return SimpleTransformerDecoder(args)

    return MultiScaleMaskedTransformerDecoder(args)



class MaskFormerHead(nn.Module):
    def __init__(
        self,
        args
    ):
        super(MaskFormerHead, self).__init__()

        self.simple = 0
        if args.predictor_type == 'SimpleTransformerPredictor':
            self.simple = 1

        self.pixel_decoder = build_bev_module(args)
        self.predictor = build_predictor(args)

        self.transformer_in_feature = args.transformer_in_feature
  

    def forward(self, features, data_dict, mask=None):  

        if self.simple:
            bev_feats, out_dict = self.pixel_decoder(features, data_dict)
            predictions = self.predictor(features['features'], bev_feats, mask)

        else:
            mask_features, bev_decoder_out, multi_scale_features = self.pixel_decoder(features, data_dict)
            #if self.transformer_in_feature == "multi_scale_bev_features":
            predictions = self.predictor(multi_scale_features, mask_features, bev_decoder_out, mask)
        
        return predictions

