'''
    This file is adapted from Mask2Former
'''

import torch.nn as nn
import torch.nn.functional as F

from ..bev_module.simplebev import SimpleBEVModule
from ..bev_module.segnet import SegnetWrapper
from ..transformer_decoder.detr_transformer import DETRTransformer


def build_bev_module(args):

    if args.bev_module_name == 'SimpleBEV':
        return SimpleBEVModule(args)
    elif args.bev_module_name == 'segnet':
        return SegnetWrapper(args)
    else:
        raise NotImplementedError
    

def build_predictor(args):
    '''
        TransformerPredictor
    '''
    return DETRTransformer(args)



class DETRHead(nn.Module):
    def __init__(
        self,
        args
    ):
        super(DETRHead, self).__init__()
        

        self.pixel_decoder = build_bev_module(args)     
        self.predictor = build_predictor(args)

        self.transformer_in_feature = args.transformer_in_feature
  

    def forward(self, features, data_dict, mask=None):
        '''
            Passes features through the pixel_decoder and then the predictor
        '''
        mask_features, _ , _ = self.pixel_decoder(features, data_dict)
        predictions = self.predictor(mask_features, mask)
        
        return predictions

