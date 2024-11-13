# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from .transformer import Transformer
from .position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class DETRTransformer(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self,
                  args,):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.num_queries = args.num_queries

        # positional encoding
        N_steps = args.pe_hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)

        self.transformer = Transformer(d_model=args.pe_hidden_dim)  # or mask_dim ?

        self.class_embed = nn.Linear(args.pe_hidden_dim, args.num_classes + 1)
        self.bbox_embed = MLP(args.pe_hidden_dim, args.pe_hidden_dim, 4, 3)
        self.query_embed = nn.Embedding(args.num_queries, args.pe_hidden_dim)

        self.input_proj = nn.Conv2d(args.mask_dim, args.pe_hidden_dim, kernel_size=1)
        
        self.aux_loss = args.deep_supervision

        # downsamples to /4
        self.downsample = nn.Sequential(
            nn.Conv2d(args.mask_dim, args.mask_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(args.mask_dim, args.mask_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )


    def forward(self, decoder_features, mask = None):
        """ 
            - decoder_features: B, MASK_DIM, X, Z

            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        # MASKED BACKBONE? do we need it, identifies padded regions in feature maps.

        decoder_features = self.downsample(decoder_features)      # B, MASK_DIM, X/4, Z/4 

        # features --> bev features  b, MASK_DIM, X, Z
        src = self.input_proj(decoder_features)                  # B, PE_HIDDEN_DIM, X, Z    
        pos = self.pe_layer(src)                                 # B, PE_HIDDEN_DIM, X, Z

        #assert mask is not None
        hs = self.transformer(src, 
                              mask, 
                              self.query_embed.weight, 
                              pos)[0]                           # B, NUM_QUERIES, PE_HIDDEN_DIM

        outputs_class = self.class_embed(hs)                    # B, NUM_QUERIES, NUM_CLASSES + 1
        outputs_coord = self.bbox_embed(hs).sigmoid()           # B, NUM_QUERIES, 4
        out = {'pred_logits': outputs_class[-1],        
               'pred_boxes': outputs_coord[-1]}
        
        print('pred_logits', outputs_class[-1].softmax(-1)[0,:10,:])
        print('pred_boxes', outputs_coord[-1][0,:10,:])

        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]