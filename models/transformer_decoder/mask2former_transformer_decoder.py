# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from: https://github.com/facebookresearch/detr/blob/master/models/detr.py
import logging
import fvcore.nn.weight_init as weight_init
from typing import Optional
import torch
from torch import nn, Tensor
from torch.nn import functional as F

from detectron2.config import configurable
from detectron2.layers import Conv2d

from .position_encoding import PositionEmbeddingSine


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        
        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        
        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


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


# Implement MultiScaleMaskedTransformerDecoder without detectron2.
# It will take all the parameters from args instead of cfg.

class MultiScaleMaskedTransformerDecoder(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.mask_classification = args.mask_classification
        self.rt_regression = args.rt_regression

        # positional encoding
        N_steps = args.pe_hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        self.dropout = args.predictor_dropout
        
        # define Transformer decoder here
        self.num_heads = args.nheads
        self.num_layers = args.dec_layers

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=args.pe_hidden_dim,
                    nhead=args.nheads,
                    dropout=self.dropout,
                    normalize_before=args.pre_norm,
                )
            )

            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(
                    d_model=args.pe_hidden_dim,
                    nhead=args.nheads,
                    dropout=self.dropout,
                    normalize_before=args.pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=args.pe_hidden_dim,
                    dim_feedforward=args.dim_feedforward,
                    dropout=self.dropout,
                    normalize_before=args.pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(args.pe_hidden_dim)

        self.num_queries = args.num_queries
        # learnable query features
        self.query_feat = nn.Embedding(args.num_queries, args.pe_hidden_dim)
        # learnable query p.e.
        self.query_embed = nn.Embedding(args.num_queries, args.pe_hidden_dim)

        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, args.pe_hidden_dim)
        self.in_channels = args.multiscale_conv_dim

        self.input_proj = nn.ModuleList()
        self.use_multiscale_features = args.use_multiscale_features
        self.multiscale_dim = args.multiscale_conv_dim
        if self.use_multiscale_features:

            for _ in range(self.num_feature_levels):
                if self.in_channels != args.pe_hidden_dim or args.enforce_input_project:
                    self.input_proj.append(Conv2d(args.in_channels, args.hidden_dim, kernel_size=1))
                    weight_init.c2_xavier_fill(self.input_proj[-1])
                else:
                    self.input_proj.append(nn.Sequential())

            # feature 0 torch.Size([1, 256, 25, 25])                                                                                                                                            | 0/6019 [00:00<?, ?it/s]
            # feature 1 torch.Size([1, 256, 50, 50])
            # feature 2 torch.Size([1, 256, 100, 100])

            # define transformer encoder layers
            # self.transformer_enc_layers = nn.ModuleList()
            # self.transformer_enc_pe_layers = nn.ParameterList()
            # for i in range(self.num_feature_levels):
            #     self.transformer_enc_pe_layers.append(nn.Parameter(torch.randn(1,
            #                                                                    args.voxel_size[0] * args.voxel_size[2] // (2**(6-2*i)),
            #                                                                    args.pe_hidden_dim,
            #                                                                    ))) # 1, h*w, d pos embed 
            #     self.transformer_enc_layers.append(
            #         nn.TransformerEncoderLayer(
            #             d_model=args.pe_hidden_dim,
            #             nhead=args.nheads,
            #             dim_feedforward=args.dim_feedforward,
            #             dropout=self.dropout,
            #             activation="relu",
            #         )
            #     )

        # output FFNs
        if self.mask_classification:
            self.class_embed = nn.Linear( args.pe_hidden_dim, args.num_classes + 1)

        if self.rt_regression:
            #self.rt_layer = nn.Linear(args.pe_hidden_dim, 5)
            self.rt_layer = MLP(args.pe_hidden_dim, args.pe_hidden_dim, 4, 3)
            
        self.mask_embed = MLP(args.pe_hidden_dim, args.pe_hidden_dim, args.mask_dim, 3)


    def forward(self, x, mask_features, bev_decoder_out, mask = None):
        # x is a list of multi-scale feature
        assert len(x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        # disable mask, it does not affect performance
        del mask

        if self.use_multiscale_features:

            # for i in range(self.num_feature_levels):
            #     # add pos embed. repeat pos embed xB
            #     x_tmp = x[i].permute(0, 2, 3, 1).flatten(1, 2)                                  # B, H*W, C
            #     x_tmp = x_tmp + self.transformer_enc_pe_layers[i].repeat(x[i].shape[0], 1, 1)   # B, H*W, C
            #     # pass it to encoder layer
            #     x_tmp = self.transformer_enc_layers[i](x_tmp)                                   # B, H*W, C
            #     x[i] = x_tmp.view(x[i].shape[0], x[i].shape[2], x[i].shape[3], -1).permute(0, 3, 1, 2)  # B, C, H, W


            for i in range(self.num_feature_levels):
                size_list.append(x[i].shape[-2:])
                pos.append(self.pe_layer(x[i], None).flatten(2))
                src.append(self.input_proj[i](x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

                # flatten NxCxHxW to HWxNxC
                pos[-1] = pos[-1].permute(2, 0, 1)
                src[-1] = src[-1].permute(2, 0, 1)


        bs = mask_features.shape[0]

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        output = self.query_feat.weight.unsqueeze(1).repeat(1, bs, 1)

        predictions_class = []
        predictions_mask = []
        predictions_rt = []

        # prediction heads on learnable query features
        outputs_class, outputs_mask, outputs_rt, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[0])
        predictions_class.append(outputs_class)
        predictions_mask.append(outputs_mask)

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels

            if self.use_multiscale_features:
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False
                # attention: cross-attention first
                output = self.transformer_cross_attention_layers[i](
                    output, src[level_index],
                    memory_mask=attn_mask,
                    memory_key_padding_mask=None,  # here we do not apply masking on padded region
                    pos=pos[level_index], query_pos=query_embed
                )

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=query_embed
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )   # B, Q, C

            outputs_class, outputs_mask, outputs_rt, attn_mask = self.forward_prediction_heads(output, mask_features, attn_mask_target_size=size_list[(i + 1) % self.num_feature_levels])

            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
            predictions_rt.append(outputs_rt)

        assert len(predictions_class) == self.num_layers + 1

        out = {
            'pred_logits': predictions_class[-1],       # B, num_queries, 2
            'pred_masks': predictions_mask[-1],         # B, num_queries, H, W
            'pred_rt': predictions_rt[-1],              # B, num_queries, 4
            'aux_outputs': self._set_aux_loss(
                predictions_class if self.mask_classification else None, predictions_rt if self.rt_regression else None, predictions_mask
            ),
            'bev_decoder_out': bev_decoder_out,
        }
        return out
    
    def forward_prediction_heads(self, output, mask_features, attn_mask_target_size):

        decoder_output = self.decoder_norm(output)              # B, Q, C
        decoder_output = decoder_output.transpose(0, 1)         # Q, B, C
        if self.mask_classification:
            outputs_class = self.class_embed(decoder_output)
        else:
            outputs_class = None

        if self.rt_regression:
            outputs_rt = self.rt_layer(decoder_output).sigmoid()
        else:
            outputs_rt = None

        mask_embed = self.mask_embed(decoder_output)            # Q, B, C
        outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)

        # NOTE: prediction is of higher-resolution
        # [B, Q, H, W] -> [B, Q, H*W] -> [B, h, Q, H*W] -> [B*h, Q, HW]
        attn_mask = F.interpolate(outputs_mask, size=attn_mask_target_size, mode="bilinear", align_corners=False)
        # must use bool type
        # If a BoolTensor is provided, positions with ``True`` are not allowed to attend while ``False`` values will be unchanged.
        attn_mask = (attn_mask.sigmoid().flatten(2).unsqueeze(1).repeat(1, self.num_heads, 1, 1).flatten(0, 1) < 0.5).bool()
        attn_mask = attn_mask.detach()

        return outputs_class, outputs_mask, outputs_rt, attn_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_rt, outputs_seg_masks):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        
        if self.rt_regression and self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b, "pred_rt": c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], outputs_rt[:-1])
            ]
        elif self.mask_classification:
            return [
                {"pred_logits": a, "pred_masks": b}
                for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])
            ]
        else:
            return [{"pred_masks": b} for b in outputs_seg_masks[:-1]]





