# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Tuple

import torch
from torch import nn
from torch.nn import functional as F

from .criterion import SetCriterion
from .matcher import HungarianMatcher, HungarianMatcherDETR
from .build_modules import *

class Mask2Former4BEV(nn.Module):
    """
    Main class for bev segmentation architectures.
    """

    def __init__(
        self,
        args
        ):
        
        super().__init__()

        #self.training = args.training
        self.bev_module_name = args.bev_module_name
        if args.bev_module_name != 'segnet':
            self.backbone = build_backbone(args)
        self.sem_seg_head = build_sem_seg_head(args)

        # loss weights
        class_weight = args.class_weight
        dice_weight = args.dice_weight
        mask_weight = args.mask_weight
        translation_weight = args.translation_weight
        heading_weight = args.heading_weight

        self.num_queries = args.num_queries
        self.num_classes = args.num_classes

        mask_classification = args.mask_classification
        rt_regression = args.rt_regression
        center_offset_loss = args.use_center_offset_loss

        # building criterion
        matcher = HungarianMatcher(
            cost_class=0 if not mask_classification else class_weight,   # no classification loss used
            cost_mask=mask_weight,
            cost_dice=dice_weight,
            num_points=args.train_num_points,
        )

        self.is_loss_weights_param = args.is_loss_weights_param
        if args.is_loss_weights_param:
            # initialize weights as 0
            self.mask_weight = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.dice_weight = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.class_weight = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.translation_weight = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)
            self.heading_weight = torch.nn.Parameter(torch.tensor(0.0), requires_grad=True)

            # mask_weight = args.mask_weight / torch.exp(self.mask_weight)
            # dice_weight = args.dice_weight / torch.exp(self.dice_weight)
            # class_weight = args.class_weight / torch.exp(self.class_weight)
            # translation_weight = args.translation_weight / torch.exp(self.translation_weight)
            # heading_weight = args.heading_weight / torch.exp(self.heading_weight)
        
        weight_dict = {"loss_mask": mask_weight, "loss_dice": dice_weight}
        if mask_classification:
            weight_dict["loss_ce"] = class_weight
        if rt_regression:
            weight_dict["loss_rt"] = translation_weight
            weight_dict["loss_giou"] = heading_weight
            #weight_dict["loss_translation"] = translation_weight
            #weight_dict["loss_heading"] = heading_weight

        if center_offset_loss:
            weight_dict["loss_center"] = args.center_offset_weight
            weight_dict["loss_offset"] = args.center_offset_weight
            weight_dict["loss_ce_simple"] = 1.0

            
        no_object_weight = args.no_object_weight
        deep_supervision = args.deep_supervision

        if deep_supervision:
            dec_layers = args.dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        losses = ["masks"]          # only mask loss used
        if mask_classification:
            losses.append("labels")
        
        if rt_regression:
            losses.append("rt")
        
        if center_offset_loss:
            losses.append("center_offset")

        self.criterion = SetCriterion(
                                        args.num_queries,
                                        args.num_classes,
                                        matcher=matcher,
                                        weight_dict=weight_dict,
                                        eos_coef=no_object_weight,
                                        losses=losses,
                                        num_points=args.train_num_points,
                                        oversample_ratio=args.oversample_ratio,
                                        importance_sample_ratio=args.importance_sample_ratio,
                                        use_pointwise_mask_loss=False if args.overfit else True
                                    )

        self.mask_classification = mask_classification
        self.rt_regression = rt_regression

        self.validate_with_gt = args.validate_with_gt

        self.inference = args.inference

        self.sem_mask = args.get_sem_masks


    def train_step(self, batch, outputs):

        targets = []    # should be list of dicts
            
        for info in batch:
            multi_seg_bev = info['multi_seg_bev']               # num_vehicles, H, W       | 0, 1
            multi_vld_bev = info['multi_valid_bev']             # num_vehicles, H, W       | 0, 1

            multi_seg_valid = multi_seg_bev * multi_vld_bev     # num_vehicles, H, W       | 0, 1

            # get the indices of non-zero vehicle masks. eliminate all zero masks
            invisible_vehicles = multi_seg_valid.sum(dim=(1, 2)) == 0                       # num_vehicles
            visible_vehicles = ~invisible_vehicles
            multi_seg_valid = multi_seg_valid[visible_vehicles]                             # num_vehicles, H, W       | 0, 1
            
            # remove same indices from translation_rotation_list
            trans_rot_list = info['translation_rotation_list'].to(multi_seg_bev.device)
            trans_rot_list = trans_rot_list[visible_vehicles]  # num_vehicles, 3
            
            # first num_vehicles are 1 (exist), the rest are 0 (non-exist)
            labels = torch.zeros(self.num_queries).long().to(multi_seg_bev.device)       # num_queries
            labels[:multi_seg_valid.shape[0]] = 1                                        # num_queries

            offset_bev = info['offset_bev'].to(multi_seg_bev.device)    # 1, H, W
            center_bev = info['center_bev'].to(multi_seg_bev.device)
            gt_masks = info['gt_masks'].to(multi_seg_bev.device)
            gt_valid = info['gt_valid'].to(multi_seg_bev.device)


            targets.append({
                            'masks': multi_seg_valid,
                            'labels': labels,
                            'boxes': trans_rot_list,   # num_vehicles, 4
                            'offset_bev': offset_bev,
                            'center_bev': center_bev,
                            'gt_masks': gt_masks,
                            'gt_valid': gt_valid
                            })

        # bipartite matching-based loss
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                if k != 'loss_ce_accuracy' and k != 'loss_ce_precision' and k != 'loss_ce_recall':
                # remove this loss if not specified in `weight_dict`
                    losses.pop(k)
                
        if self.is_loss_weights_param:
            losses['loss_mask_uncertainty'] = 0.5 * self.mask_weight
            losses['loss_dice_uncertainty'] = 0.5 * self.dice_weight
            if self.mask_classification:
                losses['loss_ce_uncertainty'] = 0.5 * self.class_weight
            if self.rt_regression:
                losses['loss_translation_uncertainty'] = 0.5 * self.translation_weight
                losses['loss_heading_uncertainty'] = 0.5 * self.heading_weight

        # check nan loss
        for k, v in losses.items():
            if k != 'loss_ce_accuracy' and k != 'loss_ce_precision' and k != 'loss_ce_recall':
                if not torch.isfinite(v).all():
                    print(f"Loss {k} is {v}")
                    raise FloatingPointError(f"Loss {k} is {v}")

        return losses, outputs
    

    def train_step_binary(self, batch, outputs):

        assert not self.mask_classification, 'mask_classification should be off for binary training'
        assert not self.rt_regression, 'rt_regression should be off for binary training'
            
        targets = []    # should be list of dicts
            
        for info in batch:
            gt_masks = info['gt_masks']                         # 1, H, W
            gt_valid = info['gt_valid']                         # 1, H, W
            offset_bev = info['offset_bev']                     # 1, H, W
            center_bev = info['center_bev']                     # 1, H, W

            gt_masks_valid = gt_masks * gt_valid                # 1, H, W    | 0, 1. TODO: full zero mask ?
            
            targets.append({
                            'masks': gt_masks_valid,
                            'offset_bev': offset_bev,
                            'center_bev': center_bev,
                            'gt_masks': gt_masks,
                            'gt_valid': gt_valid
                            })

        # loss is calculated 
        losses = self.criterion(outputs, targets)

        for k in list(losses.keys()):
            if k in self.criterion.weight_dict:
                losses[k] *= self.criterion.weight_dict[k]
            else:
                if k != 'loss_ce_accuracy' and k != 'loss_ce_precision' and k != 'loss_ce_recall':
                # remove this loss if not specified in `weight_dict`
                    losses.pop(k)

        # check nan loss
        for k, v in losses.items():
            if k != 'loss_ce_accuracy' and k != 'loss_ce_precision' and k != 'loss_ce_recall':
                if not torch.isfinite(v).all():
                    print(f"Loss {k} is {v}")
                    raise FloatingPointError(f"Loss {k} is {v}")

        return losses, outputs
    
    

    def val_step(self, batch, outputs):

        mask_cls_results = outputs["pred_logits"]       # B, num_queries, 2     not probability, projection
        mask_pred_results = outputs["pred_masks"]       # B, num_queries, H, W. not probability, dot prod
        mask_rt_results = outputs["pred_rt"]            # B, num_queries, 4     not probability, translation and rotation
        
        
        #if self.inference == 'sem_seg':
        if self.mask_classification:
            mask_cls = F.softmax(mask_cls_results, dim=-1)                  # B, num_queries, 2 | existensial probability 
        else:
            B, num_queries = mask_pred_results.shape[:2]
            mask_cls = torch.zeros((B, num_queries, self.num_classes+1))    # B, num_queries, 2
            mask_cls = mask_cls.to(mask_pred_results.device)
            mask_cls[:, :, 1] = 1.0
            #mask_cls[:, 1, 1] = 1.0
            #mask_cls[:, 0, 0] = 1.0

        mask_pred = mask_pred_results.sigmoid()                         # B, num_queries, H, W. region probability
        semseg = torch.einsum("bqc,bqhw->bchw", mask_cls, mask_pred)    # B, 2, H, W
        semseg = semseg[:, 1]                                           # B, H, W


        processed_results = []
        for i in range(mask_pred_results.shape[0]):
            processed_results.append({'pred_masks': semseg[i].unsqueeze(dim=0),  # 1, H, W
                                    'pred_query': mask_pred[i],                # num_queries, H, W
                                    'pred_logits': mask_cls[i],                # num_queries, 2
                                    'pred_rt': mask_rt_results[i] if self.rt_regression else None             # num_queries, 4
                                    }) 
            
    
        return processed_results
    

    def val_step_with_gt(self, batch, outputs):

        targets = []    # should be list of dicts
            
        for info in batch:
            multi_seg_bev = info['multi_seg_bev']               # num_vehicles, H, W       | 0, 1
            multi_vld_bev = info['multi_valid_bev']             # num_vehicles, H, W       | 0, 1

            multi_seg_valid = multi_seg_bev * multi_vld_bev     # num_vehicles, H, W       | 0, 1

            # get the indices of non-zero vehicle masks. eliminate all zero masks
            invisible_vehicles = multi_seg_valid.sum(dim=(1, 2)) == 0                       # num_vehicles
            visible_vehicles = ~invisible_vehicles
            multi_seg_valid = multi_seg_valid[visible_vehicles]                             # num_vehicles, H, W       | 0, 1
            
            # remove same indices from translation_rotation_list
            trans_rot_list = info['translation_rotation_list'].to(multi_seg_bev.device)
            trans_rot_list = trans_rot_list[visible_vehicles]  # num_vehicles, 3
            
            # first num_vehicles are 1 (exist), the rest are 0 (non-exist)
            labels = torch.zeros(self.num_queries).long().to(multi_seg_bev.device)       # num_queries
            labels[:multi_seg_valid.shape[0]] = 1                                        # num_queries
            targets.append({
                            'masks': multi_seg_valid,
                            'labels': labels,
                            'boxes': trans_rot_list   # num_vehicles, 4
                            })

        # bipartite matching-based loss
        src_masks = self.criterion(outputs, targets, return_indices=True)   # num_vehicles, H, W
        src_masks = src_masks.sigmoid().sum(dim=0, keepdim=True)            # 1, H, W

        return src_masks, None




    def forward(self, batch, training=True):
        """
        Args:
            batch list(dict): input data
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:
        """

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        data_dict = {}
        images = batch_inputs('images', batch).to(device)   # B, C, H, W
        data_dict['images'] = images
        sem_masks = None
        # if self.sem_mask:
        #     sem_masks = batch_inputs('sem_masks', batch).to(device)   # B, H, W
        #     data_dict['sem_masks'] = sem_masks
        data_dict['rots'] = batch_inputs('rots', batch).to(device)
        data_dict['trans'] = batch_inputs('trans', batch).to(device)
        data_dict['intrins'] = batch_inputs('intrins', batch).to(device)


        assert self.bev_module_name == 'SimpleBEV', 'Only SimpleBEV is supported for now.'
        
        features = self.backbone(images, sem_masks)    
        outputs = self.sem_seg_head(features, data_dict)

        if training:
            if self.num_queries == 2:
                out = self.train_step_binary(batch, outputs)
            else:
                out = self.train_step(batch, outputs)
        elif self.validate_with_gt:
            out = self.val_step_with_gt(batch, outputs)
        else:
            out = self.val_step(batch, outputs)
            
        return out #, outputs




def batch_inputs(key, batched_inputs):
    
    inp = [x[key] for x in batched_inputs]
    inp = torch.stack(inp, dim=0)
    return inp



class DETR4BEV(nn.Module):
    def __init__(self, args):
        super().__init__()

        self.bev_module_name = args.bev_module_name
        if args.bev_module_name != 'segnet':
            self.backbone = build_backbone(args)
        self.sem_seg_head = build_sem_seg_head(args)

        self.X = args.voxel_size[0]
        self.Y = args.voxel_size[1]
        self.Z = args.voxel_size[2]

        self.num_queries = args.num_queries
        self.num_classes = args.num_classes

        # building criterion
        l1_weight = 5 #10 #5
        giou_weight = 2 #1 #2
        matcher = HungarianMatcherDETR(cost_class=1, 
                                        cost_bbox=l1_weight, 
                                        cost_giou=giou_weight
                                        )
        
        weight_dict = {"loss_ce": 1, "loss_bbox": l1_weight}
        weight_dict["loss_giou"] = giou_weight
        if args.deep_supervision:
            aux_weight_dict = {}
            for i in range(args.dec_layers - 1):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)
        losses = ["labels", "boxes", "cardinality"]
        # if self.mask_on:
        #     losses += ["masks"]
        self.criterion = SetCriterion(
            self.num_classes, 
            matcher=matcher, 
            weight_dict=weight_dict, 
            eos_coef=args.no_object_weight, 
            losses=losses,
            num_points=args.train_num_points,
            oversample_ratio=args.oversample_ratio,
            importance_sample_ratio=args.importance_sample_ratio,
        )

        self.cnt = 0



    def forward(self, batch, training=True):
        """
        Args:
            batch list(dict): input data
        Returns:
            list[dict]:
                each dict has the results for one image. The dict contains the following keys:
        """

        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        images = batch_inputs('images', batch).to(device)   # B, C, H, W
        H, W = images.shape[-2:]
        data_dict = {}
        data_dict['images'] = images
        data_dict['rots'] = batch_inputs('rots', batch).to(device)
        data_dict['trans'] = batch_inputs('trans', batch).to(device)
        data_dict['intrins'] = batch_inputs('intrins', batch).to(device)

        features = self.backbone(images)    
        output = self.sem_seg_head(features, data_dict)

        if training:

            targets = []    # should be list of dicts
            
            for info in batch:
                multi_seg_bev = info['multi_seg_bev'].to(device)               # num_vehicles, H, W       | 0, 1
                multi_vld_bev = info['multi_valid_bev'].to(device)             # num_vehicles, H, W       | 0, 1

                multi_seg_valid = multi_seg_bev * multi_vld_bev     # num_vehicles, H, W       | 0, 1

                # get the indices of non-zero vehicle masks. eliminate all zero masks
                invisible_vehicles = multi_seg_valid.sum(dim=(1, 2)) == 0                       # num_vehicles
                visible_vehicles = ~invisible_vehicles
                multi_seg_valid = multi_seg_valid[visible_vehicles]                             # num_vehicles, H, W       | 0, 1
                
                # remove same indices from translation_rotation_list
                trans_rot_list = info['translation_rotation_list'].to(multi_seg_bev.device)
                trans_rot_list = trans_rot_list[visible_vehicles]  # num_vehicles, 3
                
                # first num_vehicles are 1 (exist), the rest are 0 (non-exist)
                labels = torch.zeros(self.num_queries).long().to(multi_seg_bev.device)       # num_queries
                labels[:multi_seg_valid.shape[0]] = 1                                        # num_queries
                targets.append({
                                'masks': multi_seg_valid,
                                'labels': labels,
                                'boxes': trans_rot_list   # num_vehicles, 4
                                })
                
                # print('boxes ', trans_rot_list)
                # # # save gt mask as an image
                # mask = info['gt_masks'].to(device) * info['gt_valid'].to(device)
                # # # save mask as an image
                # import torchvision
                # torchvision.utils.save_image(mask, f"gt_mask_.png")
                # exit()
                # boxes  tensor([[0.5925, 0.3175, 0.0050, 0.0150],
                # [0.4700, 0.6525, 0.0200, 0.0350],
                # [0.4100, 0.6450, 0.0400, 0.0600],
                # [0.4700, 0.5800, 0.0200, 0.0400],
                # [0.3200, 0.1225, 0.0200, 0.0450],
                # [0.6300, 0.7425, 0.0500, 0.0450],
                # [0.4800, 0.0150, 0.0100, 0.0200],
                # [0.4750, 0.1500, 0.0200, 0.0500],
                # [0.6225, 0.3175, 0.0050, 0.0150],
                # [0.6150, 0.8025, 0.0500, 0.0250],
                # [0.7025, 0.7750, 0.0450, 0.0200],
                # [0.8050, 0.7750, 0.0500, 0.0200]],

            # make output target
            # output = {'pred_logits': labels[-1].unsqueeze(0), 'pred_boxes': boxes.unsqueeze(0)} 
            loss_dict = self.criterion(output, targets)
            weight_dict = self.criterion.weight_dict
            for k in loss_dict.keys():
                if k in weight_dict:
                    loss_dict[k] *= weight_dict[k]


            box_cls = output["pred_logits"]     # B, num_queries, 2     not probability, projection
            box_pred = output["pred_boxes"]     # B, num_queries, 4     center_x, center_y, height, width
            #mask_pred = output["pred_masks"] if self.mask_on else None

            # For each box we assign the best class or the second best if the best on is `no_object`.
            box_cls = F.softmax(box_cls, dim=-1)                            # B, num_queries, 2
            _, labels = box_cls.max(-1)     
            processed_results = []
            for i in range(box_pred.shape[0]):

                box_xyxy = box_cxcywh_to_xyxy_scale(box_pred[i], scale=self.X)  # num_queries, 4
                box_xyxy = box_xyxy.int()

                # make a binary mask out of pred_boxes considering labels
                mask = torch.zeros((self.X, self.Z), dtype=torch.uint8, device=device)
                for q, box in enumerate(box_xyxy):
                    if box[0] < 0 or box[1] < 0 or box[2] > self.Z or box[3] > self.X:
                        continue
                    if labels[i][q].item() == 0:
                        continue
                    box = box.to(torch.int32)   
                    mask[box[1]:box[3], box[0]:box[2]] = 1

                processed_results.append({"pred_masks": mask.unsqueeze(dim=0),
                                          "pred_boxes": box_pred[i],  
                                          "pred_logits": box_cls[i],
                                          })
            
            return loss_dict, processed_results
        
        else:

            box_cls = output["pred_logits"]     # B, num_queries, 2     not probability, projection
            box_pred = output["pred_boxes"]     # B, num_queries, 4     center_x, center_y, height, width
            #mask_pred = output["pred_masks"] if self.mask_on else None

            # For each box we assign the best class or the second best if the best on is `no_object`.
            box_cls = F.softmax(box_cls, dim=-1)                            # B, num_queries, 2
            _, labels = box_cls.max(-1)                                     # B, num_queries  

            processed_results = []
            for i in range(box_pred.shape[0]):

                box_xyxy = box_cxcywh_to_xyxy_scale(box_pred[i], scale=self.X)  # num_queries, 4
                box_xyxy = box_xyxy.int()

                # make a binary mask out of pred_boxes considering labels
                mask = torch.zeros((self.X, self.Z), dtype=torch.uint8, device=device)
                for q, box in enumerate(box_xyxy):
                    if box[0] < 0 or box[1] < 0 or box[2] > self.Z or box[3] > self.X:
                        continue
                    if labels[i][q].item() == 0:
                        continue
                    box = box.to(torch.int32)   
                    mask[box[1]:box[3], box[0]:box[2]] = 1

                processed_results.append({"pred_masks": mask.unsqueeze(dim=0),
                                          "pred_boxes": box_pred[i],  
                                          "pred_logits": box_cls[i],
                                          })
            
            return processed_results


def box_cxcywh_to_xyxy_scale(x, scale):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c*scale - 0.5 * w*scale), (y_c*scale - 0.5 * h*scale),
         (x_c*scale + 0.5 * w*scale), (y_c*scale + 0.5 * h*scale)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)