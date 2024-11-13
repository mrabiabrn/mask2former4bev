# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/models/detr.py
"""
MaskFormer criterion.
"""

import torch
import torch.nn.functional as F
from torch import nn


#from ..utils.misc import is_dist_avail_and_initialized, nested_tensor_from_tensor_list

import sys
sys.path.append('..')

from utils import is_dist_avail_and_initialized, nested_tensor_from_tensor_list, get_world_size
from losses import reduce_masked_mean, balanced_mse_loss, SimpleLoss

def dice_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    inputs = inputs.sigmoid()  
    inputs = inputs.flatten(1)
    numerator = 2 * (inputs * targets).sum(-1)
    denominator = inputs.sum(-1) + targets.sum(-1)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss.sum() / num_masks


dice_loss_jit = torch.jit.script(
    dice_loss
)  # type: torch.jit.ScriptModule


def sigmoid_ce_loss(
        inputs: torch.Tensor,
        targets: torch.Tensor,
        num_masks: float,
    ):
    """
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    Returns:
        Loss tensor
    """

    #loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
    pt = torch.exp(-BCE_loss)  # Calculates pt as the probability of being classified correctly
    alpha = 0.25
    gamma = 2
    loss = alpha * (1 - pt) ** gamma * BCE_loss
    
    
    return loss.mean(1).sum() / num_masks


sigmoid_ce_loss_jit = torch.jit.script(
    sigmoid_ce_loss
)  # type: torch.jit.ScriptModule


def calculate_uncertainty(logits):
    """
    We estimate uncerainty as L1 distance between 0.0 and the logit prediction in 'logits' for the
        foreground class in `classes`.
    Args:
        logits (Tensor): A tensor of shape (R, 1, ...) for class-specific or
            class-agnostic, where R is the total number of predicted masks in all images and C is
            the number of foreground classes. The values are logits.
    Returns:
        scores (Tensor): A tensor of shape (R, 1, ...) that contains uncertainty scores with
            the most uncertain locations having the highest uncertainty score.
    """
    assert logits.shape[1] == 1
    gt_class_logits = logits.clone()
    return -(torch.abs(gt_class_logits))


class SetCriterion(nn.Module):
    """This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, num_queries, num_classes, matcher, weight_dict, eos_coef, losses,
                 num_points, oversample_ratio, importance_sample_ratio, use_pointwise_mask_loss):
        """Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_queries = num_queries
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[0] = self.eos_coef
        self.register_buffer("empty_weight", empty_weight)

        # pointwise mask loss parameters
        self.use_pointwise_mask_loss = use_pointwise_mask_loss
        self.num_points = num_points
        self.oversample_ratio = oversample_ratio
        self.importance_sample_ratio = importance_sample_ratio

    def loss_rt(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the vehicle rotation
        targets dicts must contain the key "rt" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_rt" in outputs
        src_rt = outputs["pred_rt"]            # [B, num_queries, 4]
        B = src_rt.shape[0]

        # batch_idx, query_idx
        src_idx = self._get_src_permutation_idx(indices)
        # batch_idx, vehicle_idx
        tgt_idx = self._get_tgt_permutation_idx(indices)

        src_rt = src_rt[src_idx]            # [sum(num_vehicles), 3]

        num_aux = src_rt.shape[1]
        
        # def WrapAngleRadians(angles_rad, min_val=-torch.pi, max_val=torch.pi):
        #     max_min_diff = max_val - min_val
        #     # Using torch.remainder for modulo operation as it's equivalent to tf.math.floormod
        #     return min_val + torch.remainder(angles_rad - min_val, max_min_diff)
        
        max_num_vehicle = max([t["rt"].shape[0] for t in targets])
        target_rts = torch.zeros(B, max_num_vehicle, num_aux, device=src_rt.device)         # [B, max(num_vehicles), 3]

        for i, tgt in enumerate(targets):
            num_vehicle = tgt["rt"].shape[0]
            target_rts[i, :num_vehicle] = tgt["rt"]

        tgt_rt = target_rts[tgt_idx]                                          # [sum(num_vehicles), 3]

        # heading_error = WrapAngleRadians(src_rt[:, -1] - tgt_rt[:, -1])       # [sum(num_vehicles)]
        # loss_heading = F.smooth_l1_loss(heading_error, torch.zeros_like(heading_error),reduction='mean',beta=1.0)

        loss_rt = F.smooth_l1_loss(src_rt, tgt_rt,
                                            reduction='mean', 
                                            beta=1.0)
        
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_rt),
            box_cxcywh_to_xyxy(tgt_rt)))

        loss_giou = loss_giou.mean()
        
        # loss_translation = F.smooth_l1_loss(src_rt[:, :-1], tgt_rt[:, :-1],
        #                                     reduction='mean', 
        #                                     beta=1.0)

        losses = {"loss_rt": loss_rt,
                  "loss_giou": loss_giou}
        return losses



    def loss_labels(self, outputs, targets, indices, num_masks):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert "pred_logits" in outputs
        src_logits = outputs["pred_logits"].float()     # [B, num_queries, 2]
        num_queries = src_logits.shape[1]

        idx = self._get_src_permutation_idx(indices)
        # target_classes_o = [1, 1, 1, 1, 1, 1, 1, 1]   # (sum(vehicle_num))
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # target_classes.shape = (B, num_queries) full of num_classes = 0
        target_classes = torch.full(
            src_logits.shape[:2], 0, dtype=torch.int64, device=src_logits.device
        )       # [B, num_queries]
        # fill the corresponding query indices to 1
        target_classes[idx] = target_classes_o.to(target_classes.device)    # [B, num_queries]
        # assume 5 queries and vehicles of 2, 3, where matched queries are [1, 3] and [2, 0, 4]
        # then target classes is
        # [[0, 1, 0, 1, 0]
        #  [1, 0, 1, 0, 1]]

        empty_weight = torch.ones(self.num_classes + 1).to(src_logits.device)
        empty_weight[0] = 0.8

        # give negative class a weight of 0.1
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), # [B, 2, num_queries]
                                  target_classes,
                                  empty_weight
                                  )
        
        # loss_bce = F.binary_cross_entropy_with_logits(
        #     src_logits.flatten(0, 1)[..., 1].unsqueeze(-1),  # [B*num_queries, 1]
        #     target_classes.transpose(0, 1).flatten(0, 1).float().unsqueeze(-1), # [B*num_queries, 1]
        #     pos_weight=torch.tensor([2.0]).to(src_logits.device)
        #     )
                                
        
        # log classification accuracy, precision, recall
        # src logits (not probabilities)
        pred_classes = src_logits.argmax(-1)        # [B, num_queries]. logit to class. 0 or 1
        B = pred_classes.shape[0]
        pred_classes = pred_classes.flatten()       # B*num_queries
        target_classes = target_classes.flatten()   # B*num_queries
        accuracy = (pred_classes == target_classes).sum() / (B*num_queries)

        precision = ((pred_classes == target_classes) & (pred_classes == 1)).sum() / (pred_classes == 1).sum()
        recall = ((pred_classes == target_classes) & (pred_classes == 1)).sum() / (target_classes == 1).sum()


        losses = {"loss_ce": loss_ce, #loss_ce,
                    "loss_ce_accuracy": accuracy,
                    "loss_ce_precision": precision,
                    "loss_ce_recall": recall
                    }
        
        return losses
    

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]                                                      # [sum(num_vehicles), 4]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)     # [sum(num_vehicles), 4]

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.mean() #loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_cxcywh_to_xyxy(src_boxes),
            box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.mean() #loss_giou.sum() / num_boxes
        return losses
    
    def loss_center_offset(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the center offset, 
        """
       
        center_bev = torch.stack([targets[i]['center_bev'] for i in range(len(targets))], dim=0).cuda() # [B, 1, X, Z]
        offset_bev = torch.stack([targets[i]['offset_bev'] for i in range(len(targets))], dim=0).cuda() # [B, 2, X, Z]
        seg_bev = torch.stack([targets[i]['gt_masks'] for i in range(len(targets))], dim=0).cuda() # [B, 1, X, Z]
        valid_bev = torch.stack([targets[i]['gt_valid'] for i in range(len(targets))], dim=0).cuda() # [B, 1, X, Z]
        
        center_bev_pred = outputs['bev_decoder_out']['instance_center'] # [B, 1, X, Z]
        offset_bev_pred = outputs['bev_decoder_out']['instance_offset'] # [B, 2, X, Z]   
        seg_bev_pred = outputs['bev_decoder_out']['segmentation']       # [B, 1, X, Z]

 
        center_loss = balanced_mse_loss(center_bev_pred, center_bev, valid_bev)
        offset_loss = torch.abs(offset_bev_pred-offset_bev).sum(dim=1, keepdim=True)
        offset_loss = reduce_masked_mean(offset_loss, seg_bev*valid_bev)

        loss_fn = SimpleLoss().cuda()
        ce_loss = loss_fn(seg_bev_pred, seg_bev, valid_bev)

        losses = {}
        losses['loss_center'] = center_loss
        losses['loss_offset'] = offset_loss
        losses['loss_ce_simple'] = ce_loss

        return losses
        
    

    def loss_masks(self, outputs, targets, indices, num_masks):
        """Compute the losses related to the masks: the focal loss and the dice loss.
        targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        # batch_idx, query_idx
        src_idx = self._get_src_permutation_idx(indices)
        # batch_idx, vehicle_idx
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]         # [B, num_queries, H, W]
        src_masks = src_masks[src_idx]            # [sum(num_vehicles), H, W]  | masks of matched queries
        masks = [t["masks"] for t in targets]     # list of tensors of shape (num_vehicles, H, W)

        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()  # [sum(num_vehicles), H, W]
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]    # [sum(num_vehicles), H, W]  | reorered target masks

        # No need to upsample predictions as we are using normalized coordinates :)
        # N x 1 x H x W
        src_masks = src_masks[:, None]
        target_masks = target_masks[:, None]

        if self.use_pointwise_mask_loss:

            with torch.no_grad():
                # sample point_coords
                point_coords = get_uncertain_point_coords_with_randomness(
                    src_masks,
                    lambda logits: calculate_uncertainty(logits),
                    self.num_points,
                    self.oversample_ratio,
                    self.importance_sample_ratio,
                )
                # get gt labels
                point_labels = point_sample(
                    target_masks,
                    point_coords,
                    align_corners=False,
                ).squeeze(1)

            point_logits = point_sample(
                src_masks,
                point_coords,
                align_corners=False,
            ).squeeze(1)

        else:

            point_logits = src_masks.flatten(1)
            point_labels = target_masks.flatten(1)

        losses = {
            "loss_mask": sigmoid_ce_loss_jit(point_logits, point_labels, num_masks),
            "loss_dice": dice_loss_jit(point_logits, point_labels, num_masks),
        }

        del src_masks
        del target_masks
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        # src --> query indices

        # batch_idx = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3 ...] where batch0 has 3 vehicles, batch1 has 5 vehicles
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])  

        # src_idx = [5, 14, 12, 5 ....] corresponding query indices
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        
        # batch_idx = [0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 3 ...] where batch0 has 3 vehicles, batch1 has 5 vehicles
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])

        # tgt_idx = [0, 2, 1, 3 ....] corresponding vehicle indices
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_masks):

        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'rt': self.loss_rt,
            'center_offset': self.loss_center_offset,
        }
        assert loss in loss_map, f"do you really want to compute {loss} loss?"
        return loss_map[loss](outputs, targets, indices, num_masks)


    def forward(self, outputs, targets, return_indices=False):
        """This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != "aux_outputs"}

        if self.num_queries == 2:
            indices = [(torch.tensor([1]), torch.tensor([0])) for _ in range(len(targets))]
        else:
            # Retrieve the matching between the outputs of the last layer and the targets
            indices = self.matcher(outputs_without_aux, targets)
        
        if return_indices:
            # batch_idx, query_idx
            src_idx = self._get_src_permutation_idx(indices)
            src_masks = outputs["pred_masks"]         # [B, num_queries, H, W]
            src_masks = src_masks[src_idx]            # [sum(num_vehicles), H, W]  | masks of matched queries
        
            return src_masks

        device = next(iter(targets[0].values())).device

        if self.num_queries == 2:
            # num masks is equal to the batch size
            num_masks = len(targets)
        
        else:

            # Compute the average number of target boxes accross all nodes, for normalization purposes
            #num_masks = sum(len(t["labels"]) for t in targets)  # total number of queries in the batch
            num_masks = sum(len(t["boxes"]) for t in targets)
            num_masks = torch.as_tensor(
                [num_masks], dtype=torch.float, device=device
            )
            if is_dist_avail_and_initialized():
                torch.distributed.all_reduce(num_masks)

            world_size = get_world_size()
            num_masks = torch.clamp(num_masks / world_size , min=1).item()
        
        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_masks))


        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if "aux_outputs" in outputs:
            for i, aux_outputs in enumerate(outputs["aux_outputs"]):
                if self.num_queries == 2:
                    indices = [(torch.tensor([1]), torch.tensor([0])) for _ in range(len(targets))]
                else:
                    indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss != "rt" and (not ("offset" in loss)):
                        l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_masks)
                        l_dict = {k + f"_{i}": v for k, v in l_dict.items()}
                        losses.update(l_dict)

        return losses

    def __repr__(self):
        head = "Criterion " + self.__class__.__name__
        body = [
            "matcher: {}".format(self.matcher),
            "losses: {}".format(self.losses),
            "weight_dict: {}".format(self.weight_dict),
            "num_classes: {}".format(self.num_classes),
            "eos_coef: {}".format(self.eos_coef),
            "num_points: {}".format(self.num_points),
            "oversample_ratio: {}".format(self.oversample_ratio),
            "importance_sample_ratio: {}".format(self.importance_sample_ratio),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
    





# Copyright (c) Facebook, Inc. and its affiliates.
import torch
from torch.nn import functional as F
# import Lİst from typing import List
from typing import List


"""
Shape shorthand in this module:

    N: minibatch dimension size, i.e. the number of RoIs for instance segmenation or the
        number of images for semantic segmenation.
    R: number of ROIs, combined over all images, in the minibatch
    P: number of points
"""


def point_sample(input, point_coords, **kwargs):
    """
    A wrapper around :function:`torch.nn.functional.grid_sample` to support 3D point_coords tensors.
    Unlike :function:`torch.nn.functional.grid_sample` it assumes `point_coords` to lie inside
    [0, 1] x [0, 1] square.

    Args:
        input (Tensor): A tensor of shape (N, C, H, W) that contains features map on a H x W grid.
        point_coords (Tensor): A tensor of shape (N, P, 2) or (N, Hgrid, Wgrid, 2) that contains
        [0, 1] x [0, 1] normalized point coordinates.

    Returns:
        output (Tensor): A tensor of shape (N, C, P) or (N, C, Hgrid, Wgrid) that contains
            features for points in `point_coords`. The features are obtained via bilinear
            interplation from `input` the same way as :function:`torch.nn.functional.grid_sample`.
    """
    add_dim = False
    if point_coords.dim() == 3:
        add_dim = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dim:
        output = output.squeeze(3)
    return output


def generate_regular_grid_point_coords(R, side_size, device):
    """
    Generate regular square grid of points in [0, 1] x [0, 1] coordinate space.

    Args:
        R (int): The number of grids to sample, one for each region.
        side_size (int): The side size of the regular grid.
        device (torch.device): Desired device of returned tensor.

    Returns:
        (Tensor): A tensor of shape (R, side_size^2, 2) that contains coordinates
            for the regular grids.
    """
    aff = torch.tensor([[[0.5, 0, 0.5], [0, 0.5, 0.5]]], device=device)
    r = F.affine_grid(aff, torch.Size((1, 1, side_size, side_size)), align_corners=False)
    return r.view(1, -1, 2).expand(R, -1, -1)


def get_uncertain_point_coords_with_randomness(
    coarse_logits, uncertainty_func, num_points, oversample_ratio, importance_sample_ratio
):
    """
    Sample points in [0, 1] x [0, 1] coordinate space based on their uncertainty. The unceratinties
        are calculated for each point using 'uncertainty_func' function that takes point's logit
        prediction as input.
    See PointRend paper for details.

    Args:
        coarse_logits (Tensor): A tensor of shape (N, C, Hmask, Wmask) or (N, 1, Hmask, Wmask) for
            class-specific or class-agnostic prediction.
        uncertainty_func: A function that takes a Tensor of shape (N, C, P) or (N, 1, P) that
            contains logit predictions for P points and returns their uncertainties as a Tensor of
            shape (N, 1, P).
        num_points (int): The number of points P to sample.
        oversample_ratio (int): Oversampling parameter.
        importance_sample_ratio (float): Ratio of points that are sampled via importnace sampling.

    Returns:
        point_coords (Tensor): A tensor of shape (N, P, 2) that contains the coordinates of P
            sampled points.
    """
    assert oversample_ratio >= 1
    assert importance_sample_ratio <= 1 and importance_sample_ratio >= 0
    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(num_boxes, num_sampled, 2, device=coarse_logits.device)
    point_logits = point_sample(coarse_logits, point_coords, align_corners=False)
    # It is crucial to calculate uncertainty based on the sampled prediction value for the points.
    # Calculating uncertainties of the coarse predictions first and sampling them for points leads
    # to incorrect results.
    # To illustrate this: assume uncertainty_func(logits)=-abs(logits), a sampled point between
    # two coarse predictions with -1 and 1 logits has 0 logits, and therefore 0 uncertainty value.
    # However, if we calculate uncertainties for the coarse predictions first,
    # both will have -1 uncertainty, and the sampled point will get -1 uncertainty.
    point_uncertainties = uncertainty_func(point_logits)
    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(point_uncertainties[:, 0, :], k=num_uncertain_points, dim=1)[1]
    shift = num_sampled * torch.arange(num_boxes, dtype=torch.long, device=coarse_logits.device)
    idx += shift[:, None]
    point_coords = point_coords.view(-1, 2)[idx.view(-1), :].view(
        num_boxes, num_uncertain_points, 2
    )
    if num_random_points > 0:
        point_coords = cat(
            [
                point_coords,
                torch.rand(num_boxes, num_random_points, 2, device=coarse_logits.device),
            ],
            dim=1,
        )
    return point_coords




def cat(tensors: List[torch.Tensor], dim: int = 0):
    """
    Efficient version of torch.cat that avoids a copy if there is only a single element in a list
    """
    assert isinstance(tensors, (list, tuple))
    if len(tensors) == 1:
        return tensors[0]
    return torch.cat(tensors, dim)




# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / union
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / area


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)
