
# # prepare arguments class
# class Arguments:

#     def __init__(self):
        
#         self.project = 'mask2former4bev'
#         self.model_name = 'mask2former4bev'
#         self.dataset_path = '/datasets/nuscenes'
#         self.version = 'trainval'

#         # write all the parameters like above
#         self.res_scale = 1
#         self.H = 1600
#         self.W = 900
#         self.resize_to = [224, 448]
#         self.crop_offset = 0
#         self.random_flip = 0
#         self.resize_lim = [1.0, 1.0]
#         self.rand_crop_and_resize = 0
#         self.cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
#         self.ncams = 6

#         self.do_shuffle_cams = 0
#         self.refcam_id = 1

#         self.backbone = "res101-simplebev"
#         self.freeze_backbone = 0
#         self.patch_size = 16

#         self.mask_classification = 1
#         self.class_weight = 1.0
#         self.dice_weight = 1.0
#         self.mask_weight = 20.0
#         self.no_object_weight = 0.1
#         self.deep_supervision = 1

#         self.train_num_points = 112*112
#         self.oversample_ratio = 3.0
#         self.importance_sample_ratio = 0.75

#         self.sem_seg_head_name = 'mask_former_head'
#         self.transformer_in_feature = 'multi_scale_bev_features'

#         self.bev_module_name = 'SimpleBEV'
#         self.bev_latent_dim = 128
#         self.multiscale_feature_channels = [64, 128, 256]
#         self.multiscale_feature_norm = 'batch'
#         self.multiscale_conv_dim = 256
#         self.voxel_size = [200, 8, 200]
#         self.bounds = [-50, 50, -5, 5, -50, 50]
#         self.do_rgb_compress = 1

#         self.use_frozen_bev_feats = 0
#         self.frozen_bev_feats_path = '/kuacc/users/mbarin22/hpc_run/mask2former4bev/checkpoints/simplebev/8x5_5e-4_rgb12_22:43:46/model-000025000.pth'

#         self.num_classes = 1

#         self.predictor_type = 'TransformerPredictor'
#         self.nheads = 8
#         self.dec_layers = 6
#         self.pe_hidden_dim = 256
#         self.predictor_dropout = 0
#         self.num_queries = 105
#         self.pre_norm = 0
#         self.dim_feedforward = 2048
#         self.enforce_input_project = 0
#         self.mask_dim = 256

#         self.use_lidar = 1

#         self.decoder_type = 'conv'

#         self.learning_rate = 4e-4
#         self.weight_decay = 1e-7
#         self.dropout = 0.0




# ckpt_path = "/kuacc/users/mbarin22/hpc_run/mask2former4bev/checkpoints/backbone:res101-simplebev_cls:1_pts:12544_bevfeats:0_2k/best.pt"
# from read_args import get_args
# args = get_args() #args = Arguments()

# import utils
# utils.init_distributed_mode(args)
# import numpy as np
# import random
# import os
# import torch
# seed = args.seed

# torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# #torch.backends.cudnn.deterministic = True
# #torch.backends.cudnn.benchmark = False
# np.random.seed(seed)
# random.seed(seed)
# os.environ['PYTHONHASHSEED'] = str(seed)
# # from dataset import NuScenesDatasetWrapper
# # datamodule = NuScenesDatasetWrapper(args)
# # valset = datamodule.val()

# from models.mask2former4bev import Mask2Former4BEV
# from collections import OrderedDict
# import torch
# import torch.nn as nn

# #model = Mask2Former4BEV(args).cuda()
# model = utils.init_model(args)
# model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
# # loaded_model = OrderedDict((key.replace('module.', ''), value) for key, value in torch.load(ckpt_path)["model"].items()) 

# # model.load_state_dict(loaded_model) #, strict=False)
# to_restore = {"epoch": 0}

# args.checkpoint_path = ckpt_path
# utils.restart_from_checkpoint(args, 
#                                 run_variables=to_restore, 
#                                 model=model)

# total = 0

# #backbone:res101-simplebev_cls:1_pts:12544_bevfeats:0_2k/best.pt --> 2.030
# #backbone:res101-simplebev_cls:1_pts:12544_bevfeats:1_2k/best.pt --> 23.018


# from evaluator import Evaluator

# evaluator = Evaluator() 

# val_dataloader =  utils.get_dataloaders(args)[1]



# print('val dataloader length:', len(val_dataloader))
# # first epoch : 12.991

# from tqdm import tqdm

# model.eval()

# val_loader = tqdm(val_dataloader)

# total_loss = 0

# for i, batch in enumerate(val_loader):

#     with torch.cuda.amp.autocast(True):

#         preds = model(batch, training=False)  # a dictionary of losses

#         for idx, pred in enumerate(preds):
#             pred_masks = pred['pred_masks'].detach().cpu()
#             gt_masks = batch[idx]['gt_masks'].cpu()
#             valid_masks = batch[idx]['gt_valid'].cpu()

#             # ===  Segmentation Evaluation ===
#             miou = evaluator.update(pred_masks, gt_masks, valid_masks)['mIoU']
    
#     metric_desc = f"mIoU: {miou * 100:.3f}"

#     # === Logger ===
#     val_loader.set_description(metric_desc)
#     # === === ===

# # === Evaluation Results ====
# miou = evaluator.get_results()['mIoU']
# print('miou:', miou)

import torch
import sys
import matplotlib.pyplot as plt
import torch
from collections import OrderedDict

# prepare arguments class
class Arguments:

    def __init__(self):
        
        
        self.is_loss_weights_param = 0
        self.project = 'mask2former4bev'
        self.model_name = 'mask2former4bev'
        self.dataset_path = '/datasets/nuscenes'
        self.version = 'trainval'

        # write all the parameters like above
        self.res_scale = 1
        self.H = 1600
        self.W = 900
        self.rand_crop_and_resize = 0
        
        self.resize_to = [448,800]
        self.crop_offset = 0
        self.random_flip = 0
        self.resize_lim = [1.0, 1.0]
        self.cams = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        self.ncams = 6

        self.do_shuffle_cams = 0
        self.refcam_id = 1

        self.backbone = "res101-simplebev"
        self.freeze_backbone = 0
        self.patch_size = 16

        self.mask_classification = 1
        self.class_weight = 1.0
        self.dice_weight = 1.0
        self.mask_weight = 20.0
        self.no_object_weight = 0.1
        self.deep_supervision = 1

        self.train_num_points = 112*112
        self.oversample_ratio = 3.0
        self.importance_sample_ratio = 0.75

        self.sem_seg_head_name = 'mask_former_head'
        self.transformer_in_feature = 'multi_scale_bev_features'

        self.bev_module_name = 'SimpleBEV'
        self.bev_latent_dim = 128
        self.multiscale_feature_channels = [64, 128, 256]
        self.multiscale_feature_norm = 'batch'
        self.multiscale_conv_dim = 256
        self.voxel_size = [200, 8, 200]
        self.bounds = [-50, 50, -5, 5, -50, 50]
        self.do_rgb_compress = 1

        self.use_frozen_bev_feats = 0
        self.frozen_bev_feats_path = '/kuacc/users/mbarin22/hpc_run/mask2former4bev/checkpoints/simplebev/8x5_5e-4_rgb12_22:43:46/model-000025000.pth'

        self.num_classes = 1

        self.predictor_type = 'TransformerPredictor'
        self.nheads = 8
        self.dec_layers = 6
        self.pe_hidden_dim = 256
        self.predictor_dropout = 0
        self.num_queries = 100
        self.pre_norm = 0
        self.dim_feedforward = 2048
        self.enforce_input_project = 0
        self.mask_dim = 256
        
        self.use_multiscale_features = 1
        self.rt_regression = 0
        self.translation_weight =1
        self.heading_weight = 1

        self.use_lidar = 1

        self.decoder_type = 'conv'

        self.learning_rate = 4e-4
        self.weight_decay = 1e-7
        self.dropout = 0.0

        self.use_center_offset_loss = 1
        self.center_offset_weight = 0.5
        self.inference = 0
        
        self.validate_with_gt = False

        self.get_sem_masks = 0

args = Arguments()

ckpt_path = "/kuacc/users/mbarin22/hpc_run/mask2former4bev/checkpoints/[448, 800]_bs:16_nq:100_backbone:res101-simplebev_offset-center:0.5-bce:1.0_cls:1-empty:0.8_pts:12544_20k/best.pt"

from dataset import NuScenesDatasetWrapper

datamodule = NuScenesDatasetWrapper(args)

from models.mask2former4bev import Mask2Former4BEV

model = Mask2Former4BEV(args).cuda()
loaded_model = OrderedDict((key.replace('module.', ''), value) for key, value in torch.load(ckpt_path)["model"].items()) 

model.load_state_dict(loaded_model,strict=True)
model.eval()

valset = datamodule.train()


class Evaluator:
    def __init__(self, threshold):
        self.reset()
        self.threshold = threshold

    def reset(self):
        print("Resetting Evaluator")
        self.mious = []

    def calculate_miou(self, pred, gt, valid):
        
        pred_round = (pred > self.threshold).float()  
        if valid is None:
            valid = torch.ones_like(gt)
            
        intersection = (pred_round*gt*valid).sum(dim=[1,2])
        union = ((pred_round+gt)*valid).clamp(0,1).sum(dim=[1,2])
        iou = (intersection/(1e-4 + union)).mean()

        # batch mean
        self.mious.append(iou.item())

    
    def update(self, pred, gt, valid=None):

        self.calculate_miou(pred, gt, valid)

        last_miou = self.mious[-1]

        results = self.get_results(reset=False)

        results['last_mIoU'] = last_miou

        return results

    def get_results(self, reset=True):

        miou = sum(self.mious) / len(self.mious)

        if reset:
            self.reset()
        
        return {'mIoU' : miou}
    

from tqdm import tqdm
valset_  = tqdm(range(len(valset)))

threshold_results = {}

for threshold in [0.5][::-1]: #,0.8,0.9
    print(f'Evaluating {threshold}')
    our_evaluator = Evaluator(threshold) 
    valset_  = tqdm(range(len(valset)))
    
    for i in valset_: 

        batch = valset[i]

        preds = model([batch], training=False)  

        for idx, pred in enumerate(preds):
            pred_masks = pred['pred_masks'].detach().cpu()
            gt_masks = batch['gt_masks'].cpu()
            valid_masks = batch['gt_valid'].cpu()

            assert pred_masks.shape == (1, 200, 200)
            assert gt_masks.shape == (1, 200, 200)
            assert valid_masks.shape == (1, 200, 200)

            # ===  Segmentation Evaluation ===
            us_miou = our_evaluator.update(pred_masks, gt_masks, valid_masks)
            miou = us_miou['mIoU']
            last_miou = us_miou['last_mIoU']

        metric_desc = f"mIoU: {miou * 100:.3f} | last mIoU: {last_miou * 100:.3f}"
        # === Logger ===
        valset_.set_description(metric_desc)
        
    threshold_results[threshold] = our_evaluator.get_results()['mIoU']
    print(f'mIoU_{threshold}: {threshold_results[threshold]}')

    # import pickle 
    # with open('threshold_results.pkl', 'wb') as f:
    #     pickle.dump(threshold_results, f) 