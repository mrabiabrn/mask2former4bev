import sys
import math
import argparse

import torch


def set_remaining_args(args):
    args.gpus = torch.cuda.device_count()


def print_args(args):

    print("====== Training ======")
    print(f"project name: {args.project}\n")

    print(f"resize_to: {args.resize_to}\n")

    print(f"model: {args.model_name}\n")

    print(f"learning_rate: {args.learning_rate}")
    print(f"batch_size: {args.batch_size}")
    #print(f"num_epochs: {args.num_epochs}")
    print(f"num_steps: {args.num_steps}")

    print("====== ======= ======\n")

def get_args():
    parser = argparse.ArgumentParser("Mask2Former4BEV")

    parser.add_argument('--project', type=str, default='mask2former4bev')
    parser.add_argument('--model_name', type=str, default='mask2former4bev')
    

    # === Data Related Parameters ===
    parser.add_argument('--dataset_path', type=str, default='/datasets/nuscenes') #'/home/mbarin/Downloads/v1.0-mini') #'/datasets/nuscenes')
    parser.add_argument('--dataset_version', type=str, default='trainval')
    parser.add_argument('--version', type=str, default='trainval')
    parser.add_argument('--res_scale', type=int, default=1)
    parser.add_argument('--H', type=int, default=1600)
    parser.add_argument('--W', type=int, default=900)
    parser.add_argument('--resize_to',  nargs='+', type=int, default=[224, 400])
    parser.add_argument('--rand_crop_and_resize', type=int, default=0)
    parser.add_argument('--random_flip', type=int, default=0)
    parser.add_argument('--cams',  nargs='+', type=str, default=['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'])
    parser.add_argument('--ncams', type=int, default=6)
    parser.add_argument('--do_shuffle_cams', type=int, default=0)
    parser.add_argument('--refcam_id', type=int, default=1)
    parser.add_argument('--get_sem_masks', action="store_true")

    # === Log Parameters ===
    parser.add_argument('--log_freq', type=int, default=10)

    # === Backbone Related Parameters ===
    parser.add_argument('--backbone', type=str, default="deit_tiny", 
                        choices=["res50","res101-simplebev", "dino-vitb-16","swin","deit_tiny","cnn","dinov2"])
    parser.add_argument('--freeze_backbone', type=int, default=0),
    parser.add_argument('--output_layer', type=str, default='res4')
    
    # === DINO Related Parameters ===
    parser.add_argument('--patch_size', type=int, default=16)

    # === Model Related Parameters ===
    parser.add_argument('--is_loss_weights_param', type=int, default=0)
    parser.add_argument('--mask_classification', type=int, default=1)
    parser.add_argument('--rt_regression', type=int, default=0)
    parser.add_argument('--translation_weight', type=float, default=0.01)
    parser.add_argument('--heading_weight', type=float, default=0.1)
    parser.add_argument('--class_weight', type=float, default=1.0)
    parser.add_argument('--dice_weight', type=float, default=1.0)
    parser.add_argument('--mask_weight', type=float, default=20.0)
    parser.add_argument('--center_offset_weight', type=float, default=0.5)
    parser.add_argument('--no_object_weight', type=float, default=0.1)
    parser.add_argument('--deep_supervision', type=int, default=1)
    parser.add_argument('--train_num_points', type=int, default=112*112)
    parser.add_argument('--oversample_ratio', type=float, default=3.0)
    parser.add_argument('--importance_sample_ratio', type=float, default=0.75)

    # === Head Related Parameters ===
    parser.add_argument('--sem_seg_head_name', type=str, default='mask_former_head')
    parser.add_argument('--transformer_in_feature', type=str, default='multi_scale_bev_features')

    # === BEV Related Parameters ===
    
    parser.add_argument('--bev_module_name', type=str, default='SimpleBEV')
    parser.add_argument('--bev_latent_dim', type=int, default=128)
    parser.add_argument('--use_multiscale_features', type=int, default=1)
    parser.add_argument('--multiscale_feature_channels',  nargs='+', type=int, default=[64, 128, 256])
    parser.add_argument('--multiscale_feature_norm', type=str, default='batch', choices=['batch', 'instance', 'group', 'layer', ''])
    parser.add_argument('--multiscale_conv_dim', type=int, default=256)
    parser.add_argument('--voxel_size',  nargs='+', type=int, default=[200, 8, 200])
    parser.add_argument('--bounds', nargs='+', type=int, default=[-50, 50, -5, 5, -50, 50])
    parser.add_argument('--do_rgb_compress', type=int, default=1)
    parser.add_argument('--use_center_offset_loss', type=int, default=1)

    parser.add_argument('--use_frozen_bev_feats', type=int, default=0)
    parser.add_argument('--frozen_bev_feats_path', type=str, default='/kuacc/users/mbarin22/hpc_run/mask2former4bev/checkpoints/simplebev/8x5_5e-4_rgb12_22:43:46/model-000025000.pth') 
    
    parser.add_argument('--num_classes', type=int, default=1)
    
    # === Predictor Related Parameters ===
    parser.add_argument('--predictor_type', type=str, default='TransformerPredictor')
    parser.add_argument('--nheads', type=int, default=8)
    parser.add_argument('--dec_layers', type=int, default=6)
    parser.add_argument('--pe_hidden_dim', type=int, default=256)
    parser.add_argument('--predictor_dropout', type=float, default=0)
    parser.add_argument('--num_queries', type=int, default=100)
    parser.add_argument('--pre_norm', type=int, default=0)
    parser.add_argument('--dim_feedforward', type=int, default=2048)
    parser.add_argument('--enforce_input_project', type=int, default=0)
    parser.add_argument('--mask_dim', type=int, default=256)


    # === Supervised Parameters ===
    parser.add_argument('--use_lidar', type=int, default=1)
    
    # === Decoder Parameters ===
    parser.add_argument('--decoder_type', type=str, default='conv', choices=['conv', 'simple'])
    
    # === Training Related Parameters ===
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-7)
    parser.add_argument('--dropout', type=float, default=0.0)

    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_steps', type=int, default=20000)
    parser.add_argument('--scheduler', type=str, default='onecycle')

   
    parser.add_argument('--seed', type=int, default=41)
    parser.add_argument('--model_save_path', type=str, default='/kuacc/users/mbarin22/hpc_run/mask2former4bev/checkpoints/')

    # === Overfit Parameters === #
    parser.add_argument('--overfit', action="store_true")
    parser.add_argument('--n_overfit_samples', type=int, default=4)

    # === Misc ===
    parser.add_argument('--save_epoch', type=int, default=1)
    parser.add_argument('--validate', action="store_true")
    parser.add_argument('--validate_with_gt', action="store_true")
    parser.add_argument('--validate_on_train', action="store_true")
    parser.add_argument('--eval_on_unseen', action="store_true")
    parser.add_argument('--use_checkpoint', action="store_true")
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--validation_epoch', type=int, default=1)
    parser.add_argument('--inference',type=str, default='sem_seg')

    args = parser.parse_args()

    set_remaining_args(args)

    return args