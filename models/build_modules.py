from .backbone.simplebev_encoder import SimpleBEVEncoder
from .backbone.swin_v2 import SwinEncoder
from .backbone.deit_tiny import DeiTTiny
from .backbone.cnn import CNNEncoder
from .backbone.dino import DINOEncoder

from .meta_arch.mask_former_head import MaskFormerHead
from .meta_arch.detr_head import DETRHead


def build_backbone(args):

    if args.backbone == 'res101-simplebev':
        return SimpleBEVEncoder(args)
    elif args.backbone == 'swin':
        return SwinEncoder(args)
    elif args.backbone == 'deit_tiny':
        return DeiTTiny(args)
    elif args.backbone == 'cnn':
        return CNNEncoder(args)
    elif args.backbone == 'dinov2':
        return DINOEncoder(args)
    else:
        raise NotImplementedError


def build_sem_seg_head(args):

    if args.sem_seg_head_name == 'mask_former_head':
        return MaskFormerHead(args)
    elif args.sem_seg_head_name == 'detr_head':
        return DETRHead(args)
    else:
        raise NotImplementedError
    



