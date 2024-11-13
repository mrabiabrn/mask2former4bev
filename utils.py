import os
import sys
import math
import random
import numpy as np

import wandb

import torch
import torch.distributed as dist

from dataset import NuScenesDatasetWrapper

from typing import List, Optional

import torch
import torch.distributed as dist
import torchvision
from torch import Tensor



def get_run_name(args):

    num_steps = str(args.num_steps//1000) + 'k'

    if args.backbone == 'swin':
        output = args.output_layer
    else:
        output = ''

    if args.get_sem_masks:
        run_name = f'{args.resize_to}_bs:{args.batch_size}_latentdim:{args.bev_latent_dim}_nq:{args.num_queries}_backbone:{args.backbone}-rgb_{num_steps}_sem_masks_20ktrain'
    
    elif args.predictor_type == 'SimpleTransformerPredictor':
        run_name = f'simple-{args.resize_to}_bs:{args.batch_size}_nq:{args.num_queries}_backbone:{args.backbone}_pts:{args.train_num_points}_{num_steps}'
    elif args.model_name == 'mask2former4bev':
        run_name = f'{args.resize_to}_bs:{args.batch_size}_nq:{args.num_queries}_backbone:{args.backbone}_cls:{args.mask_classification}-center-offset:0.5-bce:1.0-empty:0.8_pts:{args.train_num_points}_{num_steps}'
    elif args.model_name == 'detr4bev':
        run_name = f'detr_pretrained-bev_{args.resize_to}_bs:{args.batch_size}_nq:{args.num_queries}_{num_steps}_{args.learning_rate}_overfit'
    
    return run_name


# === ================ ===
# === Model Related ===

def init_model(args):

    model_name = args.model_name
    
    if model_name == 'mask2former4bev':
        from models.mask2former4bev import Mask2Former4BEV
        model = Mask2Former4BEV(args)
    elif model_name == 'detr4bev':
        from models.mask2former4bev import DETR4BEV
        model = DETR4BEV(args)
    else:
        raise NotImplementedError
    
    return model.cuda()

# === ================ ===
# === Logger Related ===

def init_logger(args,run_name):

    project_name = args.project
    if args.validate:
        project_name += '_val'
        run_name = args.checkpoint_path.split('/')[-2]

        if args.validate_on_train:
            run_name +='_train'

    wandb.init(
                project=project_name, 
                name=run_name,
                settings=wandb.Settings(_service_wait=300)
                ) 

    wandb.define_metric('train steps')
    wandb.define_metric('total iter',step_metric='train steps')
    wandb.define_metric('loss',step_metric='train steps')
    wandb.define_metric('lr',step_metric='train steps')
    wandb.define_metric('loss_mask',step_metric='train steps')
    wandb.define_metric('loss_dice',step_metric='train steps')
    wandb.define_metric('loss_ce',step_metric='train steps')
    
    wandb.define_metric('bev loss',step_metric='train steps')
    wandb.define_metric('img loss',step_metric='train steps')

    wandb.define_metric('epoch')
    wandb.define_metric('train loss',step_metric='epoch')
    wandb.define_metric('val loss',step_metric='epoch')
    wandb.define_metric('mIoU',step_metric='epoch')



# === ================ ===
# === Training Related ===

def restart_from_checkpoint(args, run_variables, **kwargs):

    checkpoint_path = args.checkpoint_path

    assert checkpoint_path is not None
    assert os.path.exists(checkpoint_path)

    # open checkpoint file
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # key is what to look for in the checkpoint file
    # value is the object to load
    # example: {'state_dict': model}
    for key, value in kwargs.items():
        if key in checkpoint and value is not None and checkpoint[key] is not None:
            try:
                msg = value.load_state_dict(checkpoint[key], strict=False)
                print("=> loaded '{}' from checkpoint with msg {}".format(key, msg))
            except TypeError:
                try:
                    msg = value.load_state_dict(checkpoint[key])
                    print("=> loaded '{}' from checkpoint".format(key))
                except ValueError:
                    print("=> failed to load '{}' from checkpoint".format(key))
        else:
            print("=> key '{}' not found in checkpoint".format(key))

    # re load variable important for the run
    if run_variables is not None:
        for var_name in run_variables:
            if var_name in checkpoint:
                run_variables[var_name] = checkpoint[var_name]



def custom_collate_fn(batch):
    
    return batch


def get_dataloaders(args):
    
    datamodule = NuScenesDatasetWrapper(args)

    trainset = datamodule.train() 
    valset = datamodule.val()

    if args.overfit:
        import random

        train_indices = list(range(20000)) #random.sample(list(range(10000)),k=1000) #1000)list(range(len(trainset)))
        val_indices = list(range(1000)) #random.sample(list(range(3000)), k=200)  #list(range(len(valset)))
        trainset = torch.utils.data.Subset(trainset, train_indices)
        valset = torch.utils.data.Subset(valset, val_indices)
        #trainset = datamodule.val_dataset
        #valset = trainset

    # train_indices = list(range(15000)) 
    # trainset = torch.utils.data.Subset(trainset, train_indices)
    # val_indices = list(range(1000))
    # valset = torch.utils.data.Subset(valset, val_indices)

    train_sampler = torch.utils.data.DistributedSampler(trainset, num_replicas=args.gpus, rank=args.gpu, shuffle=True)
    train_dataloader = torch.utils.data.DataLoader(
        trainset,
        sampler=train_sampler,
        batch_size=args.batch_size // args.gpus,
        collate_fn=custom_collate_fn,
        num_workers=6,      # cpu per gpu
        drop_last=True,
        pin_memory=True,
    )

    if args.validate_on_train:
        valset = trainset

    val_dataloader = torch.utils.data.DataLoader(valset, 
        batch_size=1, #args.batch_size // args.gpus, #1,
        collate_fn=custom_collate_fn,
        shuffle=False, 
        num_workers=6, 
        drop_last=False, 
        pin_memory=True)
     
    return train_dataloader, val_dataloader



def get_scheduler(args, optimizer, train_loader, T_max=None, repeat=1):

    if T_max is None:
        T_max = len(train_loader) * args.num_epochs

    warmup_steps = int(T_max * 0.05)
    steps = T_max - warmup_steps
    gamma = math.exp(math.log(0.5) / (steps // 3))

    linear_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-5, total_iters=warmup_steps)
    # if args.scheduler == 'exponential':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    # else:
    #     scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, 
    #                               args.learning_rate, 
    #                               total_steps=steps, 
    #                               cycle_momentum=False, 
    #                               div_factor=10, 
    #                               final_div_factor=10
    #                               )
        
            
    total_cycle = args.num_epochs * T_max
    #scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(total_cycle * 0.7), int(total_cycle * 0.9)], gamma=0.15)

    scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear_scheduler, scheduler], milestones=[warmup_steps])
    return scheduler


class EarlyStopping:
    def __init__(self, patience=5, mode='max'):
        self.patience = patience
        self.mode = mode
        self.best_metric = None
        self.counter = 0
        self.early_stop = False

        if mode not in ['min', 'max']:
            raise ValueError("Mode should be 'min' or 'max'.")

    def __call__(self, metric):

        if self.best_metric is None:
            self.best_metric = metric
            return False

        if (self.mode == 'max' and metric > self.best_metric) or (self.mode == 'min' and metric < self.best_metric):
            self.best_metric = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop
    

# === ================ ===
# ===  Related ===
    
def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

class NestedTensor(object):
    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device):
        # type: (Device) -> NestedTensor # noqa
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)
    
# _onnx_nested_tensor_from_tensor_list() is an implementation of
# nested_tensor_from_tensor_list() that is supported by ONNX tracing.
@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(tensor_list: List[Tensor]) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    # work around for
    # pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
    # m[: img.shape[1], :img.shape[2]] = False
    # which is not yet supported in onnx
    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(img, (0, padding[2], 0, padding[1], 0, padding[0]))
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(m, (0, padding[2], 0, padding[1]), "constant", 1)
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)


def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    # TODO make this more general
    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


# === ===================== ===
# ===  Distributed Settings ===

def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    # From https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L467C1-L499C42

    # launched with torch.distributed.launch
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])

    # launched with submitit on a slurm cluster
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.world_size = args.gpus
        args.gpu = args.rank % torch.cuda.device_count()

    # launched naively with `python train.py`
    # we manually add MASTER_ADDR and MASTER_PORT to env variables
    elif torch.cuda.is_available():
        print('Will run the code on one GPU.')
        args.rank, args.gpu, args.world_size = 0, 0, 1
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        os.environ['MASTER_PORT'] = '29500'
    else:
        print('Does not support training without GPU.')
        sys.exit(1)

    dist.init_process_group(
        backend="nccl",
        init_method='env://',
        world_size=args.world_size,
        rank=args.rank,
    )

    torch.cuda.set_device(args.gpu)
    print('| distributed init (rank {}): {}'.format(
        args.rank, 'env://'), flush=True)
    dist.barrier()
    setup_for_distributed(args.rank == 0)

def setup_for_distributed(is_master):
    # From https://github.com/facebookresearch/dino/blob/7c446df5b9f45747937fb0d72314eb9f7b66930a/utils.py#L452C1-L464C30
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print

def fix_random_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)
    # np.random.seed(seed)