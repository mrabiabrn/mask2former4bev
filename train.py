import os
import time
import math

import wandb
import random
import datetime
import numpy as np

import torch
import torch.nn as nn

import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from tqdm import tqdm

from read_args import get_args, print_args

import utils
import evaluator as evalmachine




def train_epoch(args, model, loss_fn, optimizer, scheduler, train_dataloader, total_iter):
    
    total_loss = 0 #torch.tensor(0.0, requires_grad=True).to(model.device)
    
    model.train()

    loader = tqdm(train_dataloader, disable=(args.gpu != 0))

    model.training = True

    evaluator = evalmachine.Evaluator() if args.overfit else None

    for i, batch in enumerate(loader):

        # batch is list of dicts. convert all values to cuda if the value is tensor
        # for k in batch:
        #     if isinstance(batch[k], torch.Tensor):
        #         batch[k] = batch[k].cuda(non_blocking=True)
        #batch[0]['images'] = batch[0]['images'].cuda(non_blocking=True)

        # fp16
        # with autocast():

        with torch.cuda.amp.autocast(True):

        
            losses, preds = model(batch)  # a dictionary of losses

            # sum all losses excluding the loss_ce_accuracy, loss_ce_precision, loss_ce_recall

            loss = sum([v for k, v in losses.items() if 'loss_ce_accuracy' not in k and 'loss_ce_precision' not in k and 'loss_ce_recall' not in k])

            # if args.overfit:
            #     for idx, pred in enumerate(preds):
            #         pred_masks = pred['pred_masks'].detach().cpu()
            #         gt_masks = batch[idx]['gt_masks'].cpu()
            #         valid_masks = batch[idx]['gt_valid'].cpu()

            #         # ===  Segmentation Evaluation ===
            #         miou = evaluator.update(pred_masks, gt_masks, valid_masks)['mIoU']
            
        
        
        optimizer.zero_grad(set_to_none=True)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()

        if args.gpu == 0:
            lr = optimizer.state_dict()["param_groups"][0]["lr"]
            mean_loss = total_loss / (i + 1)

            desc = f"lr: {lr:.6f} | loss: {mean_loss:.5f}"
            # if args.overfit:
            #     desc += f" | mIoU: {miou * 100:.3f}"
            loader.set_description(desc)

            log_info = {}
            if 'loss_mask' in losses.keys():
                log_info['loss_mask'] = losses['loss_mask'].item()
                log_info['loss_dice'] = losses['loss_dice'].item()
            
            if 'loss_ce' in losses.keys():
                log_info['loss_ce'] = losses['loss_ce'].item()
                log_info['loss_ce_accuracy'] = losses['loss_ce_accuracy'].item()
                log_info['loss_ce_precision'] = losses['loss_ce_precision'].item()
                log_info['loss_ce_recall'] = losses['loss_ce_recall'].item()

            if 'loss_center' in losses.keys():
                log_info['loss_center'] = losses['loss_center'].item()
                log_info['loss_offset'] = losses['loss_offset'].item()
                log_info['loss_ce_simple'] = losses['loss_ce_simple'].item()

            if 'loss_heading' in losses.keys():
                log_info['loss_heading'] = losses['loss_heading'].item()
                log_info['loss_translation'] = losses['loss_translation'].item()
            if 'loss_rt' in losses.keys():
                log_info['loss_rt'] = losses['loss_rt'].item()
            if 'loss_giou' in losses.keys():
                log_info['loss_giou'] = losses['loss_giou'].item()
            if 'loss_bbox' in losses.keys():   
                log_info['loss_bbox'] = losses['loss_bbox'].item()
            if 'loss_mask_uncertainty' in losses.keys():
                log_info['loss_mask_uncertainty'] = losses['loss_mask_uncertainty'].item()
                log_info['loss_dice_uncertainty'] = losses['loss_dice_uncertainty'].item()
            if 'loss_ce_uncertainty' in losses.keys():
                log_info['loss_ce_uncertainty'] = losses['loss_ce_uncertainty'].item()
            if 'loss_heading_uncertainty' in losses.keys():
                log_info['loss_heading_uncertainty'] = losses['loss_heading_uncertainty'].item()
                log_info['loss_translation_uncertainty'] = losses['loss_translation_uncertainty'].item()

            log_info['train steps'] = total_iter
            log_info['total iter'] = total_iter
            log_info['loss'] = mean_loss
            log_info['lr'] = lr
            # if args.overfit:
            #     log_info['train/mIoU'] = miou

            wandb.log(log_info)
        
        total_iter += 1

    mean_loss = total_loss / (i + 1)
    return mean_loss, total_iter




@torch.no_grad()
def val_epoch(args, model, loss_fn, val_dataloader, evaluator, log=False):

    model.eval()

    val_loader = tqdm(val_dataloader)
    
    total_loss = 0

    for i, batch in enumerate(val_loader):

        with torch.cuda.amp.autocast(True):

            preds = model(batch,training=False)  # a dictionary of losses

            #loss = sum([v for k, v in losses.items() if 'loss_ce_accuracy' not in k and 'loss_ce_precision' not in k and 'loss_ce_recall' not in k])


        if args.validate_with_gt:
            src_masks = preds.detach().cpu()
            miou = evaluator.update(src_masks, batch[0]['gt_masks'].cpu(), batch[0]['gt_valid'].cpu())['mIoU']

        else:
            for idx, pred in enumerate(preds):
                pred_masks = pred['pred_masks'].detach().cpu()
                gt_masks = batch[idx]['gt_masks'].cpu()
                valid_masks = batch[idx]['gt_valid'].cpu()

                # ===  Segmentation Evaluation ===
                miou = evaluator.update(pred_masks, gt_masks, valid_masks)['mIoU']
            
        metric_desc = f"mIoU: {miou * 100:.3f}"

        # === Logger ===
        val_loader.set_description(metric_desc)
        # === === ===

    # === Evaluation Results ====
    miou = evaluator.get_results()['mIoU']
    total_loss =  total_loss / (i+1)

    if log:
        wandb.log({
            'epoch': i,
            'val loss': total_loss / (i+1),
            'mIoU': miou
        })

    # === Logger ===
    # TODO: visualizations
    print("\n=== Results ===")
    print(f"\tmIoU: {miou * 100:.3f}")


    return miou, total_loss



def main_worker(args):

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    cudnn.benchmark = False

    print_args(args)

    # === Dataloaders ====
    train_dataloader, val_dataloader = utils.get_dataloaders(args)

    args.num_epochs = args.num_steps // len(train_dataloader)

    # === Model ===
    model = utils.init_model(args)


    print('#####################################')
    print('Number of parameters ', sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('#####################################')

    model = nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)

    # === Training Items ===
    loss_fn = None #SimpleLoss().cuda() #SimpleLoss().cuda()  #

    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)  # wd 1e-7

    #scheduler = None
    #if not args.overfit:
        
    scheduler = utils.get_scheduler(args, optimizer, train_dataloader)
    early_stopping = utils.EarlyStopping()

    # === Misc ===
    evaluator = evalmachine.Evaluator() if args.gpu == 0 else None

    run_name = utils.get_run_name(args)
    print(f"Run name: {run_name}")

    if args.gpu == 0:
        utils.init_logger(args, run_name)

    print(f"Loss, optimizer and schedulers ready.")

    # === Load from checkpoint ===
    to_restore = {"epoch": 0}
    if args.use_checkpoint:
        utils.restart_from_checkpoint(args, 
                                      run_variables=to_restore, 
                                      model=model, 
                                      optimizer=optimizer, 
                                      scheduler=scheduler)
    start_epoch = to_restore["epoch"]
    print(f"Starting from epoch {start_epoch}")


    start_time = time.time()

    dist.barrier()

    # ========================== Val =================================== #
    if args.validate:

        print("Starting Mask2Former4BEV evaluation!")
        if args.gpu == 0:
            val_epoch(args, model.module, loss_fn, val_dataloader, evaluator, log=True)

        dist.barrier()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Validation time {}'.format(total_time_str))
        dist.destroy_process_group()
        return

    # ========================== Train =================================== #

    if args.gpu == 0:
        if not os.path.exists(os.path.join(args.model_save_path, run_name)):
            os.mkdir(os.path.join(args.model_save_path, run_name))

    print("Starting training!")

    if "total_iter" not in to_restore:
        total_iter = 0
    else:
        total_iter = to_restore["total_iter"]
    
    if "best_miou" not in to_restore:
        best_miou = 0
    else:
        best_miou = to_restore["best_miou"]

    print(f'Number of epochs {args.num_epochs}')

    for epoch in range(start_epoch, args.num_epochs):
        train_dataloader.sampler.set_epoch(epoch)

        print(f"===== ===== [Epoch {epoch}] ===== =====")

        mean_loss, total_iter = train_epoch(args, model, loss_fn, optimizer, scheduler, train_dataloader, total_iter)

        # === Save Checkpoint ===
        save_dict = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "epoch": epoch + 1,
            "args": args,
            "best_miou": best_miou,
            "total_iter": total_iter
        }

        if epoch % args.save_epoch == 0:

            utils.save_on_master(save_dict, os.path.join(args.model_save_path, run_name, f"epoch_{epoch}.pt"))


        # === Validate ===
        if args.gpu == 0: # and False:
            if (epoch == 0) or ((epoch + 1) % args.validation_epoch == 0):
                miou, val_loss = val_epoch(args, model.module, loss_fn, val_dataloader, evaluator)

                if miou > best_miou:
                    best_miou = miou
                
                    utils.save_on_master(save_dict, os.path.join(args.model_save_path, run_name, "best.pt"))
                    print('best checkpoint saved ')


                # === Log ===
                wandb.log({
                        'epoch': epoch + 1,
                        'train loss': 0, # mean_loss,
                        'val loss': val_loss,
                        'mIoU': miou
                    })

                # if early_stopping(miou):
                #     break

        dist.barrier()

        print("===== ===== ===== ===== =====")

    print("\n=== Results ===")
    print(f"\tbest mIoU: {best_miou * 100:.3f}")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    dist.destroy_process_group()



if __name__ == '__main__':
    args = get_args()
    main_worker(args)




