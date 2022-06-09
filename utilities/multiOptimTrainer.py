from audioop import avg
from inspect import CO_ITERABLE_COROUTINE
import os
import time
import shutil
import numpy as np

import torch
from torch.cuda.amp import GradScaler, autocast
# from tensorboardX import SummaryWriter
import torch.nn.parallel
from monai.losses import DiceLoss, DiceCELoss 
from utilities.utils import distributed_all_gather, plot_progress, loss_normalization, weight_CE, plot_fitting_LR
import torch.utils.data.distributed
from monai.data import decollate_batch
from monai.inferers.utils import sliding_window_inference
from monai.networks.utils import one_hot 
from medpy.metric.binary import recall
from monai.metrics import compute_meandice

class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0,
                            self.sum / self.count,
                            self.sum)
                            
def train_epoch_BG_MO(model,
                loader,
                optimizer,
                scaler,
                epoch,
                loss_func,
                args):

    model.train()

    # Check gpu
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda == False:
        raise ValueError('No GPU detected')

    # Training loop
    start_time = time.time()

    run_loss = AverageMeter()
    
    optimizer1 = optimizer[0]
    optimizer2 = optimizer[1]
    optimizer1.zero_grad(set_to_none = True)
    optimizer2.zero_grad(set_to_none = True)

    for param in model.parameters() : param.grad = None

    num_batch_per_epoch = int(3*98//args.batch_size)
    for idx in range(num_batch_per_epoch):
        # for idx, batch_data in enumerate(loader):
        batch_data = next(loader)
        
        if isinstance(batch_data, list):
            data, target = batch_data.to(device)
        else:
            data, target = batch_data['data'].to(device), batch_data['target'].to(device)
        torch.save(data, f"./foo_outputs/ims/PAT00{idx}_im.pt")
        torch.save(target, f"./foo_outputs/preds/PAT00{idx}_la.pt")
        torch.cuda.empty_cache()
        for param in model.parameters(): param.grad = None
        # optimizer.zero_grad(set_to_none = True)
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
            torch.save(logits, f"./foo_outputs/preds/PAT00{idx}_pred.pt")
            
        if args.amp:
            print('---------------AMP true-------------------') # We do not use AMP for training
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.cuda.empty_cache()

            loss.backward()
            optimizer2.step()
            optimizer1.step()
            torch.cuda.empty_cache()

        if args.distributed:
            loss_list = distributed_all_gather([loss],
                                            out_numpy=True,
                                            is_valid=idx < loader.sampler.valid_length)
            run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
        else:

            run_loss.update(loss.item(), n=args.batch_size)
        
        start_time = time.time()

    torch.cuda.empty_cache()

    for param in model.parameters() : param.grad = None
    return run_loss.avg


def save_checkpoint_MO(model,
                    current_epoch,
                    args,
                    filename='model.pth',
                    best_acc=0,
                    best_rec = 0.,
                    optimizer=None,
                    scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {
            'epoch': current_epoch,
            'best_acc': best_acc,
            'best_recall': best_rec,
            'state_dict': state_dict,
            'scheduler': None
            }
    if args.multiOptim:
        save_dict['optimizer1'] = optimizer[0].state_dict()
        save_dict['optimizer2'] = optimizer[1].state_dict()
         
        save_dict['scheduler1'] = scheduler[0].state_dict()
        save_dict['scheduler2'] = scheduler[1].state_dict()
    
    filename=os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    

