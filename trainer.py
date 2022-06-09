# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
from utilities.multiOptimTrainer import train_epoch_BG_MO, save_checkpoint_MO
import torch.utils.data.distributed
from monai.data import decollate_batch
from monai.inferers.utils import sliding_window_inference
from monai.networks.utils import one_hot 
from medpy.metric.binary import recall
from monai.metrics import compute_meandice
def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)

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


def train_epoch_BG(model,
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
    
    optimizer.zero_grad(set_to_none = True)
    for param in model.parameters() : param.grad = None

    num_batch_per_epoch = int(3*98//args.batch_size)
    for idx in range(num_batch_per_epoch):
        # for idx, batch_data in enumerate(loader):
        batch_data = next(loader)
        
        if isinstance(batch_data, list):
            data, target = batch_data.to(device)
        else:
            data, target = batch_data['data'].to(device), batch_data['target'].to(device)

        torch.cuda.empty_cache()
        for param in model.parameters(): param.grad = None
        optimizer.zero_grad(set_to_none = True)
        with autocast(enabled=args.amp):
            logits = model(data)
            loss = loss_func(logits, target)
            
        if args.amp:
            print('---------------AMP true-------------------') # We do not use AMP for training
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.cuda.empty_cache()

            loss.backward()
            optimizer.step()
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

def train_epoch_MONAI(model,
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
    
    optimizer.zero_grad(set_to_none = True)
    for param in model.parameters() : param.grad = None

    for id in range(3):
        for idx, batch_data in enumerate(loader):
            # batch_data = next(loader)
            
            if isinstance(batch_data, list):
                data, target = batch_data.to(device)
            else:
                data, target = batch_data['image'].to(device), batch_data['label'].to(device)
            
            torch.cuda.empty_cache()

            with autocast(enabled=args.amp):
                logits = model(data)
                loss = loss_func(logits, target)
                
            if args.amp:
                print('---------------AMP true-------------------') # We do not use AMP for training
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.cuda.empty_cache()

                loss.backward()
                optimizer.step()
                                
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

def val_epoch_MONAI(model,
              loader,
              epoch,
              loss_func,
              acc_func,
              args,
              model_inferer=None,
              post_label=None,
              post_pred=None):
    model.eval()

    start_time = time.time()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda == False:
        raise ValueError('No GPU detected')
    else: 
        torch.cuda.empty_cache()
    
    run_loss = AverageMeter()
    metric_sum = 0.0
    metric_count = 0
    metric = []
    recall_list = []
    diced = []

    num_batch_per_val_epoch = 60
    with torch.no_grad():
        for id in range(6):   
            for idx, batch_data in enumerate(loader):
                
                if isinstance(batch_data, list):
                    data, target = batch_data.to(device)    
                else:
                    data, target = batch_data['image'].to(device), batch_data['label'].to(device)

                torch.save(data, f"./foo_outputs/ims/{batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1][:-4]}.pt")
                with autocast(enabled=args.amp):

                    model_inferer = None # NOT using sliding window

                    if model_inferer is not None:
                        
                        logits = model_inferer(data)
                        loss = loss_func(logits, target)
                    else:
                        logits = model(data)
                        loss = loss_func(logits, target)
                
                if args.distributed:
                    loss_list = distributed_all_gather([loss],
                                                out_numpy=True,
                                                is_valid=idx < loader.sampler.valid_length)
                    run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                                n=args.batch_size * args.world_size)
                else:
                    run_loss.update(loss.item(), n=args.batch_size)
            
                if not logits.is_cuda:
                    target = target.cpu()
                val_labels_list = decollate_batch(target)
                
                val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
                val_outputs_list = decollate_batch(logits)
                
                val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
                 
                 # Alternative dice computation
                pred = torch.argmax(torch.softmax(logits.squeeze(), 0), 0).detach().cpu().expand(1, 1,  args.roi_x, args.roi_y, args.roi_z)
                dice_value = compute_meandice(pred, target.detach().cpu()).item()
                
                if np.isnan(dice_value) ==True:
                    pass
                else:
                    diced.append(dice_value)
                # print(val_output_convert[0].shape)
                # print(val_labels_convert[0].shape)
                #torch.save(pred, f"./foo_outputs/preds/{batch_data['image_meta_dict']['filename_or_obj'][0].split('/')[-1][:-4]}.pt")

                acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
                acc = acc.cuda(args.rank)
                
                acc_func.reset()

                # im = val_output_convert[0].detach().cpu().numpy()
                # la =  val_labels_convert[0].detach().cpu().numpy()

                
                recall_list.append(recall(pred.detach().cpu().numpy() ,target.detach().cpu().numpy()))
                # recall_list.append(recall(im, la))

                if args.distributed:
                    acc_list = distributed_all_gather([acc],
                                                    out_numpy=True,
                                                    is_valid=idx < loader.sampler.valid_length)
                    avg_acc = np.mean([np.nanmean(l) for l in acc_list])

                else:
                    acc_list = acc.detach().cpu().numpy()
                    avg_acc = np.mean([np.nanmean(l) for l in acc_list])
                    
                if np.isnan(avg_acc) != True: 
                    metric.append(avg_acc)
        # fin_time = time.time()
    # metric =  np.array(metric)
    avg_accuracy =  np.mean(np.array(diced))
    if np.isnan(avg_accuracy) == True:
        return np.mean(metric), run_loss.avg, np.mean(recall_list)
    # print("Mean alterantive dice", np.mean(np.array(diced)))
    return avg_accuracy, run_loss.avg, np.mean(recall_list)

def val_epoch_BG(model,
              loader,
              epoch,
              loss_func,
              acc_func,
              args,
              model_inferer=None,
              post_label=None,
              post_pred=None):
    model.eval()

    start_time = time.time()

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda == False:
        raise ValueError('No GPU detected')
    else: 
        torch.cuda.empty_cache()
    
    run_loss = AverageMeter()
    metric_sum = 0.0
    metric_count = 0
    metric = []
    recall_list = []
    diced = []

    num_batch_per_val_epoch = int(3*20//1) # K*20 WITH INT(K)

    with torch.no_grad():
        for idx in range(num_batch_per_val_epoch):
            batch_data = next(loader)

            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                
                data, target = batch_data['data'].to(device), batch_data['target'].to(device)
    
            for param in model.parameters(): param.grad = None
            #torch.save(data, f"./foo_outputs/ims/PAT00{idx}_im.pt")
            #torch.save(target, f"./foo_outputs/preds/PAT00{idx}_la.pt")
            with autocast(enabled=args.amp):

                model_inferer = None

                if model_inferer is not None:
                    
                    logits = model_inferer(data)
                    
                    loss = loss_func(logits, target)
                    print(loss.item())
                else:
                    logits = model(data)

                    loss = loss_func(logits, target)
                
            #torch.save(logits, f"./foo_outputs/preds/PAT00{idx}_pred.pt")
            
            if args.distributed:
                loss_list = distributed_all_gather([loss],
                                               out_numpy=True,
                                               is_valid=idx < loader.sampler.valid_length)
                run_loss.update(np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                            n=args.batch_size * args.world_size)
            else:
                run_loss.update(loss.item(), n=args.batch_size)
        
            if not logits.is_cuda:
                target = target.cpu()
            val_labels_list = decollate_batch(target)
            
            val_labels_convert = [post_label(val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
           
            val_output_convert = [post_pred(val_pred_tensor) for val_pred_tensor in val_outputs_list]
            
            # Alternative dice computation
            
            pred = torch.argmax(torch.softmax(logits.squeeze(), 0), 0).detach().cpu().expand(1, 1, args.roi_x, args.roi_y, args.roi_z)
            # print(pred.shape, target.shape)
            dice_value = compute_meandice(pred, target.detach().cpu()).item()

            if np.isnan(dice_value) == True:
                idx -= 1
            else:
                diced.append(dice_value)
                       
            # Normal Dice computation
            acc = acc_func(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.cuda(args.rank)

            # im = val_output_convert[0].detach().cpu().numpy()
            # la =  val_labels_convert[0].detach().cpu().numpy()

            recall_list.append(recall(pred.detach().cpu().numpy() ,target.detach().cpu().numpy()))
            
            if args.distributed:
                acc_list = distributed_all_gather([acc],
                                                out_numpy=True,
                                                is_valid=idx < loader.sampler.valid_length)
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])

            else:
                acc_list = acc.detach().cpu().numpy()
                avg_acc = np.mean([np.nanmean(l) for l in acc_list])
                
            if np.isnan(avg_acc) != True: 
                metric.append(avg_acc)
        fin_time = time.time()
  
    metric =  np.array(metric)
    avg_accuracy =  np.mean(np.array(diced))
    if np.isnan(avg_accuracy) == True:
        return np.mean(metric), run_loss.avg, np.mean(recall_list)
    # print("Mean alterantive dice", np.mean(np.array(diced)))
    return avg_accuracy, run_loss.avg, np.mean(recall_list)

def save_checkpoint(model,
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
            'optimizer': optimizer.state_dict(),
            'scheduler': None
            }
    # if optimizer is not None:
    #     save_dict['optimizer'] = optimizer.state_dict()
    if scheduler is not None:
        save_dict['scheduler'] = scheduler.state_dict()
    filename=os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    

def run_training(model,
                 train_loader,
                 val_loader,
                 optimizer,
                 loss_func,
                 acc_func,
                 args,
                 model_inferer=None,
                 scheduler=None,
                 start_epoch=0,
                 val_acc_max = 0.,
                 val_rec_max =.0, 
                 post_label=None,
                 post_pred=None
                 ):
    writer = None
    # if args.logdir is not None and args.rank == 0:
    #     writer = SummaryWriter(log_dir=args.logdir)
    #     if args.rank == 0: print('Writing Tensorboard logs to ', args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = val_acc_max
    val_rec_max = val_rec_max
    TRAIN_LOSS = []
    VAL_LOSS = []
    VAL_ACC = []
    VAL_REC = []
    listLR = []
    
    
    
        
    if os.path.exists(os.path.join(args.logdir, 'learningCurves', 'Val_loss.npy')):
        VAL_REC = np.load(os.path.join(args.logdir, 'learningCurves', 'Val_rec.npy'))
        TRAIN_LOSS = np.load(os.path.join(args.logdir, 'learningCurves', 'Train_loss.npy'))
        VAL_LOSS = np.load(os.path.join(args.logdir, 'learningCurves', 'Val_loss.npy'))
        VAL_ACC = np.load(os.path.join(args.logdir, 'learningCurves', 'Val_acc.npy'))
        if not args.lrschedule == "ReduceOnPlateau": 
            listLR = np.load(os.path.join(args.logdir, 'learningCurves', 'listLR.npy'))
        for idx, case in enumerate(TRAIN_LOSS):
            print(idx, case)

    if args.mode == "FittingLR":
        args.max_epochs = 50    

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), 'Epoch:', epoch)
        epoch_time = time.time()

        if args.multiOptim:
            train_loss = train_epoch_BG_MO(model,
                                    train_loader,
                                    optimizer,
                                    scaler=scaler,
                                    epoch=epoch,
                                    loss_func=loss_func,
                                    args=args)
        else:
            if args.DA_library == "BG":
                train_loss = train_epoch_BG(model,
                                        train_loader,
                                        optimizer,
                                        scaler=scaler,
                                        epoch=epoch,
                                        loss_func=loss_func,
                                        args=args)
            else:
                train_loss = train_epoch_MONAI(model,
                                        train_loader,
                                        optimizer,
                                        scaler=scaler,
                                        epoch=epoch,
                                        loss_func=loss_func,
                                        args=args)
        TRAIN_LOSS = np.append(TRAIN_LOSS, train_loss)

        # Add LR to saving list
        # if not args.lrschedule == "ReduceOnPlateau": 
        #     listLR.append(scheduler.get_last_lr()[0])
                
        if args.rank == 0:
            print('Final training  {}/{}'.format(epoch, args.max_epochs - 1), 'loss: {:.4f}'.format(train_loss),
                  'time {:.2f}s'.format(time.time() - epoch_time))
        # if args.rank==0 and writer is not None:
        #     writer.add_scalar('train_loss', train_loss, epoch)
        b_new_best = False
        if (epoch+1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            if args.DA_library == "BG":
                val_avg_acc, val_loss, val_rec = val_epoch_BG(model,
                                        val_loader,
                                        epoch=epoch,
                                        loss_func= loss_func,
                                        acc_func=acc_func,
                                        model_inferer=model_inferer,
                                        args=args,
                                        post_label=post_label,
                                        post_pred=post_pred)
            else:
                val_avg_acc, val_loss, val_rec = val_epoch_MONAI(model,
                                        val_loader,
                                        epoch=epoch,
                                        loss_func= loss_func,
                                        acc_func=acc_func,
                                        model_inferer=model_inferer,
                                        args=args,
                                        post_label=post_label,
                                        post_pred=post_pred)

            VAL_LOSS = np.append(VAL_LOSS, val_loss)
                        
            VAL_ACC = np.append(VAL_ACC, val_avg_acc)

            VAL_REC = np.append(VAL_REC, val_rec)
            
           
            print('Final validation  {}/{}'.format(epoch, args.max_epochs - 1),
                    'loss', np.round(val_loss, 4), 'acc', val_avg_acc, 'recall', val_rec, 'time {:.2f}s'.format(time.time() - epoch_time))
            
            torch.cuda.empty_cache()
            
            # if writer is not None:
            #     writer.add_scalar('val_acc', val_avg_acc, epoch)
            # Save best model    
            if val_avg_acc > val_acc_max:
                print('New best ({:.6f} --> {:.6f}). '.format(val_acc_max, val_avg_acc))
                val_acc_max = val_avg_acc
                b_new_best = True
                if not args.save_checkpoint is None:
                    
                    if args.multiOptim:
                        save_checkpoint_MO(model, 
                                    current_epoch = epoch, 
                                    args = args,
                                    best_acc=val_acc_max,
                                    best_rec = val_rec_max,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    filename='best_model.pth')
                    else:
                        save_checkpoint(model, 
                                        current_epoch = epoch, 
                                        args = args,
                                        best_acc=val_acc_max,
                                        best_rec = val_rec_max,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        filename='best_model.pth')

                    print("Saving best model")
            if val_rec > val_rec_max:
                print('New best recall ({:.6f} --> {:.6f}). '.format(val_rec_max, val_rec))
                val_rec_max = val_rec
                b_new_best = True
                if not args.save_checkpoint is None:
                    
                    if args.multiOptim:
                        save_checkpoint_MO(model, 
                                    current_epoch = epoch, 
                                    args = args,
                                    best_acc=val_acc_max,
                                    best_rec = val_rec_max,
                                    optimizer=optimizer,
                                    scheduler=scheduler,
                                    filename='best_recall_model.pth')
                    else:               
                        save_checkpoint(model, 
                                        current_epoch = epoch, 
                                        args = args,
                                        best_acc=val_acc_max, 
                                        best_rec = val_rec_max,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        filename='best_recall_model.pth')

                    print("Saving best recall model")
            # Save latest model
            if not args.save_checkpoint is None:
                if scheduler is not None: 
                    if args.multiOptim:
                        scheduler[0].step()
                        scheduler[1].step()
                    else:
                        scheduler.step()
                if args.multiOptim:
                    save_checkpoint_MO(model, 
                                        current_epoch = epoch, 
                                        args = args,
                                        best_acc=val_acc_max,
                                        best_rec = val_rec_max,
                                        optimizer=optimizer,
                                        scheduler=scheduler,
                                        filename='latest_model.pth')
                else:
                    save_checkpoint(model,  
                                    current_epoch = epoch, 
                                    args = args,
                                    best_acc=val_avg_acc,
                                    best_rec = val_rec_max,
                                    optimizer=optimizer,
                                    scheduler=scheduler, # Save 
                                    filename='latest_model.pth')
                print("Saving checkpoint")
        # Plotting progress
        # plot_progress(normlossTR, normlossVL, VAL_ACC, epoch, args)
        Dir = os.path.join(args.logdir, 'learningCurves')
        if not os.path.isdir(Dir): os.mkdir(Dir)
        np.save(os.path.join(Dir,'Train_loss.npy'), np.array(TRAIN_LOSS))
        np.save(os.path.join(Dir,'Val_loss.npy'), np.array(VAL_LOSS))
        np.save(os.path.join(Dir,'Val_acc.npy'), np.array(VAL_ACC))
        np.save(os.path.join(Dir,'Val_rec.npy'), np.array(VAL_REC))
        if not args.lrschedule == "ReduceOnPlateau": 
            np.save(os.path.join(Dir,'listLR.npy'), np.array(listLR))
        
        plot_progress(TRAIN_LOSS, VAL_LOSS, VAL_ACC, epoch, args)
        print("Saving training progress plot")

        if args.mode == 'FittingLR':
            
            plot_fitting_LR(TRAIN_LOSS, VAL_ACC, listLR, epoch, args)

            # if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
            #     save_checkpoint(model,
            #                     epoch,
            #                     args,
            #                     best_acc=val_acc_max,
            #                     filename='model_final_checkpoint.pt')
                # if b_new_best:
                #     print('Copying to model.pt new best model!!!!')
                #     shutil.copyfile(os.path.join(args.logdir, 'model_final.pt'), os.path.join(args.logdir, 'model.pt'))

        
        print()
        print()
        print()

    print('Training Finished !, Best Accuracy: ', val_acc_max)

    return val_acc_max
