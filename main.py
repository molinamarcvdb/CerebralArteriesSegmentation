from logging import raiseExceptions
import os
from pickle import FALSE
import numpy as np
import torch
import json as js
from inferers.inference import inference
from evaluation import eval_metrics
import torch.nn.parallel
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.utils.data.distributed
from monai.inferers import sliding_window_inference, SlidingWindowInferer
from monai.losses import TverskyLoss
from networks.dice import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.utils.enums import MetricReduction
from monai.transforms import AsDiscrete,Activations,Compose
from monai.data import load_decathlon_datalist, pad_list_data_collate, DataLoader, Dataset
from monai.networks.nets import DynUNet
from networks.swinUNETR import SwinUNETR
from training.generate_splits import do_split, generate_splits
from utilities.utils import generate_MONAI_json, generate_json_BG, resample_nifti, craniumExtraction
from training.data_augmentation import default_3D_augmentation_params, get_default_augmentation, get_patch_size, get_default_test_augmentation
from training.Data_loading import  load_dataset, DataLoader3D, unpack_dataset
from networks.unetr import UNETR
#import self_attention_cv as SACV
# from self_attention_cv import UNETR
from utilities.data_utils import get_loader, Sampler
from utilities.utils import check_model, initialize_model, save_training_schemes
from trainer import run_training
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR, ExponentialLR
from functools import partial
import argparse
from monai.transforms import (EnsureType, Compose, Activations, LoadImaged, AddChanneld, Transpose,Activations,AsDiscrete, RandGaussianSmoothd, CropForegroundd, SpatialPadd,
                            ScaleIntensityRangePercentilesd, ScaleIntensityd, ToTensord, HistogramNormalized, AdjustContrastd, RandSpatialCropd, Rand3DElasticd, RandAffined, RandZoomd,
                            Spacingd, Orientationd, Resized, ThresholdIntensityd, RandShiftIntensityd, BorderPadd, RandGaussianNoised, RandAdjustContrastd,NormalizeIntensityd,RandFlipd)


parser = argparse.ArgumentParser(description='UNETR segmentation pipeline')
parser.add_argument('--mode', '-m', required = True, type = str, help = 'Set the mode with which the framework is going to be plugged in: Btw Training, Testing, FittingLR')
parser.add_argument('--preprocessing', default=None, help='start training with preprocessed images')
parser.add_argument('--monai', action='store_true', help='start model from monai')
parser.add_argument('--preproc_folder', default = './unetr_base/preprocessed/Task300_Cerebral/', help = 'Ste preproceeing ffolder')
parser.add_argument('--fold', default=None, required = True, type = int, help='Fold to train in this iteration ')
parser.add_argument('--lowres', default=None, type = bool, help='Use low resolution images')
parser.add_argument('--transforms', default=None, type = bool, help='Wheather to use image trasnforms or not')
parser.add_argument('--craniumExtraction', default=None, type = bool, help='Whether to usecraniumExtracted images')
parser.add_argument('--logdir', default='test', type=str, help='directory to save the tensorboard logs')
parser.add_argument('--pretrained_dir', default='./pretrained_models/', type=str, help='pretrained checkpoint directory')
parser.add_argument('--data_dir', default='./unetr_base/raw_data/Cerebral300/', type=str, help='dataset JSON directory')
parser.add_argument('--raw_data_dir', default='/dataset/dataset0/', type=str, help='raw dataset directory')
parser.add_argument('--json_list', default='dataset_0.json', type=str, help='dataset json file')
parser.add_argument('--pretrained_model_name', default='UNETR_model_best_acc.pth', type=str, help='pretrained model name')
parser.add_argument('--save_checkpoint', action='store_true', help='save checkpoint during training')
parser.add_argument('--max_epochs', default=200, type=int, help='max number of training epochs')
parser.add_argument('--batch_size', default=1, type=int, help='number of batch size')
parser.add_argument('--sw_batch_size', default=2, type=int, help='number of sliding window batch size')
parser.add_argument('--optim_lr', default=1e-4, type=float, help='optimization learning rate')
parser.add_argument('--optim_name', default='adamw', type=str, help='optimization algorithm')
parser.add_argument('--loss_func', default='DiceCE', type=str, help='Set LossFunction algorithm')
parser.add_argument('--reg_weight', default=1e-4, type=float, help='regularization weight')
parser.add_argument('--momentum', default=0.99, type=float, help='momentum')
parser.add_argument('--noamp', action='store_true', help='do NOT use amp for training')
parser.add_argument('--val_every', default=1, type=int, help='validation frequency')
parser.add_argument('--distributed', action='store_true', help='start distributed training')
parser.add_argument('--world_size', default=1, type=int, help='number of nodes for distributed training')
parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', type=str, help='distributed url')
parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')
parser.add_argument('--workers', default=4, type=int, help='number of workers') #8
parser.add_argument('--num_threads_in_mt', default=12, type=int, help=' Number of subprocesses to be used in MultiThreadedAugmentor') #8
parser.add_argument('--model_name', default='unetr', type=str, help='model name')
parser.add_argument('--DA_library', default='BG', type=str, help='Whereas batchgenerators or monai is used for D.Aug and D.Loading')
parser.add_argument('--pos_embed', default='perceptron', type=str, help='type of position embedding')
parser.add_argument('--norm_name', default='instance', type=str, help='normalization layer type in decoder')
parser.add_argument('--num_heads', default=12, type=int, help='number of attention heads in ViT encoder')
parser.add_argument('--mlp_dim', default=3072, type=int, help='mlp dimention in ViT encoder')
parser.add_argument('--hidden_size', default=768, type=int, help='hidden size dimention in ViT encoder')
parser.add_argument('--feature_size', default=16, type=int, help='feature size dimention')
parser.add_argument('--in_channels', default=1, type=int, help='number of input channels')
parser.add_argument('--out_channels', default=2, type=int, help='number of output channels')
parser.add_argument('--res_block', action='store_true', help='use residual blocks')
parser.add_argument('--conv_block', action='store_true', help='use conv blocks')
parser.add_argument('--use_normal_dataset', action='store_true', help='use monai Dataset class')
parser.add_argument('--a_min', default=-175.0, type=float, help='a_min in ScaleIntensityRanged')
parser.add_argument('--a_max', default=250.0, type=float, help='a_max in ScaleIntensityRanged')
parser.add_argument('--b_min', default=0.0, type=float, help='b_min in ScaleIntensityRanged')
parser.add_argument('--b_max', default=1.0, type=float, help='b_max in ScaleIntensityRanged')
parser.add_argument('--space_x', default=0.42, type=float, help='spacing in x direction')
parser.add_argument('--space_y', default=0.42, type=float, help='spacing in y direction')
parser.add_argument('--space_z', default=0.40, type=float, help='spacing in z direction')
parser.add_argument('--roi_x', default=112, type=int, help='roi size in x direction')
parser.add_argument('--roi_y', default=112, type=int, help='roi size in y direction')
parser.add_argument('--roi_z', default=112, type=int, help='roi size in z direction')
parser.add_argument('--dropout_rate', default=0.0, type=float, help='dropout rate')
parser.add_argument('--RandFlipd_prob', default=0.1, type=float, help='RandFlipd aug probability')
parser.add_argument('--RandRotate90d_prob', default=0.15, type=float, help='RandRotate90d aug probability')
parser.add_argument('--RandScaleIntensityd_prob', default=0.2, type=float, help='RandScaleIntensityd aug probability')
parser.add_argument('--RandShiftIntensityd_prob', default=0.2, type=float, help='RandShiftIntensityd aug probability')
parser.add_argument('--infer_overlap', default=0.5, type=float, help='sliding window inference overlap')
parser.add_argument('--lrschedule', default='warmup_cosine', type=str, help='type of learning rate scheduler')
parser.add_argument('--warmup_epochs', default=35, type=int, help='number of warmup epochs')
parser.add_argument('--resume_ckpt', action='store_true', help='resume training from pretrained checkpoint')
parser.add_argument('--resume_jit', action='store_true', help='resume training from pretrained torchscript checkpoint')
parser.add_argument('--smooth_dr', default=1e-6, type=float, help='constant added to dice denominator to avoid nan')
parser.add_argument('--smooth_nr', default=0.0, type=float, help='constant added to dice numerator to avoid zero')
parser.add_argument('--multiOptim', action='store_true', help='Use AdamW for ViT and SGD for Upsampling path')


def main():
    args = parser.parse_args()
    args.amp = not args.noamp
    if args.lowres == True:
        args.logdir = './runs/' + args.logdir
        if not os.path.isdir(args.logdir): os.mkdir(args.logdir)
        args.logdir = args.logdir + '/Task300_Cerebral_LowRes/'
        if not os.path.isdir(args.logdir): os.mkdir(args.logdir)
        args.logdir = args.logdir + f'fold_{args.fold}/'
        if not os.path.isdir(args.logdir): os.mkdir(args.logdir)
    else:
        args.logdir = './runs/' + args.logdir + '/Task300_Cerebral_FullRes/'
        if not os.path.isdir(args.logdir): os.mkdir(args.logdir)
        args.logdir = args.logdir + f'fold_{args.fold}/'
        if not os.path.isdir(args.logdir): os.mkdir(args.logdir)

    print(args.logdir)
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print('Found total gpus', args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker,
                 nprocs=args.ngpus_per_node,
                 args=(args,))
    else:
        if torch.cuda.is_available() == True:
            main_worker(gpu='cuda', args=args)
        else:
            main_worker(gpu='cpu', args=args)

def main_worker(gpu, args):

    if args.mode == 'Training':

        #Save training inputs
        save_training_schemes(args)

        if args.distributed:
            torch.multiprocessing.set_start_method('fork', force=True)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
        
        # Set Device
        args.gpu = gpu

        # Distributed
        if args.distributed:
            args.rank = args.rank * args.ngpus_per_node + gpu
            dist.init_process_group(backend=args.dist_backend,
                                    init_method=args.dist_url,
                                    world_size=args.world_size,
                                    rank=args.rank)
        # Check and set gpu
        use_cuda = torch.cuda.is_available()
        print("GPU Availavility: ", use_cuda)
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.device(args.gpu)
        torch.backends.cudnn.benchmark = True
        args.test_mode = False
        
        # Inference size
        args.patch_size = inf_size = [args.roi_x, args.roi_y, args.roi_x]

        # Cranium extraction ?
        if args.craniumExtraction:
            
            caseDir = os.path.join(args.preproc_folder, "images_cranium")
            if not os.path.exists(caseDir): os.mkdir(caseDir)

            if len(os.listdir(caseDir)) < 147 and args.DA_library == "MONAI":
                
                print('Extracting cranium from images..')
                
                craniumExtraction(args)
                
                print('Done!')
            
            caseDir = os.path.join(args.preproc_folder, "Data_plans_v2.1_stage0_Cranium")

            if args.DA_library == "BG" and not os.path.exists(caseDir):
            
                print('Extracting cranium from images..')
            
                craniumExtraction(args)
            
                print('Done!')
        
        # Get images train[0]/val[1] loader
        loader = get_loader(args)      
        
        
        pretrained_dir = args.pretrained_dir
        
        if (args.model_name is None) or args.model_name == 'unetr':

            print('Using MONAI UNETr')

            model = UNETR(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y, args.roi_z),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate)

            model = model.to(device)

        elif args.model_name == "SwinUNETR":

            print("Using Monai Swin UNTER")

            model = SwinUNETR(
                img_size = (args.roi_x, args.roi_y, args.roi_z),
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                feature_size = 48,
                depths = [2, 2, 2, 2],
                num_heads = [3, 6, 12, 24],
                norm_name = "instance",
                dropout_path_rate = args.dropout_rate
            )
            
            model = model.to(device)
        
        elif args.model_name == "nnUNet":
            
            model = DynUNet(
                        spatial_dims = len(args.patch_size),
                        in_channels = args.in_channels,
                        out_channels = args.out_channels,
                        kernel_size= [3,3,3],
                        strides = [2,2,2], 
                        upsample_kernel_size= [2,2,2],
                        filters = [64, 96, 128, 192, 256, 384, 512, 768, 1024][: len(2)],
                        dropout=args.dropout,
                        norm_name = ("INSTANCE", {"affine": True}),
                        act_name = ("leakyrelu", {"inplace": True, "negative_slope": 0.01}),
                        deep_supervision= True,
                        res_block = True)
            model.to(device) 
        
       
        else:
            raise ValueError('Unsupported model ' + str(args.model_name))
        
         # Load Pretrained models
            #   Weights
        if args.resume_ckpt:
            fileDir = os.path.join(pretrained_dir, args.pretrained_model_name)
            model_dict = torch.load(fileDir, map_location=torch.device(device))
            model_dict = check_model(model, model_dict, fileDir)
            model.load_state_dict(model_dict)
            print('Using pretrained weights')
        else: 
            #Initialize model
            print('Not using pretraining - Initializing weights (Xavier & Normal Distributions)...')
            model_dict = initialize_model(model)
            model.load_state_dict(model_dict)
        
        #   TorchScript
        if args.resume_jit:
            if not args.noamp:
                print('Training from pre-trained checkpoint does not support AMP\nAMP is disabled.')
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    
    # Set LOSS FUNCTION
        if args.loss_func == 'DiceCe':

            print('Using DiceCE loss')

            loss_func = DiceCELoss(to_onehot_y=True,
                                softmax=True,
                                sigmoid = False,
                                squared_pred=True,
                                reduction = "mean",
                                smooth_nr=0, 
                                smooth_dr=1e-6,
                                lambda_dice = 0.7,
                                lambda_ce = 0.30,
                                ce_weight = torch.cuda.FloatTensor([0.1,2])
                                )
        elif args.loss_func == "Tversky":

            print('Using Tversky loss')
            
            loss_func = TverskyLoss(include_background= True,
                                    to_onehot_y = True,
                                    sigmoid = False,
                                    softmax = True,
                                    alpha = 0.3, 
                                    beta = 0.7,
                                    reduction = "mean",
                                    smooth_dr = 1e-6,
                                    smooth_nr = 0
                                    )
        
        elif args.loss_func == 'DiceFocal':

            print('Using DiceFocal loss')
            
            loss_func = DiceFocalLoss(include_background=True,
                                    to_onehot_y = True,
                                    sigmoid = False,
                                    softmax = True,
                                    squared_pred= True,
                                    jaccard = False,
                                    reduction = 'mean',
                                    smooth_dr = 0, 
                                    smooth_nr = 1e-6,
                                    gamma = 0.90,
                                    focal_weight = [0.1, 50],
                                    lambda_dice= 1,
                                    lambda_focal= 0.5
                                    )
                                
        else:

            print('Using Dice loss')

            loss_func = DiceLoss(include_background=True, 
                                to_onehot_y=False, 
                                sigmoid=True, 
                                softmax=False, 
                                squared_pred=False, 
                                jaccard=False, 
                                reduction='mean', 
                                smooth_nr=1e-05, 
                                smooth_dr=1e-05)   

        post_label = AsDiscrete(to_onehot = args.out_channels)
        post_pred = Compose(
            [Activations (softmax = True), AsDiscrete(argmax=True, to_onehot = args.out_channels)]
            )
        
        dice_acc = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

        model_inferer = partial(sliding_window_inference,
                                roi_size=inf_size,
                                sw_batch_size=args.sw_batch_size,
                                predictor=model,
                                overlap=args.infer_overlap,
                                sw_device = "cuda",
                                device = 'cuda',
                                )
        

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total parameters count', pytorch_total_params)

                   

        if args.distributed:
            
            if args.norm_name == 'batch':
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.gpu],
                                                            output_device=args.gpu,
                                                            find_unused_parameters=True).to(device)
        # OPTIMIZER
        if args.multiOptim:
            print('Using both adaptive and stochastic optimizers')
            params = list(model.encoder1.parameters()) + list(model.encoder2.parameters()) + list(model.encoder3.parameters()) + list(model.encoder4.parameters()) + list(model.decoder2.parameters()) + list(model.decoder3.parameters()) + list(model.decoder4.parameters()) + list(model.decoder5.parameters()) + list(model.out.parameters())
            optimizer1 = torch.optim.AdamW(model.vit.parameters(),
                                            lr=args.optim_lr,
                                            betas = (0.95, 0.99),
                                            eps = 1e-7,
                                            weight_decay=args.reg_weight,
                                            amsgrad=True)
            #optimizer1 = torch.optim.RAdam(model.parameters(),
                                            # lr = args.optim_lr,
                                            # betas=(0.95, 0.99),
                                            # eps = 1e-7,
                                            # weight_decay=args.reg_weight)
                
            optimizer2 = torch.optim.SGD(model.parameters(),
                                        lr=5e-4,
                                        momentum=args.momentum,
                                        nesterov=True,
                                        weight_decay=args.reg_weight)
            
            scheduler1 = torch.optim.lr_scheduler.CyclicLR(optimizer1, 
                                            base_lr = 8e-6,
                                            max_lr = 8e-5, 
                                            step_size_up=25, 
                                            step_size_down=25, 
                                            mode='triangular2', 
                                            gamma=1.0, 
                                            scale_fn=None, 
                                            scale_mode='cycle', 
                                            cycle_momentum=False, 
                                            base_momentum=0.8, 
                                            max_momentum=0.99, 
                                            last_epoch=- 1, 
                                            verbose=True)
            # scheduler1 = torch.optim.lr_scheduler.ExponentialLR(optimizer1, 
            #                                                     gamma=0.99, 
            #                                                     verbose=True)
            scheduler2 = torch.optim.lr_scheduler.ExponentialLR(optimizer2, 
                                                                gamma=0.99, 
                                                                verbose=True)
            
        else:
            if args.optim_name == 'adam':
                optimizer = torch.optim.Adam(model.parameters(),
                                            lr=args.optim_lr,
                                            betas = (0.95, 0.99),
                                            eps = 1e-7,
                                            weight_decay=args.reg_weight,
                                            amsgrad=True)
            elif args.optim_name == 'adamw':
                optimizer = torch.optim.AdamW(model.parameters(),
                                            lr=args.optim_lr,
                                            betas = (0.95, 0.99),
                                            eps = 1e-7,
                                            weight_decay=args.reg_weight,
                                            amsgrad=True)
            elif args.optim_name == 'sgd':
                optimizer = torch.optim.SGD(model.parameters(),
                                            lr=args.optim_lr,
                                            momentum=args.momentum,
                                            nesterov=True,
                                            weight_decay=args.reg_weight)
            elif args.optim_name == 'RAdam':
                optimizer = torch.optim.RAdam(model.parameters(),
                                            lr = args.optim_lr,
                                            betas=(0.95, 0.99),
                                            eps = 1e-7,
                                            weight_decay=args.reg_weight)
            elif args.optim_name == 'NAdam':
                optimizer = torch.optim.NAdam(model.parameters(),
                                            lr = args.optim_lr,
                                            betas=(0.90, 0.99),
                                            eps = 1e-7,
                                            weight_decay=args.reg_weight)
            elif args.optim_name == 'Adamax':
                optimizer = torch.optim.Adamax(model.parameters(),
                                            lr = args.optim_lr,
                                            betas=(0.90, 0.99),
                                            eps = 1e-7,
                                            weight_decay=args.reg_weight)

            elif args.optim_name == "rprop": # NOT USE
                optimizer = torch.optim.Rprop(model.parameters(),
                                            lr = args.optim_lr,
                                            etas=(0.4, 1.5),
                                            step_sizes=(1e-6, 10)
                                            )
            elif args.optim_name == "adagrad": # NOT USE
                optimizer = torch.optim.Adagrad(model.parameters(),
                                                lr = args.optim_lr,
                                                lr_decay= 0.99,
                                                weight_decay=args.reg_weight,
                                                eps = 1e-7)
            else:
                raise ValueError('Unsupported Optimization Procedure: ' + str(args.optim_name), "Available Optimizers are: adam, adamw, sgd, RAdam, rprop and adagrad")

            # LR SCHEDULER

            if args.lrschedule == 'warmup_cosine':
                print('LR scheduler: Warmup CosineAnnealing')
                scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                                                        warmup_epochs=args.warmup_epochs,
                                                        max_epochs=args.max_epochs,
                                                        warmup_start_lr= 0.9*args.optim_lr)
            elif args.lrschedule == 'cosine_anneal':
                print('LR scheduler: CosineAnnealing')
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                                    T_max=args.max_epochs,
                                                                    eta_min=2e-7)
                # if args.checkpoint is not None:
                #     scheduler.step(epoch=start_epoch)
            elif args.lrschedule == 'expLR':
                print('LR scheduler: Exponential LR')
                scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 
                                                                    gamma=0.99, 
                                                                    verbose=True)
            elif args.lrschedule == "ReduceOnPlateau": 
                print('LR scheduler: ReduceOnPlateau ')
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min',
                                                        factor=0.5, 
                                                        patience=6, 
                                                        threshold=0.001, 
                                                        threshold_mode='rel', 
                                                        cooldown=4, 
                                                        min_lr=1e-7, 
                                                        eps=1e-06, 
                                                        verbose=True)
            elif args.lrschedule == 'CycleLR':
                print('LR scheduler: CycleLR')
                scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                                base_lr = 5e-5,
                                                max_lr = 1e-4, 
                                                step_size_up=25, 
                                                step_size_down=25, 
                                                mode='triangular2', 
                                                gamma=1.0, 
                                                scale_fn=None, 
                                                scale_mode='cycle', 
                                                cycle_momentum=False, 
                                                base_momentum=0.8, 
                                                max_momentum=0.99, 
                                                last_epoch=- 1, 
                                                verbose=True)
            else:
                print("NOT USING SCHEDULER")
                scheduler = None

        best_acc = 0
        start_epoch = 0
        val_acc_max = 0.
        val_rec_max = 0.

        # RESUME TRAINING
        latestmodelPath = os.path.join(args.logdir, "latest_model.pth")

        if os.path.exists(latestmodelPath):

            checkpoint = torch.load(latestmodelPath, map_location='cpu')
            checkpoint0 = torch.load(f'{latestmodelPath[:-16]}best_model.pth', map_location='cpu')
            checkpoint1 = torch.load(f'{latestmodelPath[:-16]}best_recall_model.pth', map_location='cpu')

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k.replace('backbone.','')] = v
            model.load_state_dict(new_state_dict, strict=False)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                start_epoch +=1
            if 'best_acc' in checkpoint:
                best_acc = checkpoint0['best_acc']
                val_acc_max = best_acc
            if 'best_recall' in checkpoint1:
                val_rec_max = checkpoint1['best_recall']
            if args.multiOptim:
                optimizer1.load_state_dict(checkpoint['optimizer1'])
                optimizer2.load_state_dict(checkpoint['optimizer2'])
                optimizer = [optimizer1,  optimizer2]
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                    if args.multiOptim:
                        scheduler1.load_state_dict(checkpoint['scheduler1'])
                        scheduler2.load_state_dict(checkpoint['scheduler2'])
                        scheduler = [scheduler1, scheduler2]
                    else:
                        scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(latestmodelPath, start_epoch, best_acc))    
            model = model.to(device)

        # RUN TRAINING
        if not args.multiOptim:
            accuracy = run_training(model=model,
                                    train_loader=loader[0],
                                    val_loader=loader[1],
                                    optimizer=optimizer,
                                    loss_func=loss_func,
                                    acc_func=dice_acc,
                                    args=args,
                                    model_inferer=model_inferer,
                                    scheduler=scheduler,
                                    start_epoch=start_epoch,
                                    val_acc_max = val_acc_max,
                                    val_rec_max = val_rec_max,
                                    post_label=post_label,
                                    post_pred=post_pred)
        else:
            accuracy = run_training(model=model,
                                    train_loader=loader[0],
                                    val_loader=loader[1],
                                    optimizer=[optimizer1, optimizer2],
                                    loss_func=loss_func,
                                    acc_func=dice_acc,
                                    args=args,
                                    model_inferer=model_inferer,
                                    scheduler=[scheduler1, scheduler2],
                                    start_epoch=start_epoch,
                                    val_acc_max = val_acc_max,
                                    val_rec_max = val_rec_max,
                                    post_label=post_label,
                                    post_pred=post_pred)
        return accuracy



    elif args.mode == 'Testing':

        # Check and set gpu
        use_cuda = torch.cuda.is_available()
        print("GPU Availavility: ", use_cuda)
        device = torch.device("cuda" if use_cuda else "cpu")

        if args.DA_library == 'MONAI' and args.transforms:
            val_transforms = Compose([
                                    LoadImaged(keys=["image"]),
                                    AddChanneld(keys=["image"]),
                                    AdjustContrastd(keys = ["image"],
                                        gamma = 0.85,),             
                                    ToTensord(keys=["image"]),])
        else:
            print('trueee')
            val_transforms = Compose([
                                    LoadImaged(keys=["image"]),
                                    AddChanneld(keys=["image"]),
                                   
                                    ToTensord(keys=["image"]),])
         # Inference size
        args.patch_size = inf_size = [args.roi_x, args.roi_y, args.roi_x]

        # Set json file 
        if args.DA_library == 'BG':
            
            if args.craniumExtraction:
                datalist_json = f'{args.preproc_folder}data_dicts/dataset_lowres_noCran.json'
            else: 
                datalist_json = os.path.join(args.logdir, f'dataset_lowres.json')

            filename = f'{args.logdir}/dataset_lowres_fold_{args.fold}.json'
            if not os.path.exists(filename):
                splitPath = os.path.join(args.logdir, 'splits_final.pkl')
                if not os.path.exists(splitPath):
                    ValueError(f'{splitPath} not present in {args.logdir}')
                filename = generate_MONAI_json(splitPath, datalist_json, args)
        else:
            data_dir = args.preproc_folder
            if args.lowres is not None:
                filename = f'{args.logdir}/dataset_lowres_fold_{args.fold}.json'
            else: 
                filename = f'{args.logdir}/dataset_fullres_fold_{args.fold}.json'

        
        datalist_json = filename
        # Get test images (.nii.gz format) in a list
        images = []
        with open(datalist_json) as json_file:
            data = js.load(json_file)
        if args.preprocessing is not None:
            for case in range(len(data['test'])):
                images.append(data['test'][case][-12:-4])
        else:
            for case in range(len(data['test'])):
                images.append(data['test'][case][-15:-7]) #############Check
        del data   
        
        if args.preprocessing is not None: data_dir = args.preproc_folder 
        else: data_dir = args.data_dir

        # if args.DA_library == 'MONAI':
        test_dataset = load_decathlon_datalist(datalist_json,
                                            True,
                                            "test",
                                            base_dir=data_dir)
        
        test_ds = Dataset(data = test_dataset, transform=val_transforms)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = DataLoader(test_ds, 
                                shuffle=False, 
                                batch_size=1, 
                                num_workers=args.workers, 
                                collate_fn = pad_list_data_collate,
                                sampler = test_sampler,
                                pin_memory=torch.cuda.is_available(),
                                persistent_workers=True)
        # elif args.DA_library == "BG":
            
        #     if args.lowres is not None:
        #         folder_with_preprocessed_data = os.path.join(args.preproc_folder, 'Data_plans_v2.1_stage0')
        #     else:
        #         folder_with_preprocessed_data = os.path.join(args.preproc_folder, 'Data_plans_v2.1_stage1')
            
        #     dataset = load_dataset(folder_with_preprocessed_data)

        #     # Training // Validation //Tests splits
        #     _, _, dataset_ts = do_split(dataset, args)

        #     # Data Loader

        #     dl_ts =  DataLoader3D(dataset_ts, args.patch, args.patch_size, 1,
        #                         False, oversample_foreground_percent=0.33, pad_mode="constant", 
        #                         pad_sides = None, memmap_mode='r')
       
        # Make directory for test outputs
        path = os.path.join(args.logdir, 'test')
        if not os.path.isdir(path): os.mkdir(path)

        torch.cuda.empty_cache()
        # Inference size
        inf_size = [args.roi_x, args.roi_y, args.roi_x]

        # Load model
        if args.model_name == "unetr":
            model = UNETR(
                    in_channels=args.in_channels,
                    out_channels=args.out_channels,
                    img_size=(args.roi_x, args.roi_y, args.roi_z),
                    feature_size=args.feature_size,
                    hidden_size=args.hidden_size,
                    mlp_dim=args.mlp_dim,
                    num_heads=args.num_heads,
                    pos_embed=args.pos_embed,
                    norm_name=args.norm_name,
                    conv_block=True,
                    res_block=True,
                    dropout_rate=args.dropout_rate)
        elif args.model_name == "SwinUNETR":
            print("Using Monai Swin UNTER")

            model = SwinUNETR(
                img_size = (args.roi_x, args.roi_y, args.roi_z),
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                feature_size = 48,
                depths = [2, 2, 2, 2],
                num_heads = [3, 6, 12, 24],
                norm_name = "instance",
                dropout_path_rate = args.dropout_rate
            )

        else:
            raise ValueError('No existing model, set a valid model_name...')

        model.to(device)

        model_inferer = partial(sliding_window_inference,
                                roi_size=inf_size,
                                sw_batch_size=3,
                                predictor=model,
                                overlap=args.infer_overlap,
                                sw_device = "cuda",
                                device = 'cuda',
                                mode = 'gaussian'
                                )
        

        dice_acc = DiceMetric(include_background=True, reduction="mean", get_not_nans=True)
        post_pred = Compose(
            [Activations (softmax = True), AsDiscrete(argmax=True, threshold = True)]
            )
        post_label = AsDiscrete(to_onehot=args.out_channels)
        # model = UNETR(img_shape=tuple(inf_size), input_dim=args.in_channels, output_dim=args.out_channels,
        #         embed_dim=args.hidden_size, patch_size=16, num_heads=args.num_heads,
        #         ext_layers=[3, 6, 9, 12], norm='instance',
        #         base_filters=16,
        #         dim_linear_block=args.mlp_dim)
        
        

        netPath = os.path.join(args.logdir, 'best_model.pth')
        print(f'Using model located at {netPath}')
        if os.path.exists(netPath):
            # model.load_state_dict(torch.load(netPath)['state_dict'])
            # model.to(device)
            checkpoint = torch.load(netPath, map_location='cpu')
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k.replace('backbone.','')] = v
            model.load_state_dict(new_state_dict, strict=False)
            
        else:
            raise ValueError('No existing best_model.pth file in the given folder, check args.logdir term...')
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f'Starting {args.mode}'                                     )
        print(f'Number of images used for {args.mode}:', len(test_dataset))
        print('                                                     ')

        inference(model, 
                    test_loader = test_loader, 
                    images = images, 
                    device=device,
                    model_inferer = model_inferer,
                    acc_func = dice_acc, 
                    post_pred = post_pred,
                    post_label = post_label, 
                    output_dir=path, 
                    data_dir = data_dir, 
                    sw_batch_size=args.sw_batch_size, 
                    arguments = args)
        
        print('')
        print('Inference Done !')
        print('')
        print('Proceeding to predictions resampling')
        del test_loader, model
        torch.cuda.empty_cache()
        resample_nifti(args, path)
        print('')
        print('Procedding with test set evaluation')
        print('')

        eval_metrics(args)

        print('Evaluation ended')

    elif args.mode == 'FittingLR':
        #Save training inputs
        save_training_schemes(args)

        if args.distributed:
            torch.multiprocessing.set_start_method('fork', force=True)
        np.set_printoptions(formatter={'float': '{: 0.3f}'.format}, suppress=True)
        
        # Set Device
        args.gpu = gpu

        # Distributed
        if args.distributed:
            args.rank = args.rank * args.ngpus_per_node + gpu
            dist.init_process_group(backend=args.dist_backend,
                                    init_method=args.dist_url,
                                    world_size=args.world_size,
                                    rank=args.rank)
        # Check and set gpu
        use_cuda = torch.cuda.is_available()
        print("GPU Availavility: ", use_cuda)
        device = torch.device("cuda" if use_cuda else "cpu")
        torch.device(args.gpu)
        torch.backends.cudnn.benchmark = True
        args.test_mode = False
        
        # Inference size
        args.patch_size = inf_size = [args.roi_x, args.roi_y, args.roi_x]

        # Cranium extraction ?
        if args.craniumExtraction:
            caseDir = os.path.join(args.preproc_folder, "images_cranium")
            if not os.path.exists(caseDir): os.mkdir(caseDir)

            if len(os.listdir(caseDir)) < 147 and args.DA_library == "MONAI":
                print('Extracting cranium from images..')
                craniumExtraction(args)
                print('Done!')
            caseDir = os.path.join(args.preproc_folder, "Data_plans_v2.1_stage0_Cranium")

            if args.DA_library == "BG" and not os.path.exists(caseDir):
                print('Extracting cranium from images..')
                craniumExtraction(args)
                print('Done!')
        
        # Get images train[0]/val[1] loader_C
        loader = get_loader(args)      
        
        pretrained_dir = args.pretrained_dir
        if (args.model_name is None) or args.model_name == 'unetr':
            # if args.monai: 
            print('Using MONAI UNETr')
            model = UNETR(
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                img_size=(args.roi_x, args.roi_y, args.roi_z),
                feature_size=args.feature_size,
                hidden_size=args.hidden_size,
                mlp_dim=args.mlp_dim,
                num_heads=args.num_heads,
                pos_embed=args.pos_embed,
                norm_name=args.norm_name,
                conv_block=True,
                res_block=True,
                dropout_rate=args.dropout_rate)
            model = model.to(device)

        elif args.model_name == "SwinUNETR":
            print("Using Monai Swin UNTER")

            model = SwinUNETR(
                img_size = (args.roi_x, args.roi_y, args.roi_z),
                in_channels=args.in_channels,
                out_channels=args.out_channels,
                feature_size = 48,
                depths = [2, 2, 2, 2],
                num_heads = [3, 6, 12, 24],
                norm_name = "instance",
                dropout_path_rate = args.dropout_rate
            )
            # else:

            #     model = SACV.UNETR(img_shape=tuple(inf_size), input_dim=args.in_channels, output_dim=args.out_channels,
            #         embed_dim=args.hidden_size, patch_size=16, num_heads=args.num_heads,
            #         ext_layers=[3, 6, 9, 12], norm='instance',
            #         base_filters=16,
            #         dim_linear_block=args.mlp_dim)

            model = model.to(device)
        
       
        else:
            raise ValueError('Unsupported model ' + str(args.model_name))
        
         # Load Pretrained models
            #   Weights
        if args.resume_ckpt:
            fileDir = os.path.join(pretrained_dir, args.pretrained_model_name)
            model_dict = torch.load(fileDir, map_location=torch.device(device))
            model_dict = check_model(model, model_dict, fileDir)
            model.load_state_dict(model_dict)
            print('Using pretrained weights')
        else:
            print('Not using pretraining - Initializing weights (Xavier & Normal Distributions)...')
            model_dict = initialize_model(model)
            model.load_state_dict(model_dict)
        
        #   TorchScript
        if args.resume_jit:
            if not args.noamp:
                print('Training from pre-trained checkpoint does not support AMP\nAMP is disabled.')
                args.amp = args.noamp
            model = torch.jit.load(os.path.join(pretrained_dir, args.pretrained_model_name))
    
    # Set LOSS FUNCTION
        if args.loss_func == 'DiceCe':

            print('Using DiceCE loss')

            loss_func = DiceCELoss(to_onehot_y=True,
                                softmax=True,
                                sigmoid = False,
                                squared_pred=True,
                                reduction = "mean",
                                smooth_nr=0, 
                                smooth_dr=1e-6,
                                lambda_dice = 0.7,
                                lambda_ce = 0.30,
                                ce_weight = torch.cuda.FloatTensor([0.1,2])
                                )
        elif args.loss_func == "Tversky":

            print('Using Tversky loss')
            
            loss_func = TverskyLoss(include_background= True,
                                    to_onehot_y = True,
                                    sigmoid = False,
                                    softmax = True,
                                    alpha = 0.3, 
                                    beta = 0.7,
                                    reduction = "mean",
                                    smooth_dr = 1e-6,
                                    smooth_nr = 0
                                    )
        
        elif args.loss_func == 'DiceFocal':

            print('Using DiceFocal loss')
            
            loss_func = DiceFocalLoss(include_background=True,
                                    to_onehot_y = True,
                                    sigmoid = False,
                                    softmax = True,
                                    squared_pred= True,
                                    jaccard = False,
                                    reduction = 'mean',
                                    smooth_dr = 0, 
                                    smooth_nr = 1e-6,
                                    gamma = 0.85,
                                    focal_weight = [0.1, 70],
                                    lambda_dice= 0.70,
                                    lambda_focal= 0.5
                                    )
                                
        else:

            print(f'{args.loss_func} - Not found. Porceding with Dice loss')

            loss_func = DiceLoss(include_background=True, 
                                to_onehot_y=False, 
                                sigmoid=True, 
                                softmax=False, 
                                squared_pred=False, 
                                jaccard=False, 
                                reduction='mean', 
                                smooth_nr=1e-05, 
                                smooth_dr=1e-05)   

        post_label = AsDiscrete(to_onehot = args.out_channels)
        post_pred = Compose(
            [Activations (softmax = True), AsDiscrete(argmax=True, to_onehot = args.out_channels)]
            )
        
        dice_acc = DiceMetric(include_background=False, reduction="mean", get_not_nans=True)

        model_inferer = partial(sliding_window_inference,
                                roi_size=inf_size,
                                sw_batch_size=args.sw_batch_size,
                                predictor=model,
                                overlap=args.infer_overlap,
                                sw_device = "cuda",
                                device = 'cuda',
                                )
        

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('Total parameters count', pytorch_total_params)

                   

        if args.distributed:
            
            if args.norm_name == 'batch':
                model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
            model = torch.nn.parallel.DistributedDataParallel(model,
                                                            device_ids=[args.gpu],
                                                            output_device=args.gpu,
                                                            find_unused_parameters=True).to(device)
        # OPTIMIZER
        if args.optim_name == 'adam':
            optimizer = torch.optim.Adam(model.parameters(),
                                        lr=args.optim_lr,
                                        eps = 1e-5,
                                        weight_decay=args.reg_weight)
        elif args.optim_name == 'adamw':
            optimizer = torch.optim.AdamW(model.parameters(),
                                        lr=args.optim_lr,
                                        eps = 1e-6,
                                        weight_decay=args.reg_weight)
        elif args.optim_name == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(),
                                        lr=args.optim_lr,
                                        momentum=args.momentum,
                                        nesterov=True,
                                        weight_decay=args.reg_weight)
        else:
            raise ValueError('Unsupported Optimization Procedure: ' + str(args.optim_name))

        # LR SCHEDULER

        if args.lrschedule == 'CycleLR':
            scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, 
                                            base_lr = args.optim_lr,
                                            max_lr = 1, 
                                            step_size_up=50, 
                                            step_size_down=None, 
                                            mode='triangular', 
                                            gamma=1.0, 
                                            scale_fn=None, 
                                            scale_mode='cycle', 
                                            cycle_momentum=True, 
                                            base_momentum=0.8, 
                                            max_momentum=0.9, 
                                            last_epoch=- 1, 
                                            verbose=False)

        best_acc = 0
        start_epoch = 0
        val_acc_max = 0.
        val_rec_max = 0.

        # RESUME TRAINING
        latestmodelPath = os.path.join(args.logdir, "latest_model.pth")

        if os.path.exists(latestmodelPath):

            checkpoint = torch.load(latestmodelPath, map_location='cpu')
            checkpoint0 = torch.load(f'{latestmodelPath[:-16]}best_model.pth', map_location='cpu')
            checkpoint1 = torch.load(f'{latestmodelPath[:-16]}best_recall_model.pth', map_location='cpu')

            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in checkpoint['state_dict'].items():
                new_state_dict[k.replace('backbone.','')] = v
            model.load_state_dict(new_state_dict, strict=False)
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch']
                start_epoch +=1
            if 'best_acc' in checkpoint:
                best_acc = checkpoint0['best_acc']
                val_acc_max = best_acc
            if 'best_recall' in checkpoint1:
                val_rec_max = checkpoint1['best_recall']
            
            if args.multiOptim:
                optimizer1.load_state_dict(checkpoint['optimizer1'])
                optimizer2.load_state_dict(checkpoint['optimizer2'])
                optimizer = [optimizer1,  optimizer2]
            else:
                optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                    if args.multOptim:
                        scheduler1.load_state_dict(checkpoint['scheduler1'])
                        scheduler2.load_state_dict(checkpoint['scheduler2'])
                        scheduler = [scheduler1, scheduler2]
                    else:
                        scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(latestmodelPath, start_epoch, best_acc))    
            model = model.to(device)
        # RUN TRAINING
        accuracy = run_training(model=model,
                                train_loader=loader[0],
                                val_loader=loader[1],
                                optimizer=optimizer,
                                loss_func=loss_func,
                                acc_func=dice_acc,
                                args=args,
                                model_inferer=model_inferer,
                                scheduler=scheduler,
                                start_epoch=start_epoch,
                                val_acc_max = val_acc_max,
                                val_rec_max = val_rec_max,
                                post_label=post_label,
                                post_pred=post_pred)
        return accuracy


if __name__ == '__main__':
    main()