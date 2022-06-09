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

from ast import Raise
from logging import raiseExceptions
import os
import math
import numpy as np
from functools import partial
import torch
import shutil
from monai import transforms, data
from monai.data import load_decathlon_datalist, pad_list_data_collate
from training.generate_splits import do_split, generate_splits

from utilities.utils import generate_MONAI_json, generate_json_BG, editDatalist
from training.data_augmentation import default_3D_augmentation_params, get_default_augmentation, get_patch_size
from training.Data_loading import  load_dataset, DataLoader3D, unpack_dataset

class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank:self.total_size:self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[:(self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0,high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def get_loader(args):
    
    if args.preprocessing is None:
        data_dir = args.data_dir   
        datalist_json = os.path.join(data_dir, args.json_list)
        if not os.path.exists(datalist_json):
            raise ValueError('Not json found: Run configuration.py to generate vanilla json')
    else:
        data_dir = args.preproc_folder
        if args.lowres is not None:
            filename_org = filename =  f'{args.preproc_folder}data_dicts/dataset_lowres.json'

            if args.craniumExtraction:
                print('Using cranium extracted images')
                filename = f'{args.preproc_folder}data_dicts/dataset_lowres_noCran.json'
        else: 
            filename = f'{args.preproc_folder}data_dicts/dataset_fullres.json'
            pass 
        shutil.copyfile(filename_org, os.path.join(args.logdir, "dataset_lowres.json"))
        folder_filename = os.path.join(args.logdir, os.path.basename(filename))
        if os.path.exists(filename):
            print('                                           ')
            print(f'Using json dict of preprocessed cases:{filename}')
            shutil.copyfile(filename, folder_filename)
            datalist_json = filename
        else:
            print('                                           ')
            print('Generating json dict for preprocessed cases')
            
            datalist_json = generate_json_BG(args)
            
            if args.craniumExtraction:
                datalist_json = editDatalist(datalist_json, args)
    
###############

    train_transform = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.AdjustContrastd(keys = ["image"],
                                        gamma = 0.9
                                        ),
            # transforms.RandGaussianSmoothd(keys = ["image"],
            #                             sigma_x=(0.05, 0.3), 
            #                             sigma_y=(0.05, 0.3), 
            #                             sigma_z=(0.05, 0.3), 
            #                             prob=0.1,   
            #                             approx='sampled'
            #                             ),            
            transforms.RandRotated(keys = ["image", "label"], 
                                    prob=0.15, 
                                    range_x = [-0.15, 0.15], range_y = [-0.15, 0.15], range_z = [-0.15, 0.15],
                                    mode = "nearest", 
                                    padding_mode = 'zeros', 
                                    align_corners=True),           
                      
            transforms.RandZoomd(keys = ["image", "label"], prob = 0.15, min_zoom = 0.95, max_zoom = 1.2, mode = "nearest"),                        
                       
            transforms.RandSpatialCropd(keys = ["image", "label"], 
                                        roi_size = (args.roi_x, args.roi_y, args.roi_x),
                                        max_roi_size=None, 
                                        random_center=True, 
                                        random_size=False),
                   
            transforms.ToTensord(keys=["image", "label"]),])

    val_transform = partial(transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.AdjustContrastd(keys = ["image"],
                                        gamma = 0.9
                                        ),
            # transforms.RandGaussianSmoothd(keys = ["image"],
            #                             sigma_x=(0.05, 0.3), 
            #                             sigma_y=(0.05, 0.3), 
            #                             sigma_z=(0.05, 0.3), 
            #                             prob=0.1,   
            #                             approx='sampled'
            #                             ),
            
            transforms.RandRotated(keys = ["image", "label"], prob=0.15, range_x = [-0.15, 0.15], range_y = [-0.15, 0.15], range_z = [-0.15, 0.15], mode = "nearest", padding_mode = 'zeros', align_corners=True),
            
                      
            transforms.RandZoomd(keys = ["image", "label"], prob = 0.15, min_zoom = 0.95, max_zoom = 1.2, mode = "nearest"),                        
                       
            transforms.RandSpatialCropd(keys = ["image", "label"], 
                                        roi_size = (args.roi_x, args.roi_y, args.roi_x),
                                        max_roi_size=None, 
                                        random_center=True, 
                                        random_size=False),
                   
            transforms.ToTensord(keys=["image", "label"]),]))
    
    min_transforms = transforms.Compose([
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.RandSpatialCropd(keys = ["image", "label"], 
                                        roi_size = (args.roi_x, args.roi_y, args.roi_x),
                                        max_roi_size=(args.roi_x, args.roi_y, args.roi_x), 
                                        random_center=True, 
                                        random_size=False),
            transforms.ToTensord(keys=["image", "label"]),])

    if args.test_mode:
        test_files = load_decathlon_datalist(datalist_json,
                                            True,
                                            "test",
                                            base_dir=data_dir)
        
        test_ds = data.Dataset(data=test_files, transform=val_transform)
        test_sampler = Sampler(test_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(test_ds,
                                     batch_size=1,
                                     shuffle=False,
                                     collate_fn = pad_list_data_collate,
                                     num_workers=args.workers,
                                     sampler=test_sampler,
                                     pin_memory=True,
                                     persistent_workers=True)
        loader = test_loader
    else:  
        # Set preprocessed folder
        if args.lowres is not None:
            folder_with_preprocessed_data = os.path.join(args.preproc_folder, 'Data_plans_v2.1_stage0')
            if args.craniumExtraction:
                folder_with_preprocessed_data = os.path.join(args.preproc_folder, 'Data_plans_v2.1_stage0_Cranium')

        else:
            folder_with_preprocessed_data = os.path.join(args.preproc_folder, 'Data_plans_v2.1_stage1')
        
        # Generate split_final.pkl form dataset.json
        splitPath = os.path.join(args.logdir, 'splits_final.pkl')
        if not os.path.exists(splitPath):
            generate_splits(1, args)
        
        if args.DA_library != 'BG':

            filename = f'{args.logdir}/dataset_lowres_fold_{args.fold}.json'
            if not os.path.exists(filename):
                datalist_json = generate_MONAI_json(splitPath, datalist_json, args)
            
            datalist = load_decathlon_datalist(filename,
                                            True,
                                            "training",
                                            base_dir=data_dir)
                    
            if args.use_normal_dataset:
                if args.transforms:
                    train_ds = data.Dataset(data=datalist, transform=train_transform)
                else:
                    train_ds = data.Dataset(data=datalist, transform=min_transforms)
            else:
                if args.transforms:
                    train_ds = data.CacheDataset(
                        data=datalist,
                        transform=train_transform,
                        cache_num=24,
                        cache_rate=1.0,
                        num_workers=args.workers,
                    )
                else:
                    print('trueeeeeeeeeeeeeeee')
                    train_ds = data.CacheDataset(
                        data=datalist,
                        transform=min_transforms,
                        cache_num=24,
                        cache_rate=1.0,
                        num_workers=args.workers)
            train_sampler = Sampler(train_ds, shuffle = False) if args.distributed else None
            train_loader = data.DataLoader(train_ds,
                                        batch_size=args.batch_size,
                                        shuffle=(train_sampler is None),
                                        collate_fn = pad_list_data_collate,
                                        num_workers=args.workers,
                                        sampler=train_sampler,
                                        pin_memory=True,
                                        persistent_workers=True)

            val_files = load_decathlon_datalist(filename,
                                                True,
                                                "validation",
                                                base_dir=data_dir)
            if args.transforms:
                val_ds = data.Dataset(data=val_files, transform=val_transform)
            else:
                val_ds = data.Dataset(data=val_files, transform=min_transforms)
            val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
            val_loader = data.DataLoader(val_ds,
                                        batch_size=1,
                                        shuffle=False,
                                        collate_fn = pad_list_data_collate,
                                        num_workers=args.workers,
                                        sampler=val_sampler,
                                        pin_memory=True,
                                        persistent_workers=True)
        else:
            # Get case identifiers to gather the dataset filenames to the cases
            dataset = load_dataset(folder_with_preprocessed_data)

            # Training // Validation splits
            dataset_tr, dataset_val, _ = do_split(dataset, args)

            basic_generator_patch_size = get_patch_size(args.patch_size, default_3D_augmentation_params['rotation_x'],
                                                            default_3D_augmentation_params['rotation_y'],
                                                            default_3D_augmentation_params['rotation_z'],
                                                            default_3D_augmentation_params['scale_range'])

            dl_tr = DataLoader3D(dataset_tr, basic_generator_patch_size, args.patch_size, args.batch_size,
                                False, oversample_foreground_percent=0.33, pad_mode="constant", 
                                pad_sides = None, memmap_mode='r')

            dl_val = DataLoader3D(dataset_val, basic_generator_patch_size, args.patch_size, 1,
                                False, oversample_foreground_percent=0.33, pad_mode="constant", 
                                pad_sides = None, memmap_mode='r')
                
            # Batch Generators Data Augmentation Package
            train_loader, val_loader  = get_default_augmentation(dl_tr,
                                                                dl_val,
                                                                patch_size = (args.roi_x, args.roi_y, args.roi_x),
                                                                params = default_3D_augmentation_params    
                                                                )
        loader = [train_loader, val_loader]

    return loader