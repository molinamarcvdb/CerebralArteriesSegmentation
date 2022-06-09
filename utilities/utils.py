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

from logging import raiseExceptions
import torch
import os
import SimpleITK as sitk
import shutil
import numpy as np
import nibabel as nib
from nibabel.processing import conform, smooth_image
from scipy.ndimage import gaussian_laplace
from skimage.measure import regionprops, label
from nilearn import plotting
from nilearn.image import resample_to_img, load_img
from SimpleITK import BinaryMorphologicalClosingImageFilter
from scipy.spatial.distance import euclidean
import matplotlib
matplotlib.use("agg")
import matplotlib.pyplot as plt
import json
from batchgenerators.utilities.file_and_folder_operations import *
from glob import glob

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

def distributed_all_gather(tensor_list,
                           valid_batch_size=None,
                           out_numpy=False,
                           world_size=None,
                           no_barrier=False,
                           is_valid=None):

    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g,v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out

def plot_progress(TRAIN_LOSS, VAL_LOSS, VAL_ACC, epoch, args):
    
    font = {'weight': 'normal',
            'size': 18}

    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(30, 24))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    x_values = list(range(epoch + 1))
       
    ax.plot(x_values, TRAIN_LOSS, color='b', ls='-', label="loss_tr")

    ax.plot(x_values, VAL_LOSS, color='r', ls='-', label="loss_val")

    if len(VAL_ACC) == len(x_values):
        ax2.plot(x_values, VAL_ACC, color='g', ls='--', label="evaluation metric")

    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax2.set_ylabel("evaluation metric")
    # ax.set_ylim([0, 1])
    # ax2.set_ylim([0, 1])
    ax.legend()
    ax2.legend(loc=9)

    fig.savefig(os.path.join(args.logdir, "progress.png"))
    plt.close()

def plot_fitting_LR(TRAIN_LOSS, VAL_ACC, listLR, epoch, args):
    
    font = {'weight': 'normal',
            'size': 18}

    matplotlib.rc('font', **font)

    fig = plt.figure(figsize=(30, 24))
    ax = fig.add_subplot(111)
    ax2 = ax.twinx()

    x_values = list(range(epoch + 1))
       
    ax.plot(listLR, TRAIN_LOSS, color='b', ls='-', label="loss_tr")
# 
    # ax2.plot(listLR, VAL_ACC, color='r', ls='-', label="loss_val")

    if len(VAL_ACC) == len(x_values):
        ax2.plot(listLR, VAL_ACC, color='g', ls='--', label="evaluation metric")

    ax.set_xlabel("Learning Rates")
    ax.set_ylabel("Loss")
    ax2.set_ylabel("Evaluation metric")
    # ax.set_ylim([0, 1])
    # ax2.set_ylim([0, 1])
    ax.legend()
    ax2.legend(loc=9)

    fig.savefig(os.path.join(args.logdir, "progress_LR.png"))
    plt.close()

def check_model(model, model_dict, fileDir):
    if len(model.state_dict().keys())==len(model_dict.keys()):
        layers = list()
        # Search incompatibilitites
        for key in model_dict.keys():
            m_shape = model.state_dict()[str(key)].shape
            m_d_shape = model_dict[str(key)].shape
            if m_shape != m_d_shape:
                print()
                print(f'Layer {key} needs to be adapted')
                layers.append(tuple([str(key), m_shape]))
        # Adapt shapes
        if len(layers) > 0: 
            for item in layers:
                layer = model_dict[item[0]]
                print( 'Initial', item[0], layer.shape)
                # Generate tensor
                tensor = torch.empty(item[1], dtype=torch.float32, device = 'cuda')
                # Initialize weights
                if tensor.dim() < 2:
                    mean, std = torch.mean(layer), torch.std(layer)
                    tensor = torch.nn.init.normal_(tensor, mean, std)
                else:
                    tensor = torch.nn.init.xavier_normal_(tensor, gain = torch.nn.init.calculate_gain('relu'))
                # Charge tensor to model
                model_dict[item[0]] = tensor
                print('Final', item[0], model_dict[item[0]].shape)
                print()
            # Move previous model
            caseDir = os.path.dirname(fileDir)
            newDir = os.path.join(caseDir, 'past_pretrained')
            if not os.path.isdir(newDir): os.mkdir(newDir)
            shutil.copyfile(fileDir, os.path.join(newDir, os.path.basename(fileDir)))
            # Save model 
            torch.save(model_dict, fileDir)

            return model_dict

        else:
            print('No incompatibilities found between the pretrained model and our model, proceding...')
            print()
            return model_dict
    else:
        raise ValueError('There are not the same layers in the UNETR model and the pretrained one !')

def initialize_model(model):
    model_dict = model.state_dict()
    for key in model_dict.keys():
        layer = model_dict[str(key)]
       
        # Generate tensor
        tensor = torch.empty(layer.shape, dtype=torch.float32, device = 'cuda')
        # Initialize weights
        if tensor.dim() < 2:
            mean, std = torch.mean(layer), torch.std(layer)
            tensor = torch.nn.init.normal_(tensor, mean, std)
        else:
            tensor = torch.nn.init.xavier_normal_(tensor, gain = torch.nn.init.calculate_gain('relu'))
        # Charge tensor to model
        model_dict[str(key)] = tensor
    return model_dict
    
def loss_normalization(list_loss, norm_loss):
    '''
    Input: 
            - A list with the accumulated losses per each epoch
            - A list with the past losses per epoch normalized
    '''
    if isinstance(list_loss, list) == True:
        initial_loss = list_loss[0]
        last_item = list_loss[-1]
        norm = 1 + ((last_item - initial_loss)/initial_loss)
        norm_loss.append(norm)
        return norm_loss
    pass

def save_training_schemes(args):
    trainProperties = {}
    trainProperties['Batch_size'] = args.batch_size
    trainProperties['Loss_Function'] = args.loss_func
    trainProperties['fold'] = args.fold
    trainProperties['Transforms'] = args.transforms
    trainProperties['Initial_LR'] = args.optim_lr
    trainProperties['LR_scheduler'] = args.lrschedule
    trainProperties['Optimizer'] = args.optim_name
    trainProperties['Patch_Size'] = (args.roi_x, args.roi_y, args.roi_z)
    trainProperties['Hidden_size'] = args.hidden_size
    trainProperties['Weigh_decay'] = args.reg_weight
    trainProperties['DA_library'] = args.DA_library

    if args.resume_ckpt:
        trainProperties['Pretrained'] = args.pretrained_model_name
    else:
        trainProperties['Pretrained'] = None

    filename = os.path.join(args.logdir, 'trainProperties.json')
    
    with open(filename, 'w') as outfile:
        json.dump(trainProperties, outfile)



def from_numpy_to_itk(image_np):#, image_itk):

    # # read image file
    # reader = sitk.ImageFileReader()
    # reader.SetFileName(image_itk)
    # image_itk = reader.Execute()

    image_np = np.transpose(image_np, (2, 1, 0))
    image = sitk.GetImageFromArray(image_np)
    # image.SetDirection(image_itk.GetDirection())
    # image.SetSpacing(image_itk.GetSpacing())
    # image.SetOrigin(image_itk.GetOrigin())
    return image

def unpack_dataset(args):

    print('                                        ')
    print(' Unpacking Dataset for preprocessing use')
    if args.lowres is not None:
        dir_i = os.path.join(args.preproc_folder, 'images_lowres')
        dir_l = os.path.join(args.preproc_folder, 'labels_lowres')
    else:
        dir_i = os.path.join(args.preproc_folder, 'images_fullres')
        dir_l = os.path.join(args.preproc_folder, 'labels_fullres')
    if not os.path.exists(dir_i):  

        preDir = args.preproc_folder
        # caseDir = './unetr_base/preprocessed/Data_plans_v2.1_stage0'

        if args.lowres is not None:    
            args.source = './Data_plans_v2.1_stage0/'
        else:
            args.source = './Data_plans_v2.1_stage1/'

        caseDir = os.path.join(preDir, args.source)

        list_npy = []
        for case in sorted(os.listdir(caseDir)):
            if case.endswith('.npy'):
                list_npy.append(os.path.join(caseDir, case))
                
        for case in list_npy:

            label = np.abs(np.load(case)[1]) # -1- --> label
            image = np.load(case)[0]

            if args.lowres is not None:
                dir_i = os.path.join(args.preproc_folder, 'images_lowres')
                dir_l = os.path.join(args.preproc_folder, 'labels_lowres')
            else:
                dir_i = os.path.join(args.preproc_folder, 'images_fullres')
                dir_l = os.path.join(args.preproc_folder, 'labels_fullres')


            if not os.path.isdir(dir_i): os.mkdir(dir_i)
            if not os.path.isdir(dir_l): os.mkdir(dir_l)

            filename_i = os.path.join(dir_i, f'{case[-12:]}')
            filename_l = os.path.join(dir_l, f'{case[-12:]}')

            np.save(filename_i, image)
            np.save(filename_l, label)

    return dir_i, dir_l

def generate_MONAI_json(splitPath, datalist_json, args):

    print(' Transforming .json for MONAI classes')
    
    # Load file in which training and validation splitting assessement is saved 
    splits = load_pickle(splitPath)

    # Load .json file where test cases are dict listed
    with open(datalist_json) as json_file:
        js_dict = json.load(json_file)
    
    # Get dict listed train//val cases
    train_fold = splits[args.fold]['train']
    val_fold = splits[args.fold]['val']
    test_fold = js_dict[f'fold {args.fold}']['test']

    # Set directories where cases are gathere
    if args.preprocessing is None:
        imagesTr = './imagesTr/'
        labelsTr = './labelsTr/'
        imagesTs = './imagesTs/'
        labelsTs = './labelsTs/'
    else:
        if args.lowres is not None:
            imagesTr = './images_lowres/'
            labelsTr = './labels_lowres/'
            if args.craniumExtraction:
                imagesTr = './images_cranium/'
        else:
            imagesTr = './images_fullres/'
            labelsTr = './labels_fullres/'

    list_imagesTr_json = []
        

    list_labelsTr_json = []
    list_imagesVl_json = []
    list_labelsVl_json = []
    list_imagesTs_json = []

    if args.preprocessing is None:
        list_imagesTr_json = [os.path.join(imagesTr, case + '.nii.gz') for case in train_fold] 
        list_labelsTr_json = [os.path.join(labelsTr, case + '.nii.gz') for case in train_fold]
        list_imagesVl_json = [os.path.join(imagesTr, case + '.nii.gz') for case in val_fold]        
        list_labelsVl_json = [os.path.join(labelsTr, case + '.nii.gz') for case in val_fold]
        list_imagesTs_json = [case for case in test_fold] 
    else:
        list_imagesTr_json = [os.path.join(imagesTr, case + '.npy') for case in train_fold] 
        list_labelsTr_json = [os.path.join(labelsTr, case + '.npy') for case in train_fold]
        list_imagesVl_json = [os.path.join(imagesTr, case + '.npy') for case in val_fold]        
        list_labelsVl_json = [os.path.join(labelsTr, case + '.npy') for case in val_fold]
        list_imagesTs_json = [case['image'] for case in test_fold] 

    dataset = {}
    dataset = {
        "name": "Data preprocessing",
        "description": "Cerebral Vessels Segmentation",
        "reference": "Hospital Vall dHebron",
        "licence": "-",
        "release": "1.0 02/02/2022",
        "tensorImageSize": "3D",
        "modality": {
            "0": "CT"
        },
        "labels": {
            "0": "background",
            "1": "vessel"
        },
        "numTraining": len(list_imagesTr_json),
        "numTest": len(list_imagesTs_json),
        "test": [],
        "training": [],
        "validation" : []
    }

    # We prepare the preprocessing samples for the json file
    
    # Training
    aux = []
    for idx, _ in enumerate(list_imagesTr_json):
        aux = np.append(aux, {
                        "image": list_imagesTr_json[idx],
                        "label": list_labelsTr_json[idx]
                    })
    aux = list(aux)#.tolist()
    # Validation
    aux1 = []
    for idx, _ in enumerate(list_imagesVl_json):
        aux1 = np.append(aux1, {
                        "image": list_imagesVl_json[idx],
                        "label": list_labelsVl_json[idx]
                    })
    aux1 = list(aux1)#.tolist()
    # Testing
    aux2 = []
    for idx2, _ in enumerate(list_imagesTs_json):
        aux2 = np.append(aux2, list_imagesTs_json[idx2])
    aux2 = list(aux2)
    # if len(aux2) > 0:
        # aux2 = aux2.tolist()

    dataset['training'] = aux
    dataset['validation'] = aux1
    dataset['test'] = aux2

    if args.lowres is not None:
        filename = f'{args.logdir}/dataset_lowres_fold_{args.fold}.json'
    else: 
        filename = f'{args.logdir}/dataset_fullres_fold_{args.fold}.json'

    with open(filename, 'w') as outfile:
        json.dump(dataset, outfile)

    return filename


        

def generate_json_BG(args):
    folder = args.preproc_folder
    # Check wether the input is a current directory
    if not os.path.isdir(folder): raise AssertionError(f"Is not a directory {folder} ")

    dir_images, dir_labels = unpack_dataset(args) #####################
    
    list_images_base = np.array(sorted(os.listdir(dir_images))) 
    list_labels_base = np.array(sorted(os.listdir(dir_labels)))
    list_case = list(zip(list_images_base, list_labels_base))
    np.random.shuffle(list_case)    
    list_images_base, list_labels_base = zip(*list_case)
    
    print('Total number of preprocessed images available in database:', len(list_images_base))
    print('                                                                    ')
    
    # Partition train&val / test 
    DATASET_SIZE = len(list_images_base)
    tr_prop = 0.8 # includes validation (5:1)
    ts_prop = 1.0 - tr_prop
    samp_tr = int(np.round(tr_prop * DATASET_SIZE))
    samp_ts = int(np.round(ts_prop * DATASET_SIZE))
    while samp_tr + samp_ts > DATASET_SIZE:
        samp_ts += -1
    
    print('                                            ')
    print('Number of images used for training:', samp_tr)
    print('Number of images used for testing: ', samp_ts)
    print('                                            ')
     
    # We generate an order vector to shuffle the samples for training
    print(f'Shuffling dataset (SEED = {100})')
    
    np.random.seed(100)
    order = np.arange(len(list_images_base))
    np.random.shuffle(order)

    list_images_base_sh = [list_images_base[i] for i in order]
    list_labels_base_sh = [list_labels_base[i] for i in order]
    
    
    if args.lowres is not None:

        imagesTr = './images_lowres/'
        labelsTr = './labels_lowres/'
        imagesTs = imagesTr
        labelsTs = labelsTr
    else:
        imagesTr = './images_fullres/'
        labelsTr = './labels_fullres/'
        
    # list_imagesTr_json = sorted(list_imagesTr_json)
    # list_labelsTr_json = sorted(list_labelsTr_json)
    # list_imagesTs_json = sorted(list_imagesTs_json)

    # Shuffling 
    # list_case_1 = list(zip(list_imagesTr_json, list_labelsTr_json))
    # np.random.shuffle(list_case_1)
    # list_imagesTr_json, list_labelsTr_json = zip(*list_case)
    # # list_labelsTs_json = sorted(list_labelsTs_json)

    dataset = {}
    dataset = {
        "name": "StrokeVessels",
        "description": "Upper Trunk Vessels Segmentation",
        "reference": "Hospital Vall dHebron",
        "licence": "-",
        "release": "1.0 08/01/2020",
        "tensorImageSize": "3D",
        "lowres": "LOWRES?",
        "trainer": "trainer",
        "modality": {
            "0": "CT"
        },
        "labels": {
            "0": "background",
            "1": "vessel"
        },
        "model": "Task300_Cerebral",
        "dataset config": 1,
        "dataset size":len(list_images_base),
        "numTraining": samp_tr,
        "numTest": samp_ts,
        "fold 0": {
            "training": [],
        },
        "fold 1": {
            "training": [],
        },
        "fold 2": {
            "training": [],
        },
        "fold 3": {
            "training": [],
        },
        "fold 4": {
            "training": [],
        }
    }

    

    array_images_base_sh = np.array(list_images_base_sh)
    folds = 5

    for fold in range(folds):
        if fold == 0:    
            imagesTs_fold = array_images_base_sh[: samp_ts]
            imagesTr_fold = array_images_base_sh[samp_ts:]
        elif fold == (folds - 1):
            imagesTs_fold = array_images_base_sh[fold * samp_ts:]
            imagesTr_fold = array_images_base_sh[: fold * samp_ts]
        else:
            imagesTs_fold = array_images_base_sh[fold * samp_ts: (fold + 1) * samp_ts]
            imagesTr_fold = np.append(array_images_base_sh[: fold * samp_ts], array_images_base_sh[(fold + 1) * samp_ts:])
        
        imagesTr_fold_json  = [None] * len(imagesTr_fold)
        labelsTr_fold_json  = [None] * len(imagesTr_fold)
        imagesTs_fold_json = [None] * len(imagesTs_fold)
        # list_labelsTs_json = [None] * len(list_labels_base_testing)

        for idx, _ in enumerate(imagesTr_fold):
            imagesTr_fold_json[idx] = imagesTr + imagesTr_fold[idx]
            
            labelsTr_fold_json[idx] = labelsTr + imagesTr_fold[idx]

        for idx, _ in enumerate(imagesTs_fold):
            imagesTs_fold_json[idx] = imagesTr + imagesTs_fold[idx]

        imagesTr_fold_json = sorted(imagesTr_fold_json)
        labelsTr_fold_json = sorted(labelsTr_fold_json)
        imagesTs_fold_json = sorted(imagesTs_fold_json)  

        # We prepare the preprocessing samples for the json file
        # train/val partition 80:20
        # Training
        aux = []
        for idx, _ in enumerate(imagesTr_fold_json):
            aux = np.append(aux, {
                            "image": imagesTr_fold_json[idx],
                            "label": labelsTr_fold_json[idx]
                        })
        aux = list(aux)#.tolist()
        # Validation
        aux1 = []
        for idx2, _ in enumerate(imagesTs_fold_json):
            aux1 = np.append(aux1, {
                            "image": imagesTs_fold_json[idx2],
                        })

        aux1 = list(aux1)#.tolist()
        # Testing
        dataset[f'fold {fold}']['training'] = aux
        dataset[f'fold {fold}']['test'] = aux1

    if args.lowres is not None:
        filename = f'{args.preproc_folder}/data_dicts/dataset_lowres.json'
    else: 
        filename = f'{args.preproc_folder}/data_dicts/dataset_fullres.json'

    with open(filename, 'w') as outfile:
        json.dump(dataset, outfile)
    
    shutil.copyfile(filename, os.path.join(args.logdir, 'dataset_lowres.json'))

    return filename

def craniumExtraction(args):

    if args.lowres is not None:
        dir_i = images_lowres = os.path.join(args.preproc_folder, 'images_lowres')
        # dir_l = os.path.join(args.preproc_folder, 'labels_lowres')
    else:
        pass

    # Create new folder to save the dataset
    caseDir = dir_c =  os.path.join(args.preproc_folder, "images_cranium")

    if args.DA_library == "BG":

        folder_with_preprocessed_data = os.path.join(args.preproc_folder, 'Data_plans_v2.1_stage0')

        print('Copy preprocessed lowres cases directory...')
        # shutil.copytree(folder_with_preprocessed_data, os.path.join(args.preproc_folder, 'Data_plans_v2.1_stage0_Cranium'))
        print("Done !")
        caseDir = dir_i = os.path.join(args.preproc_folder, 'Data_plans_v2.1_stage0_Cranium')

    for case in sorted(os.listdir(dir_i)):
        if case.endswith('.npy'):

            # Load iamge
            image = np.load(os.path.join(dir_i, case))
            
            if len(os.listdir(images_lowres)) < 147:
                # Aplly Laplacian filter
                laplacian = gaussian_laplace(image, sigma = 0.02, mode = "nearest")

                # Get cranium "binary" mask
                tolerance = 0.1 * np.ptp(laplacian)
                threshold = np.min(laplacian) + tolerance
                craniumMask = np.where(laplacian<=threshold, np.max(image), 0)

                # Get new image
                new_image = image-craniumMask

                # Set substitution value to the min value of the original image
                new_image = np.where(new_image==0, np.min(image), new_image)
                
            else:
                new_image = np.load(os.path.join(dir_c, case))

                image[0] = new_image

            # Save image
            if args.DA_library == "MONAI":
                filename = os.path.join(caseDir, case)
                np.save(filename, new_image)

            elif args.DA_library == "BG":
                filename = os.path.join(caseDir, case)
                np.save(filename, image)

def editDatalist(filename, args):
    '''
    With the purpose of generating a json for craniumExtracted images --> just changing ./images_lowres/ --> ./images_cranium/
    '''
    with open(filename) as json_file:
        js_dict = json.load(json_file)
    
    for i in range(5):
        for idx in range(len(js_dict['fold'+ ' '+ str(i)]['training'])):
            js_dict['fold'+ ' '+ str(i)]['training'][idx]['image'] = './images_cranium/' + js_dict['fold'+ ' '+ str(i)]['training'][idx]['image'][-12:]
    
        for idx in range(len(js_dict['fold'+ ' '+ str(i)]['test'])):
            js_dict['fold'+ ' '+ str(i)]['test'][idx] = './images_cranium/' + js_dict['fold'+ ' '+ str(i)]['test'][idx]['image'][-12:]

    filename = f'{args.preproc_folder}data_dicts/dataset_lowres_noCran.json'
    with open(filename, 'w') as outfile:
        json.dump(js_dict, outfile)

    shutil.copyfile(filename, os.path.join(args.logdir, 'dataset_lowres_noCran.json'))

    return os.path.join(args.logdir, 'dataset_lowres_noCran.json')  
        
   

def get_nifti_label(args, image):
    
    if args.preprocessing is None:
        casePath = os.path.join('.unetr_base/raw_data/Cerebral300/labelsTr', image + '.nii.gz')
    else:
        casePath = os.path.join(args.preproc_folder, 'gt_segmentations/'+ image + '.nii.gz')
   
    nifti = nib.load(casePath)
    sitk_label = sitk.ReadImage(casePath)
    return nifti, sitk_label

def weight_CE(target):

    weights_0 = []
    weights_1 = []

    for idx_batch in range(target.shape[0]):
        # 
        case = target[idx_batch]
        background = case[0]
        foregorund = case[1]
        
        Shape = background.shape
        total_num_voxels = Shape[0]*Shape[1]*Shape[2]
        
        num_voxels_BG = torch.sum(background).item()
        num_voxels_FG = torch.sum(foregorund).item()
        
        freq_0 = torch.tensor(num_voxels_BG/total_num_voxels).item()
        freq_1 = torch.tensor(num_voxels_FG/total_num_voxels).item()
        
        weights_0.append(1/freq_0)
        weights_1.append(1/freq_1)
    
    weights_0 = np.mean(np.array(weights_0))
    weights_1 = np.mean(np.array(weights_1))
        
    return weights_0, weights_1

    
def getPredHeadAff(label_header, dictionary):
    '''
    This fucntion considers that the prediction has already been swapAxed and transposed (Anti-Preprocessing)'
    '''
    # Get Spacing
    Spacing = (dictionary['spacing_after_resampling'])
    Spacing =  [Spacing[1], Spacing[0], Spacing[2]][::-1]
    Spacing.append(1)

    # Get Origin
    Origin = (dictionary['itk_origin'])

    # Create affine
    affinePred =  0 * np.eye(4,4, k = 1)

    affinePred[0][0] = -Spacing[0]
    affinePred[1][1] = Spacing[1]
    affinePred[2][2] = Spacing[2]
    affinePred[3][3] = Spacing[3]
    affinePred[0][3] = abs(Origin[0])
    affinePred[1][3] = abs(Origin[1])
    affinePred[2][3] = Origin[2]
    
    # SIze
    Size = dictionary['size_after_resampling']
    Shape = (Size[1], Size[0], Size[2])[::-1] # 
    print(Shape)

    # Modify Header 
    head = label_header
    head['pixdim'] = [-1] + Spacing + [1, 1, 1]
    head['dim'][1:4] = Shape
    head['srow_x'] = affinePred[0]
    head['srow_y'] = affinePred[1]
    head['srow_z'] = affinePred[2]

    return affinePred, head

def resample_nifti(args, path):
    '''
    Resample predcitons saved in preds, need to catch also the label to resemble the prediction to it  
    '''
    test_dir_labels = os.path.join(path, 'labels')
    test_dir_preds = os.path.join(path, 'non_resampled')

    list_label = sorted(glob(os.path.join(test_dir_labels, '*.nii.gz')))
    list_pred  = sorted(glob(os.path.join(test_dir_preds , '*.nii.gz')))

    print(len(list_label), len(list_pred))

    assert len(list_label) == len(list_pred)

    for idx in range(len(list_label)):
        # Load files
        print(f'Evaluation for {list_pred[idx][-15:]}')
        target = nib.load(list_label[idx])
        pred  = nib.load(list_pred[idx])
        sitkIm = sitk.ReadImage(list_pred[idx])
        header = pred.header
        affine = pred.affine

        PP_arr = pred.get_fdata()

        # Label components
        print('Labelling components to eliminate over-segmentations...')
        labelMask = label(PP_arr)
        properties = regionprops(labelMask)

        areas = [x.area for x in properties] 
        idx_island = np.argmax(areas)
        centroid = np.round(properties[idx_island].centroid, 2)

        for id in range(len(properties)):
            distance_idx = euclidean(centroid, np.round(properties[id].centroid, 2))
            if properties[id].area < 750 or distance_idx > 100:
                labelnum = id + 1
                labelMask = np.where(labelMask == labelnum, 0, labelMask)
        labelMask = np.where(labelMask >= 1, 1, 0)   

        # BinaryMorphologicalClosingImageFilter
        # print('Performing binary closing...')
        # Closer = BinaryMorphologicalClosingImageFilter()
        # Closer.SetKernelRadius([1, 1, 1])
        # Postprocessed = Closer.Execute(sitkIm)
        # PP_arr = np.swapaxes(sitk.GetArrayFromImage(Postprocessed), 0, 2)

        # To NIFTI
        header['datatype'] = np.array(8, dtype = np.int32)
        nif_sv = nib.Nifti1Image(labelMask, affine, header)
        
        print(f'Previous shape: {pred.get_fdata().shape} ')
        print('Resampling...')

        # Resample to iamge
        pixdims = target.header['pixdim'][1:4]

        resampled_pred = conform(nif_sv, out_shape = target.shape , voxel_size = [pixdims[0], pixdims[1], pixdims[2]]).get_fdata()
        resampled = np.flipud(np.where(resampled_pred>=1, 1, 0))

        headerL = target.header
        headerL['datatype'] = np.array(8, dtype = np.int32)
        niigz = nib.Nifti1Image(resampled.astype(np.int32), target.affine, headerL)
        # resampled_pred = resample_to_img(nif_sv, target, interpolation = 'nearest')
        print(f'New shape: {niigz.shape} ')

        # Save pred
        predPath =  os.path.join(path, 'preds')
        if not os.path.exists(predPath): os.mkdir(predPath)
        nib.save(niigz, os.path.join(predPath, list_pred[idx][-15:]))


    # nifti_resampled = resample_to_img(nifti_pred, label)
    # print('Final shape', nifti_resampled.get_fdata().shape)
    