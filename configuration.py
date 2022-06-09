import numpy as np
import SimpleITK as sitk
import os
from glob import glob
import shutil
import json
import pickle as pkl


#def preprocessing():
''' ############################## Preprocessing #########################################

Preprocessing of the whole dataset as described by Isensee et al. (2019).

Images (labels) should be in the database_images/ (database_labels/) 
dir inside: ./nnunet/nnUNet_base/nnUNet_raw/Task00_grid/ 

Git repository: https://github.com/perecanals/nnunet_vh2020.git
Original nnunet (Isensee et al. 2020[1]): https://github.com/MIC-DKFZ/nnUNet.git

[1] Fabian Isensee, Paul F. JÃƒÆ’Ã‚Â¤ger, Simon A. A. Kohl, Jens Petersen, Klaus H. Maier-Hein "Automated Design of Deep Learning 
Methods for Biomedical Image Segmentation" arXiv preprint arXiv:1904.08128 (2020).

    '''
################################## Preprocessing #########################################

# Paths
path_images_base = './database_vh/database_images'
path_labels_base = './database_vh/database_labels'

path_imagesTr = './unetr_base/raw_data/Cerebral300/imagesTr'
path_labelsTr = './unetr_base/raw_data/Cerebral300/labelsTr'
path_imagesTs = './unetr_base/raw_data/Cerebral300/imagesTs'
path_labelsTs = './unetr_base/raw_data/Cerebral300/labelsTs'

task_path = './unetr_base/raw_data/Cerebral300'
if not os.path.isdir(task_path): os.mkdir(task_path)
if not os.path.isdir(path_imagesTr): os.mkdir(path_imagesTr)
if not os.path.isdir(path_labelsTr): os.mkdir(path_labelsTr)
if not os.path.isdir(path_imagesTs): os.mkdir(path_imagesTs)
if not os.path.isdir(path_labelsTs): os.mkdir(path_labelsTs)

# List all available images and labels
dir_images_base = os.fsencode(path_images_base)
dir_labels_base = os.fsencode(path_labels_base)
list_images_base = []; list_labels_base = []
for file in os.listdir(dir_images_base):
    filename = os.fsdecode(file)
    if filename.endswith('.gz'):
        list_images_base.append(filename)
        continue
    else:
        continue
for file in os.listdir(dir_labels_base):
    filename = os.fsdecode(file)
    if filename.endswith('.gz'):
        list_labels_base.append(filename)
        continue
    else:
        continue

print('Total number of images available in database:', len(list_images_base))
print('                                                                    ')

# Remove all preexisting nifti files
print('Removing possibly preexisting nifti files...')


print('done')
print('    ')

# Copy files to corresponding directories
print('Copying new files...')
# Partition train&val / test 
# partition = int(0.80*len(list_images_base))
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
#     # Test
# list_images_base_testing = list_images_base[samp_tr:]
# list_labels_base_testing = list_labels_base[samp_tr:]
#     # Train
# list_images_base = list_images_base[:samp_tr]
# list_labels_base = list_labels_base[:samp_tr]

# for image in list_images_base:
#     # Spacing and Resizing
#     print(f'Resizing and resampling imageTr {image}')
#     pkl_file = f'./unetr_base/preprocessed/Task300_Cerebral/Data_plans_v2.1_stage0/{image[:-7]}.pkl'
#     with open(pkl_file, 'rb') as f:
#         a = pkl.load(f)
#     Size = a['size_after_resampling']
#     Shape = [Size[0], Size[2], Size[1]]
#     im_stk = sitk.ReadImage(os.path.join(path_images_base, image))
#     resampled_img = sitk.Resample(im_stk, Shape, sitk.Transform(), sitk.sitkNearestNeighbor, im_stk.GetOrigin(),[0.67910341, 0.67910341, 0.63218294],
#                         im_stk.GetDirection(), 0.0, im_stk.GetPixelID())
#     sitk.WriteImage(resampled_img, os.path.join(path_imagesTr, image))
#     # shutil.copyfile(os.path.join(path_images_base, image),   os.path.join(path_imagesTr, image))
for label in list_labels_base:
    print(f'Resizing and resampling labelTr {label}')
    pkl_file = f'./unetr_base/preprocessed/Task300_Cerebral/Data_plans_v2.1_stage0/{label[:-7]}.pkl'
    with open(pkl_file, 'rb') as f:
        a = pkl.load(f)
    Size = a['size_after_resampling']
    Shape = [Size[0], Size[2], Size[1]]
    im_stk = sitk.ReadImage(os.path.join(path_labels_base, label))
    resampled_img = sitk.Resample(im_stk, Shape, sitk.Transform(), sitk.sitkNearestNeighbor, im_stk.GetOrigin(),[0.67910341, 0.67910341, 0.63218294],
                        im_stk.GetDirection(), 0.0, im_stk.GetPixelID())
    sitk.WriteImage(resampled_img, os.path.join(path_labelsTr, label))
#     # shutil.copyfile(os.path.join(path_labels_base, label),   os.path.join(path_labelsTr, label))
# for image in list_images_base_testing:
#     print(f'Resizing and resampling imageTs {image}')
#     pkl_file = f'./unetr_base/preprocessed/Task300_Cerebral/Data_plans_v2.1_stage0/{image[:-7]}.pkl'
#     with open(pkl_file, 'rb') as f:
#         a = pkl.load(f)
#     Size = a['size_after_resampling']
#     Shape = [Size[0], Size[2], Size[1]]
#     im_stk = sitk.ReadImage(os.path.join(path_images_base, image))
#     resampled_img = sitk.Resample(im_stk, Shape, sitk.Transform(), sitk.sitkNearestNeighbor, im_stk.GetOrigin(),[0.67910341, 0.67910341, 0.63218294],
#                         im_stk.GetDirection(), 0.0, im_stk.GetPixelID())
#     sitk.WriteImage(resampled_img, os.path.join(path_imagesTs, image))
#     # shutil.copyfile(os.path.join(path_images_base, image),   os.path.join(path_imagesTs, image))
# for label in list_labels_base_testing:
#     print(f'Resizing and resampling labelsTs {image}')
#     pkl_file = f'./unetr_base/preprocessed/Task300_Cerebral/Data_plans_v2.1_stage0/{image[:-7]}.pkl'
#     with open(pkl_file, 'rb') as f:
#         a = pkl.load(f)
#     Size = a['size_after_resampling']
#     Shape = [Size[0], Size[2], Size[1]]
#     im_stk = sitk.ReadImage(os.path.join(path_labels_base, label))
#     resampled_img = sitk.Resample(im_stk, Shape, sitk.Transform(), sitk.sitkNearestNeighbor, im_stk.GetOrigin(),[0.67910341, 0.67910341, 0.63218294],
#                         im_stk.GetDirection(), 0.0, im_stk.GetPixelID())
#     sitk.WriteImage(resampled_img, os.path.join(path_labelsTs, label))
    # shutil.copyfile(os.path.join(path_labels_base, label), os.path.join(path_labelsTs, label))

print('done')
print('    ')

# Write the .json file 

print('Total number of preprocessed images available in database:', len(list_images_base))
print('                                                                    ')

    
# We generate an order vector to shuffle the samples for training
print(f'Shuffling dataset (SEED = {100})')

np.random.seed(100)
order = np.arange(len(list_images_base))
np.random.shuffle(order)

list_images_base_sh = [list_images_base[i] for i in order]
list_labels_base_sh = [list_labels_base[i] for i in order]



    
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
    "dataset size":DATASET_SIZE,
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

imagesTr = './imagesTr/'
labelsTr = './labelsTr/'
imagesTs = './imagesTs/'
labelsTs = './labelsTs/'

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

filename = os.path.join(task_path, 'dataset_0.json')

with open(filename, 'w') as outfile:
    json.dump(dataset, outfile)

