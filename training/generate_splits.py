
from collections import OrderedDict
import pickle
import os
import numpy as np
import json
import shutil
from batchgenerators.utilities.file_and_folder_operations import *
from sklearn.model_selection import KFold

def generate_splits(DATASET_CONFIG, args):
    ''' ############################# Set fold files #########################################

    Set training and validation splits according to dataset configuration 1 (variable testing set).

    Git repository: https://github.com/perecanals/nnunet_vh2020.git

    '''
    #Paths
    if args.preprocessing is None:
        path_preprocessed = f'{args.data_dir}'
    else:
        path_preprocessed = f'{args.preproc_folder}/data_dicts/'
    
    if os.path.exists(os.path.join(path_preprocessed, 'splits_final.pkl')):
        os.remove(os.path.join(path_preprocessed, 'splits_final.pkl'))

    if args.preprocessing is None:
        dataset_json = os.path.join(args.data_dir, 'dataset_0.json')
    elif args.craniumExtraction is not None:
        dataset_json = os.path.join(args.logdir, 'dataset_lowres_noCran.json')
    else:    
        dataset_json = os.path.join(args.logdir, 'dataset_lowres.json')
        

    list_imagesTr = []
    list_imagesTs = []

    with open(dataset_json) as json_file:
        data = json.load(json_file)
        assert data['dataset config'] == DATASET_CONFIG
        if args.preprocessing is None:
            for image in data[f'fold {args.fold}']['training']:
                list_imagesTr.append(image['image'][-15:])
            for image in data[f'fold {args.fold}']['test']:
                list_imagesTs.append(image[-15:])
        else:
            if args.craniumExtraction is not None:
                for image in data[f'fold {args.fold}']['training']:
                    list_imagesTr.append(image['image'][-12:])
                for image in data[f'fold {args.fold}']['test']:
                    list_imagesTs.append(image[-12:])
            else:
                for image in data[f'fold {args.fold}']['training']:
                    list_imagesTr.append(image['image'][-12:])
                for image in data[f'fold {args.fold}']['test']:
                    list_imagesTs.append(image['image'][-12:])

    array_imagesTr = np.array(list_imagesTr).astype('<U8')
    array_imagesTs = np.array(list_imagesTs).astype('<U8')
    
    np.random.seed(10) # 10 set at for no reason. You could set a second random seed here
    np.random.shuffle(array_imagesTr)
    # np.random.shuffle(array_imagesTs)
    splits = []

    tr_size = round(len(array_imagesTr) * 5 / 6)
    vl_size = round(len(array_imagesTr) / 6)
    while tr_size + vl_size > len(array_imagesTr): vl_size += -1

    print('Generating splits')
    print('                 ')

    for fold in range(5):
        split_fold = OrderedDict([
            ('train', array_imagesTr[:tr_size]), 
            ('val', array_imagesTr[tr_size:]),
            ('test', array_imagesTs)
        ])
        splits.append(split_fold)

        array_imagesTr = np.roll(array_imagesTr, vl_size)
        
        if fold == args.fold:
            print('Fold', args.fold)
            print('train:', split_fold['train'])
            print('val:', split_fold['val'])

    with open(os.path.join(path_preprocessed, 'splits_final.pkl'), 'wb') as handle:
        pickle.dump(splits, handle, protocol=pickle.HIGHEST_PROTOCOL)

    shutil.copyfile(os.path.join(path_preprocessed, 'splits_final.pkl'), os.path.join(args.logdir, 'splits_final.pkl'))



def do_split(dataset, args):
    """
    This is a suggestion for if your dataset is a dictionary (my personal standard)
    :return:
    """
    splits_file = join(args.logdir, "splits_final.pkl")
    if not isfile(splits_file):
        print("Creating new split...")
        splits = []
        all_keys_sorted = np.sort(list(dataset.keys()))
        kfold = KFold(n_splits=5, shuffle=True, random_state=12345)
        for i, (train_idx, test_idx) in enumerate(kfold.split(all_keys_sorted)):
            train_keys = np.array(all_keys_sorted)[train_idx]
            test_keys = np.array(all_keys_sorted)[test_idx]
            splits.append(OrderedDict())
            splits[-1]['train'] = train_keys
            splits[-1]['val'] = test_keys
        save_pickle(splits, splits_file)

    splits = load_pickle(splits_file)

    if args.fold == "all":
        tr_keys = val_keys = list(dataset.keys())
    else:
        tr_keys = splits[args.fold]['train']
        val_keys = splits[args.fold]['val']
        ts_keys = splits[args.fold]['test'] 
        

    tr_keys.sort()
    val_keys.sort()
    

    dataset_tr = OrderedDict()
    for i in tr_keys:
        dataset_tr[i] = dataset[i]

    dataset_val = OrderedDict()
    for i in val_keys:
        dataset_val[i] = dataset[i]

    dataset_ts = OrderedDict()
    for i in ts_keys:
        dataset_ts[i] = dataset[i]
    
    
    return dataset_tr, dataset_val, dataset_ts
