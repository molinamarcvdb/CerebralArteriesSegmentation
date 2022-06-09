import os

import numpy as np
import SimpleITK as sitk
import pickle as pkl

from time import time
from tqdm import tqdm
from monai.inferers import sliding_window_inference


import nibabel as nib

# from batchgenerators.utilities.file_and_folder_operations import maybe_mkdir_p

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast

from monai.inferers import sliding_window_inference, SimpleInferer
from monai.data.utils import dense_patch_slices, compute_importance_map, decollate_batch
from monai.transforms import AsDiscrete, Spacing, Resize
from batchgenerators.augmentations.utils import resize

from utilities.utils import from_numpy_to_itk, get_nifti_label
from inferers.inference_export import resample_data_or_seg, save_segmentation_nifti_from_softmax

# from misc import timer

def inference(model, test_loader, images, device, model_inferer, acc_func, post_pred, post_label, output_dir, data_dir, sw_batch_size, arguments):
    args = arguments
    Dir = os.path.join(output_dir, 'preds')
    if not os.path.isdir(Dir): os.mkdir(Dir)

    model.eval()
    model.to(device)
      
    with torch.no_grad():
        print()
        print('Starting inference...')
        
        for idx, batch in enumerate(test_loader):
            
            if isinstance(batch, list):
                data = batch
            else:
                data = batch['image'].to(device)
            
            # data = batch['image'].to(device)
            with autocast(enabled=arguments.amp):
                img_name = batch['image_meta_dict']['filename_or_obj'][0].split('/')[-1]
                print(img_name)
                print(data.shape)
                pred = np.expand_dims(torch.argmax(torch.softmax(model_inferer(data)[0], axis = 0), axis = 0).cpu().numpy().astype(np.int64).squeeze(), axis = 0)
                print(pred.shape)
               
                
            # Get affine and header from label 
            label, sitk_label = get_nifti_label(args, images[idx])
            affine = label.affine
            header = label.header

            # # # Foregorund, Background
            # foreground = pred
            # background = 1 - foreground

            # pixdims = header['pixdim'][1:4]

            # # Initialize objects
            # resampling = Spacing(pixdim=pixdims, diagonal=True, mode='nearest')
            # resizing = Resize(spatial_size=label.get_fdata().shape, size_mode='all', mode = 'nearest')

            # # Resample
            # label_resampled = resampling(np.expand_dims(pred, 0), affine=affine)[0][0, :]
            # foreground_resampled = resampling(np.expand_dims(foreground, 0), affine=affine)[0][0, :]
            # background_resampled = resampling(np.expand_dims(background, 0), affine=affine)[0][0, :]
            # label_resampled = np.argmax(np.append([background_resampled], [foreground_resampled], axis=0), axis=0) 
            # label_resampled = pred
            # # Resizing
            # # label_RR = resizing(label_resampled)

            # Get preprocessing properties information
            pkl_file = f'{args.preproc_folder}Data_plans_v2.1_stage0/{images[idx]}.pkl'
            with open(pkl_file, 'rb') as f:
                properties = pkl.load(f)

            # Set filenames
            filename = os.path.join(output_dir, 'preds')
            if not os.path.isdir(filename): os.mkdir(filename)
            filename = os.path.join(filename, f'{images[idx]}.nii.gz')
            filename_l = os.path.join(output_dir, 'labels')
            if not os.path.exists(filename_l): os.mkdir(filename_l)

            # Saving softmax 
            save_segmentation_nifti_from_softmax(segmentation_softmax = pred,
                                                out_fname = filename,
                                                properties_dict = properties,
                                                order = 1,
                                                region_class_order = None, 
                                                seg_postprogess_fn = None,
                                                seg_postprocess_args = None,
                                                resampled_npz_fname = None,
                                                non_postprocessed_fname = None, 
                                                force_separate_z = None,  
                                                interpolation_order_z = 0,
                                                verbose = True)

            
            # # Get spacing and array Shape
            # a = pkl
            # Size = a['size_after_cropping']
            # Shape = (Size[1], Size[0], Size[2])[::-1]
            # Spacing = a['itk_spacing']
            # Spacing =  (Spacing[1], Spacing[0], Spacing[2])

            # Direction = a['itk_direction']
            # Origin = a['itk_origin']

            
            # Resampling and Resizing
            # im_stk = sitk.GetImageFromArray(pred)
            # print('sitk pred' , im_stk.GetSize())
            # resampled_img = sitk.Resample(im_stk, Shape, sitk.Transform(), sitk.sitkNearestNeighbor, sitk_label.GetOrigin(), Spacing, sitk_label.GetDirection(), 0.0, sitk_label.GetPixelID())
            # print('resampled', resampled_img.GetSize())
            # resampled_arr = np.array(sitk.GetArrayFromImage(resampled_img))

            # swapped_arr = np.swapaxes(resampled_arr, 0, 1)

            # resampled_img = sitk.GetImageFromArray(swapped_arr)
            # print(resampled_img.GetSize())

            # resampled_img.CopyInformation(sitk_label)
            # resampled_img = pred
            
            # # # Nifti format
            # nifti_pred = nib.Nifti1Image(resampled_img.astype(np.int64), affine, header)
            
            # # # Save
            # filename = os.path.join(output_dir, 'preds')
            # if not os.path.isdir(filename): os.mkdir(filename)
            # filename_l = os.path.join(output_dir, 'labels')
            # if not os.path.exists(filename_l): os.mkdir(filename_l)



            # # sitk.WriteImage(resampled_img, os.path.join(filename, f'{images[idx]}.nii.gz'))
            # # sitk.WriteImage(sitk_label,  os.path.join(filename_l, f'{images[idx]}.nii.gz'))
            # nib.save(nifti_pred, os.path.join(filename, f'{images[idx]}.nii.gz'))
            # nib.save(label, os.path.join(filename_l, f'{images[idx]}.nii.gz'))

            
def custom_sliding_window_inference(inputs, roi_size, sw_batch_size, predictor, overlap=0.25, blend_mode="constant"):
    """ Adapted from MONAI Alpha 0.1.0 (due to memory issues)
    
    Use SlidingWindow method to execute inference.

    Args:
        inputs (torch Tensor): input image to be processed (assuming NCHW[D])
        roi_size (list, tuple): the window size to execute SlidingWindow inference.
        sw_batch_size (int): the batch size to run window slices.
        predictor (Callable): given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap (float): Amount of overlap between scans.
        blend_mode (str): How to blend output of overlapping windows. Options are 'constant', 'gaussian'. 'constant'
            gives equal weight to all predictions while gaussian gives less weight to predictions on edges of windows.

    Note:
        must be channel first, support both 2D and 3D.
        input data must have batch dim.
        execute on 1 image/per inference, run a batch of window slices of 1 input image.
    """
    num_spatial_dims = len(inputs.shape) - 2
    assert len(roi_size) == num_spatial_dims, f"roi_size {roi_size} does not match input dims."
    assert overlap >= 0 and overlap < 1, "overlap must be >= 0 and < 1."

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    # TODO: Enable batch sizes > 1 in future
    if batch_size > 1:
        raise NotImplementedError

    original_image_size = [image_size[i] for i in range(num_spatial_dims)]
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = [i for k in range(len(inputs.shape) - 1, 1, -1) for i in (0, max(roi_size[k - 2] - inputs.shape[k], 0))]
    inputs = F.pad(inputs, pad=pad_size, mode="constant", value=0)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)

    slice_batches = []
    for slice_index in range(0, len(slices), sw_batch_size):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))
        input_slices = []
        for curr_index in slice_index_range:
            if num_spatial_dims == 3:
                slice_i, slice_j, slice_k = slices[curr_index]
                input_slices.append(inputs[0, :, slice_i, slice_j, slice_k])
            else:
                slice_i, slice_j = slices[curr_index]
                input_slices.append(inputs[0, :, slice_i, slice_j])
        slice_batches.append(torch.stack(input_slices))

    # Perform predictions
    output_rois = list()
    for data in tqdm(slice_batches, desc='Sliced batches'):
        seg_prob = predictor(data).float() #, device='cuda').float() # batched patch segmentation
        output_rois.append(seg_prob)

    # stitching output image
    output_classes = output_rois[0].shape[1]
    output_shape = [batch_size, output_classes] + list(image_size)

    # Create importance map
    importance_map = compute_importance_map(roi_size, mode=blend_mode, sigma_scale=0.125, device=inputs.device)

    # allocate memory to store the full output and the count for overlapping parts
    output_image = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)
    count_map = torch.zeros(output_shape, dtype=torch.float32, device=inputs.device)

    for window_id, slice_index in enumerate(range(0, len(slices), sw_batch_size)):
        slice_index_range = range(slice_index, min(slice_index + sw_batch_size, len(slices)))

        # store the result in the proper location of the full output. Apply weights from importance map.
        for curr_index in slice_index_range:
            if num_spatial_dims == 3:
                slice_i, slice_j, slice_k = slices[curr_index]
                output_image[0, :, slice_i, slice_j, slice_k] += (
                    importance_map * output_rois[window_id][curr_index - slice_index, :]
                )
                count_map[0, :, slice_i, slice_j, slice_k] += importance_map
            else:
                slice_i, slice_j = slices[curr_index]
                output_image[0, :, slice_i, slice_j] += (
                    importance_map * output_rois[window_id][curr_index - slice_index, :]
                )
                count_map[0, :, slice_i, slice_j] += importance_map

    # account for any overlapping sections
    output_image /= count_map

    if num_spatial_dims == 3:
        return output_image[..., : original_image_size[0], : original_image_size[1], : original_image_size[2]].half()
    return output_image[..., : original_image_size[0], : original_image_size[1]]  # 2D


def _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap):
    assert len(image_size) == num_spatial_dims, "image coord different from spatial dims."
    assert len(roi_size) == num_spatial_dims, "roi coord different from spatial dims."

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            # scan interval is (1-overlap)*roi_size
            scan_interval.append(int(roi_size[i] * (1 - overlap)))
    return tuple(scan_interval)