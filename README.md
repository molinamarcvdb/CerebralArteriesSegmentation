
# Cerebral Arteries Segmentation

Cerebrovascular diseases are a leading cause of death and disability worldwide, with stroke being a significant contributor. Characterizing structural and physiological changes in vascular structures is crucial for identifying altered functioning and potential severe attacks. This also serves as a practical guideline to support neuro-interventionist preoperative planning.

Endovascular mechanical thrombectomy (MT) is the primary surgical treatment for patients with Acute Ischemic Stroke (AIS), often assisted by traditional imaging modalities used to detect large vessel occlusions (LVO). However, the overall tortuosity of the endovascular device pathway can influence the results of MT, making accurate evaluation crucial. Surgeons typically assess these pathways using Computed Tomography Angiographies (CTA) in coronal and sagittal planes. Unfortunately, 2D image analysis can introduce significant errors, leading to under or overestimation of angulation and tortuosity.

Recent approaches utilize semi-automatic methods to process CTAs and obtain 3D vascular segmentations, which are further analyzed to compute features at each point of the vasculature of interest. Despite these advancements, integrating these calculations into emergency room workflows requires fully-automatic configurations.

## Segmentation Framework

This repository contains a ready-to-use framework for the automatic segmentation of cerebral vasculature from CTA images. The main networks provided are:

- **UNETR** [1]: A transformer-based model tailored for 3D medical image segmentation.
- **Swin UNETR** [2]: A Swin Transformer model designed for semantic segmentation of brain tumors in MRI images, now adapted for cerebral artery segmentation.

### Key Features:
- **Modes**: The framework supports multiple modes, including Training, Testing, and Fitting Learning Rate (FittingLR).
- **Preprocessing Options**: Preprocessing of images, including cranium extraction, is supported.
- **Distributed Training**: The framework supports distributed training across multiple nodes.
- **Model Options**: Choose between different models such as UNETR, SwinUNETR, or nnUNet.
- **Custom Augmentations**: Integrates custom data augmentations and preprocessing pipelines.
- **Checkpointing**: Save and resume from checkpoints, supporting both AMP and standard training modes.
- **Loss Functions**: Multiple loss functions are available, including DiceCE, Tversky, and DiceFocal.

### Usage

To train a model, use the following command:

```bash
python main.py --mode Training --fold 0 --model_name unetr --max_epochs 200 --batch_size 2
```

Additional arguments can be customized according to your specific needs, such as learning rate, preprocessing options, model architecture, etc.

### References

[1] Hatamizadeh, Ali, et al. “UNETR: Transformers for 3D Medical Image Segmentation.” ArXiv:2103.10504 [Cs, Eess], 9 Oct. 2021, [arxiv.org/abs/2103.10504](https://arxiv.org/abs/2103.10504).

[2] Hatamizadeh, Ali, et al. “Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images.” ArXiv:2201.01266 [Cs, Eess], 4 Jan. 2022, [arxiv.org/abs/2201.01266](https://arxiv.org/abs/2201.01266).

