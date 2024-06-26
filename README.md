# Cerebral Arteries Segmentation

Cerebrovascular diseases are a leading cause of death and disability worldwide with stroke being
increasingly a major contributor. Characterizing structural and physiological changes in vascular
structure can identify altered functioning and potential severe attacks, as well as serve as a 
practical guideline supporting the neuro-interventionist preoperative planning.

Endovascular mechanical thrombectomy (MT) is the main surgical treatment for patients with
Acute Ischemic Stroke (AIS) assisted by traditional imaging modalities used to detect large
vessel occlusions (LVO). Problematically, the overall tortuosity of the endovascular device
pathway influence results of MT, hence surgeons evaluate it by using Computed Tomography
Angiographies (CTA) coronal and sagittal planes. However, this can lead to clinical judgment
errors since 2D image analysis introduces nonnegligible errors under or overestimating
angulation and tortuosity. Recently developed approaches use semi-automatic methods to
process CTAs to obtain vascular 3D segmentations which are then further processed to
compute features at each point of the vasculature of interest. Despite this fact, to enable its
proper integration in the emergency room there is the need to leverage the calculation of the
ideal access pathways to a fully-automatic configuration. 

The most challenging task in this regard is the CTA automatic segmentation of the vasculature of 
interest, here a fully autoamted method is presented. Concretely, this repository contains a 
ready to use framework where the main network that one can use is UNETR [1] and since the last
edition (not optimized) SwinUNTER [2].


[1] Hatamizadeh, Ali, et al. “UNETR: Transformers for 3D Medical Image Segmentation.” ArXiv:2103.10504 [Cs, Eess], 
9 Oct. 2021, arxiv.org/abs/2103.10504.

[2] Hatamizadeh, Ali, et al. “Swin UNETR: Swin Transformers for Semantic Segmentation of Brain Tumors in MRI Images.” ArXiv:2201.01266 [Cs, Eess], 4 Jan. 2022, arxiv.org/abs/2201.01266. Accessed 9 June 2022.
