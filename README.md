# fastforward
anonymous MICCAI 2024 submission on forward splatting for 3D deformable medical image registration 

## Fast forward: Rephrasing 3D deformable image registration through density alignment and splatting

In this work, we propose to use a straightforward forward splatting technique based on differentiable rasterisation. Contrary to prior work, we rephrase the problem of deformable image registration as a density alignment of rasterised volumes based on intermediate point cloud representations that can be automatically obtained through e.g. geometric vessel filters or surface segmentations. Our experimental validation demonstrates state- of-the-art performance over a wide range of registration tasks including intra- and inter-patient alignment of thorax and abdomen.

The provide code reproduces the key elements of our method an demonstrates how to implement, train and evaluate this novel approach. Due to limitation of anonymous repositories we cannot yet upload trained models and pre-processed data, but will guide users through those steps.

### Features
- [x] Toy example (petal shapes) as jupyter notebook
- [x] Utility code to perform all relevant steps of our approach
- [x] Source code to train and evaluate main approach
- [x] Guidance on how to pre-process thorax and abdomen data
- [ ] Full processing code and pre-computed point clouds
- [ ] All trained models as pth
- [ ] More models and ablations

Our method relies on an inter-tweened use of the forward splatting operation that was first defined in the following ICCV 2023 paper  <[https://openaccess.thecvf.com/content/ICCV2023/papers/Heinrich_Chasing_Clouds_Differentiable_Volumetric_Rasterisation_of_Point_Clouds_as_a_ICCV_2023_paper.pdf](Chasing Cloud / PDF)>. 

