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

Our method relies on an inter-tweened use of the forward splatting operation that was first defined in the following ICCV 2023 paper  <https://openaccess.thecvf.com/content/ICCV2023/papers/Heinrich_Chasing_Clouds_Differentiable_Volumetric_Rasterisation_of_Point_Clouds_as_a_ICCV_2023_paper.pdf>. It crucially extends this concept by defining the forward splatting for both dense and sparse image alignment problems (and not only point clouds) and placing it into a multi-step, inverse-consistent registration framework with U-Net backbones (rather than the complicated PointPWC). It comes with much improved versatility and higher accuracy for a wider range of tasks and sets new state-of-the-art performance in many benchmarks.

![Concept](fastforward_miccai_concept.png?raw=true "Concept")

Given two sparse point clouds $\mathcal{F}$ and $\mathcal{M}$ we rasterise 3D volumes $\mathbf{F}$ and $\mathbf{M}$ by **forward splatting** $\zeta$ as input to a registration U-Net (including a B-spline transform) which predicts a dense deformation $\phi$ that is sampled sparsely at the points of $\mathcal{F}$ (vector addition). An $L_1$ loss of a newly splatted  $\mathbf{F}_{\phi}=\zeta(\mathcal{F},\phi)$ and $\mathbf{M}$ is used to derive a well differentiable loss and **stable multi-stage model** without explicit regularisation.




