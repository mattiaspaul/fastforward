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

Within our utility code the functions and derivatives for forward splatting are provided that yield the 3D volumetric rasterisation of a point cloud as follows:
```
splat = DiVRoC().apply    
smooth_c = nn.Sequential(GaussianSmoothing(.7),nn.Sigmoid())
target = smooth_c(splat(val_target,pts_target,(1,1,160,160,160))
```
potentially using unit weighted points with ``val_target = torch.ones_like(pts_target[...,:1]).transpose(2,1)``. We provide the short-hand ``target = divroc(pts_target)`` for this step and enable a warping operation as ``source = divroc(pts_source,smooth_flow)``. Within this operation, the dense displacement field ``smooth_flow`` is first sparsely sampled at ``pts_source`` and the vectors added to those points before performing the splatting step.

A straightforward implementation of an iterative (highly performant) alignment of two volumes represented as point clouds will look like follows:
```
bspline_flow = torch.zeros(1,3,128,128,128).cuda().requires_grad_()
optimizer = torch.optim.Adam([bspline_flow], lr=0.015)
kernel = 7; half_width = (kernel-1)//2
avg5 = nn.AvgPool3d((kernel,kernel,kernel),stride=(1,1,1),padding=(half_width,half_width,half_width)).cuda()
with torch.no_grad():
    target = divroc(pts_target)
for iter in range(iters):
    optimizer.zero_grad()
    smooth_flow = (avg5(avg5(bspline_flow)))
    source = divroc(pts_source,smooth_flow)
    loss = nn.L1Loss()(target,source)
    loss.backward()
    optimizer.step()
```
By initialising bspline_flow with an affine transform that aligns the clouds based on mean and standard deviation: ``x = torch.cat((torch.diag(pts_target.squeeze().std(0)/pts_source.squeeze().std(0)),(pts_target.squeeze().mean(0)-pts_source.squeeze().mean(0)).view(3,1)),1).cuda()``, we reach new SOTA performance for point cloud registration with a TRE of 2.00mm on the challenging PVT-COPD dataset.

Yet, further improvements are possible when training a (multi-step) registration U-Net - which takes a channel-concatenation of the two rasterised point clouds as input and predicts the transform $\phi$, ideally in an inverse-consistent way. Given randomly initialised U-Nets, e.g. 
```
from monai.networks.nets.unet import UNet
unet = UNet(spatial_dims=3,in_channels=2,out_channels=3,channels=(8,16,32,64,64,64),strides=(2,2,1,2,1)).cuda()
```
we define one symmetric registration step with forward splatting as:
```
def divroc_sym_step(pts_fix1,val_fix1,pts_mov1,val_mov1,unet):
    
    kernel = 5; half_width = (kernel-1)//2
    avg5_ = nn.AvgPool3d(kernel,stride=2,padding=half_width)
    avg5 = nn.AvgPool3d(kernel,stride=1,padding=half_width)

    with torch.no_grad():
        fixed = divroc(pts_fix1,None,val_fix1).data
        moving = divroc(pts_mov1,None,val_mov1).data
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output_fwd = torch.tanh(unet(torch.cat((fixed,moving),1)))*.25
        output_bwd = torch.tanh(unet(torch.cat((moving,fixed),1)))*.25
        field_fwd = F.interpolate(avg5(avg5(avg5_(avg5_(output_fwd-output_bwd)))),size=(H,W,D),mode='trilinear').float()
        field_bwd = F.interpolate(avg5(avg5(avg5_(avg5_(output_bwd-output_fwd)))),size=(H,W,D),mode='trilinear').float()

    smooth_hr = disp_square(field_fwd)
    warped_mov = divroc(pts_mov1,smooth_hr,val_mov1)
    
    smooth_hr = disp_square(field_bwd)
    warped_fix = divroc(pts_fix1,smooth_hr,val_fix1)
    
    pts_mov1 = divroc_add(pts_mov1.data.clone(),disp_square(field_fwd/2)).data
    pts_fix1 = divroc_add(pts_fix1.data.clone(),disp_square(field_bwd/2)).data
    return fixed,moving,warped_fix,warped_mov,pts_fix1,pts_mov1,field_fwd,field_bwd
```
Note that the expected displacement range for each step is limited to 0.25 (using normalised pytorch coordinates) and a B-spline transform is hard coded after the U-Net. We employ scaling-and-squaring using ``disp_square`` to ensure diffeomorphic transforms. Following the excellent work of Hastings Greer <https://doi.org/10.1007/978-3-031-43999-5_65> we find a mid-point transformation. Note that the returned point clouds (pts_fix1, pts_mov1) are already transformed to be close to one another. Once a second transformation has been estimated the inverse-consistent composition is performed as follows:
```
twostep_fwd = compose(compose(disp_square(field1_fwd/2),disp_square(field2_fwd)),disp_square(field1_fwd/2))
twostep_bwd = compose(compose(disp_square(field1_bwd/2),disp_square(field2_bwd)),disp_square(field1_bwd/2))
```

Our experimental results demonstrate a great benefit of using test-time-adaptation (TTA), which means we fine-tune the network weights for a few iterations (20-50) given the test data (using automatically created semantic point clouds). While the effect is similar to instance optimisation it is less complex because we are simply optimising the same loss function as done during training and it is still very fast.







