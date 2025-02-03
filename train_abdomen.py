import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd.functional import jacobian
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from torchvision.utils import flow_to_image
import os
from tqdm import tqdm,trange
import argparse
import time

from util_midl25 import GaussianSmoothing,DiVRoC
device = 'cuda'
H = W = D = 128

import monai
from monai.networks.nets.unet import UNet

import sys
sys.path.insert(0,'../point-transformer/')

from lib.pointops.functions import pointops

import nibabel as nib
class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
    def forward(self, x):
        x = torch.sigmoid(x/2)*2-1
        return x
smooth_c = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1),nn.AvgPool3d(3,stride=1,padding=1),nn.AvgPool3d(3,stride=1,padding=1),Sigmoid())
splat = DiVRoC().apply

def seg_to_pts_list(seg1,N):
    onehot = F.one_hot(seg1.cuda().long(),14).permute(3,0,1,2)
    grad = torch.stack(torch.gradient(onehot,dim=(1,2,3))).abs().sum(0)
    pts_idx = torch.nonzero(grad.view(14,-1))

    new_count = torch.ones(14).cuda().int()*N#(pts_idx[:,0].bincount().float().pow(.8)*2).int()
    old_count = pts_idx[:,0].bincount().int()
    pts = torch.empty(0,3).cuda()
    mesh1 = F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,192,160,256)).view(-1,3)
    for i in range(14):
        idx = pts_idx[pts_idx[:,0]==i,1]
        pts = torch.cat((pts,mesh1[idx]))
    old_count_c = old_count.cumsum(0).int()
    new_count_c = new_count.cumsum(0).int()
    idx_fps = torch.sort(pointops.furthestsampling(pts, old_count_c, new_count_c)).values  # (m)
    pts_new = mesh1[pts_idx[idx_fps,1]]
    pts_new_list = torch.split(pts_new,tuple(new_count))
    return pts_new_list,idx_fps


    
def dice_coeff(outputs, labels, max_label):
    dice = torch.FloatTensor(max_label-1).fill_(0)
    for label_num in range(1, max_label):
        iflat = (outputs==label_num).view(-1).float()
        tflat = (labels==label_num).view(-1).float()
        intersection = torch.mean(iflat * tflat)
        dice[label_num-1] = (2. * intersection) / (1e-8 + torch.mean(iflat) + torch.mean(tflat))
    return dice
def divroc(pts,field=None,val=None,shape=(1,1,H,W,D)):
    shape = (pts.shape[0],shape[1],shape[2],shape[3],shape[4])
    if(val is None):
        val = torch.ones_like(pts[...,:1]).transpose(2,1)
    else:
        shape = (pts.shape[0],val.shape[1],shape[2],shape[3],shape[4])

    if(field is not None):
        disp_est = F.grid_sample(field,pts,align_corners=False).permute(0,2,3,4,1)
        return smooth_c(splat(val,pts+(disp_est).view_as(pts),shape))
    else:
        return smooth_c(splat(val,pts,shape))
        
def divroc_add(pts,field):
    disp_est = F.grid_sample(field,pts,align_corners=False).permute(0,2,3,4,1)
    return pts+(disp_est).view_as(pts)

def disp_square(field):
    field2 = field/2**5
    B,_,H,W,D = field.shape
    grid1 = F.affine_grid(torch.eye(3,4).unsqueeze(0).to(device).repeat(B,1,1),(B,1,H,W,D),align_corners=False)
    for i in range(5):
        field2 = F.grid_sample(field2,field2.permute(0,2,3,4,1)+grid1,align_corners=False,padding_mode='border')+field2 #compose
    return field2

def compose(field,field1):
    B,_,H,W,D = field.shape
    grid1 = F.affine_grid(torch.eye(3,4).unsqueeze(0).to(device).repeat(B,1,1),(B,1,H,W,D),align_corners=False)
    field2 = F.grid_sample(field1.float(),field.permute(0,2,3,4,1)+grid1,align_corners=False,padding_mode='border')+field #compose
    return field2  

def divroc_sym_step(pts_fix1,val_fix1,pts_mov1,val_mov1,unet):
    kernel = 3; half_width = (kernel-1)//2
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
        

def divroc_asym_step(pts_fix1,val_fix1,pts_mov1,val_mov1,unet):
    kernel = 3; half_width = (kernel-1)//2
    avg5_ = nn.AvgPool3d(kernel,stride=2,padding=half_width)
    avg5 = nn.AvgPool3d(kernel,stride=1,padding=half_width)

    with torch.no_grad():
        fixed = divroc(pts_fix1,None,val_fix1).data
        moving = divroc(pts_mov1,None,val_mov1).data
    with torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output_fwd = torch.tanh(unet(torch.cat((fixed,moving),1)))*.25
        field_fwd = F.interpolate(avg5(avg5(avg5_(avg5_(output_fwd)))),size=(H,W,D),mode='trilinear').float()

    smooth_hr = disp_square(field_fwd)
    warped_mov = divroc(pts_mov1,smooth_hr,val_mov1)
    
    pts_mov1 = divroc_add(pts_mov1.data.clone(),disp_square(field_fwd)).data
    return fixed,moving,None,warped_mov,pts_fix1,pts_mov1,field_fwd,None
        

def main(args):

    print('steps',int(args.steps),'gpu',int(args.gpu),'symmetry',int(args.symmetry),'iterations',int(args.iterations))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(args.gpu))
    


    smooth_c = nn.Sequential(nn.AvgPool3d(3,stride=1,padding=1),nn.AvgPool3d(3,stride=1,padding=1),nn.AvgPool3d(3,stride=1,padding=1),Sigmoid())

    depth = 5; maxch = 64
    channels = (8,16,32,64,64,64)
    strides = (2,2,1,2,1)
    if(depth==4):
        strides = (2,2,1,2)
        channels = (8,16,32,64,64)
    if(depth==3):
        strides = (2,2,2)
        channels = (8,16,32,64)
    channels = tuple([min(c,maxch) for c in channels])
    
    #imgs = torch.zeros(30,192,160,256)
    segs = torch.zeros(30,192,160,256)

    for i in trange(30):
        segs[i] = torch.from_numpy(nib.load('AbdomenCTCT_00'+str(i+1).zfill(2)+'_0000.nii.gz').get_fdata()).float()
    
    
    N = 4096
    all_pts_lists = torch.zeros(30,14*N,1,1,3).cuda()
    all_values = torch.zeros(30,14,14*N,1,1).cuda()
    onehot = F.one_hot(torch.arange(14).unsqueeze(1).repeat(1,N),14).view(14*N,14).t().float().cuda()
    #all_values = torch.zeros(30,14,N,1,1).cuda()
    

    for i in trange(30):
        with torch.no_grad():
            pts_new_list1,idx_fps = seg_to_pts_list(segs[i],N)
            all_pts_lists[i] = torch.stack(pts_new_list1).cuda().view(14*N,1,1,3) 
            uu,ii,cc = torch.unique_consecutive(idx_fps, return_inverse=True, return_counts=True) 
            csum = cc.cumsum(0)-1
            values = torch.zeros(len(ii)).cuda().scatter_add_(0,csum,torch.ones(len(csum)).cuda())
            values1 = values.unsqueeze(0)*onehot
            #.view(1,14,N,1,1)#
            all_values[i] = values1.view(1,14,-1,1,1)
            
    unet1_ = []; unets = []
    for i in range(int(args.steps)):
        unet1_.append(UNet(spatial_dims=3,in_channels=28,out_channels=3,channels=channels,strides=strides).to(device))
        unets.append(torch.compile(unet1_[i]))
    #unet1[0].load_state_dict(torch.load('ff1_divroc_abdomen_ic_2nd.pth'))

    sym = int(args.symmetry)==1
    if(sym):
        ic_str = 'ic'
    else:
        ic_str = 'asym'
    pts = all_pts_lists.clone().data
    val = all_values.clone().data
    #def reg_net_divroc1(pts,val,unets,sym=True):
    optimizers = []
    for i in range(len(unets)):
        optimizers.append(torch.optim.Adam(unets[i].parameters(), lr=0.015))
    num_iters = int(args.iterations)
    #divroc_sym_step1 = torch.compile(divroc_sym_step)
    for iter in trange(num_iters):
        idx = torch.randperm(30)[:8].view(4,2)
        if(iter==num_iters-1):
            idx = torch.arange(8).view(4,2)
        pts_fix1 = pts[idx[:,0]].data.clone()
        val_fix1 = val[idx[:,0]].data.clone()
        pts_mov1 = pts[idx[:,1]].data.clone()
        val_mov1 = val[idx[:,1]].data.clone()
        for i in range(len(unets)):
            optimizers[i].zero_grad()
            if(sym):
                fixed,moving,warped_fix,warped_mov,pts_fix1,pts_mov1,field_fwd,field_bwd = divroc_sym_step(pts_fix1,\
                                        val_fix1,pts_mov1,val_mov1,unets[i])
            else:
                fixed,moving,warped_fix,warped_mov,pts_fix1,pts_mov1,field_fwd,field_bwd = divroc_asym_step(pts_fix1,\
                                        val_fix1,pts_mov1,val_mov1,unets[i])
                
            
            loss = nn.L1Loss()(fixed,warped_mov)
            if(sym):
                loss += nn.L1Loss()(moving,warped_fix)

            loss.backward()
            optimizers[i].step()
            if((iter==num_iters-1)&(i==0)):
                field_fwd_1st = field_fwd.data.clone()
                if(sym):
                    field_bwd_1st = field_bwd.data.clone()
        if(iter%500==499):
            if(int(args.steps)==2):
                torch.save([unets[0].state_dict(),unets[1].state_dict()],'models_abdomen/ff2_divroc_abdomen_'+ic_str+'.pth')
            else:
                torch.save(unets[0].state_dict(),'models_abdomen/ff1_divroc_abdomen_'+ic_str+'.pth')
               
           
    #return field_fwd_1st.data,field_bwd_1st.data,field_fwd.data,field_bwd.data
    
    field1_fwd = field_fwd_1st.data
    field2_fwd = field_fwd.data
    if(sym):
        field1_bwd = field_bwd_1st.data
        field2_bwd = field_bwd.data
    #reg_net_divroc1(all_pts_lists.clone().data,all_values.clone().data,unet1,sym=sym) 
    if(int(args.steps)==2):    
        with torch.no_grad():
            twostep_fwd = compose(compose(disp_square(field1_fwd/2),disp_square(field2_fwd)),disp_square(field1_fwd/2))
            if(sym):
                twostep_bwd = compose(compose(disp_square(field1_bwd/2),disp_square(field2_bwd)),disp_square(field1_bwd/2))
    else:
        twostep_fwd = disp_square(field1_fwd)
        if(sym):
            twostep_bwd = disp_square(field1_bwd)


    idx = torch.arange(8).view(4,2)
    dice = torch.zeros(3,4,13)
    #multichannel1 (without affine)
    warped_seg = F.grid_sample(segs[idx[:,0]].unsqueeze(1).float().cuda(),F.interpolate(twostep_fwd[:4],size=(192,160,256),mode='trilinear').permute(0,2,3,4,1)+F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,192,160,256)),mode='nearest')
    if(sym):
        warped_seg1_05 = F.grid_sample(segs[idx[:,1]].unsqueeze(1).float().cuda(),F.interpolate(disp_square(field1_bwd/4),size=(192,160,256),mode='trilinear').permute(0,2,3,4,1)+F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,192,160,256)),mode='nearest')
        warped_seg0_05 = F.grid_sample(segs[idx[:,0]].unsqueeze(1).float().cuda(),F.interpolate(disp_square(field1_fwd/4),size=(192,160,256),mode='trilinear').permute(0,2,3,4,1)+F.affine_grid(torch.eye(3,4).unsqueeze(0).cuda(),(1,1,192,160,256)),mode='nearest')
    for i in range(4):
        dice[0,i] = dice_coeff(segs[idx[i,0]].squeeze().cpu(),segs[idx[i,1]].squeeze().cpu(),14)
        if(sym):
            dice[2,i] = dice_coeff(warped_seg1_05[i].squeeze().cpu(),warped_seg0_05[i].squeeze().cpu(),14)
        dice[1,i] = dice_coeff(segs[idx[i,1]].squeeze().cpu(),warped_seg[i].squeeze().cpu(),14)
    print('dice_0.0',dice[0].mean(),'liver',dice[0,:,6].mean())
    print('dice_1.0',dice[1].mean(),'liver',dice[1,:,6].mean())
    print('dice_0.5',dice[2].mean(),'liver',dice[2,:,6].mean())
    if(int(args.steps)==2):
        torch.save(dice,'models_abdomen/ff2_divroc_abdomen_'+ic_str+'_dice.pth')
    else:
        torch.save(dice,'models_abdomen/ff1_divroc_abdomen_'+ic_str+'_dice.pth')

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--iterations")
    parser.add_argument("-g", "--gpu")
    parser.add_argument("-s", "--steps")
    parser.add_argument("-y", "--symmetry")

    args = parser.parse_args()
    main(args)
