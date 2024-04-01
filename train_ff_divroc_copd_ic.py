import torch
import torch.nn as nn
import torch.nn.functional as F
import torchinfo
from torch.utils.checkpoint import checkpoint
import time
from tqdm import trange,tqdm
from torch.autograd import Function
from torch.autograd.functional import jacobian
import sys
import numpy as np
import os
import argparse
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ['CUDA_VISIBLE_DEVICES'] = str(0)

import monai
from monai.networks.nets.unet import UNet

from util_divroc3d import GaussianSmoothing,DiVRoC#Sigmoid5,
device = 'cuda'
class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()
    def forward(self, x):
        x = torch.sigmoid(x/2)*2-1
        return x
smooth_c = nn.Sequential(GaussianSmoothing(.7),Sigmoid(),)#
H=W=D=128
splat = DiVRoC().apply


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
    
    #kernel = 7; half_width = 3 
    #avg5_ = nn.AvgPool3d(7,stride=2,padding=3).cuda()
    #avg5 = nn.AvgPool3d(5,stride=(1,1,1),padding=2).cuda()
    
    kernel = 5; half_width = (kernel-1)//2
    avg5_ = nn.AvgPool3d(kernel,stride=2,padding=half_width)
    avg5 = nn.AvgPool3d(kernel,stride=1,padding=half_width)

    batch = 4
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
    kernel = 5; half_width = (kernel-1)//2
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
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train DiVRoC with for PVT/COPD.')
    
    parser.add_argument("-i", "--iterations")
    parser.add_argument("-g", "--gpu")
    parser.add_argument("-s", "--steps")
    parser.add_argument("-y", "--symmetry")

    args = parser.parse_args()
    
    print('steps',int(args.steps),'gpu',int(args.gpu),'symmetry',int(args.symmetry),'iterations',int(args.iterations))
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(int(args.gpu))
    
    train_data = torch.load('pvt_64k_train988_val.pth')
    pts_src_64k = train_data['pts_src_64k'].float().cuda()
    pts_tgt_64k = train_data['pts_tgt_64k'].float().cuda()
    val_src_64k = train_data['val_src_64k'].float().cuda()
    val_tgt_64k = train_data['val_tgt_64k'].float().cuda()

    
    H=W=D=128
    

    smooth_c = nn.Sequential(GaussianSmoothing(.7),Sigmoid(),)#
    splat = DiVRoC().apply

        
    dim0 = torch.tensor([255.5,255.5,240])#pvt_copd_dim[ii].cpu()#
    spacing = torch.ones(3)*1.25#pvt_copd_spacing[ii].cpu()#
    sym = int(args.symmetry)==1
    if(sym):
        ic_str = 'ic'
    else:
        ic_str = 'asym'
    batch = 4
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
    #initialise UNets
    unet1_ = []; unets = [];
    for i in range(int(args.steps)):
        unet1_.append(UNet(spatial_dims=3,in_channels=2,out_channels=3,channels=channels,strides=strides).cuda())
        unets.append(torch.compile(unet1_[i]))
        
    optimizers = []
    for i in range(len(unets)):
        optimizers.append(torch.optim.Adam(unets[i].parameters(), lr=0.015))
    
    ##DIVROC
    num_iters = int(args.iterations)
    for iter in trange(num_iters):
        idx = torch.randperm(988)[:batch]
        pts_fix1 = pts_tgt_64k[idx].float().view(4,-1,1,1,3).cuda().clone().detach()
        pts_mov1 = pts_src_64k[idx].float().view(4,-1,1,1,3).cuda().clone().detach()
        val_fix1 = val_tgt_64k[idx].float().view(4,1,-1,1,1).cuda().clone().detach()
        val_mov1 = val_src_64k[idx].float().view(4,1,-1,1,1).cuda().clone().detach()

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
            
            
               
        if(iter%500==499):
            if(int(args.steps)==2):
                torch.save([unets[0].state_dict(),unets[1].state_dict()],'models_pvt/ffv2_divroc_pvt_'+ic_str+'.pth')
            else:
                torch.save(unets[0].state_dict(),'models_pvt/ffv1_divroc_pvt_'+ic_str+'.pth')
               
     
    

