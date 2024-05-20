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

H=W=D=128

class GaussianSmoothing(nn.Module):
    def __init__(self, sigma):
        super(GaussianSmoothing, self).__init__()
        
        sigma = torch.tensor([sigma])
        N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1
    
        weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N), 2) / (2 * torch.pow(sigma, 2)))
        weight /= weight.sum()
        
        self.weight = weight
        
    def forward(self, x):
        device = x.device
        
        x = filter1D(x, self.weight.to(device), 0)
        x = filter1D(x, self.weight.to(device), 1)
        x = filter1D(x, self.weight.to(device), 2)
        
        return x
class DiVRoC(Function):
    @staticmethod
    def forward(ctx, input, grid, shape):
        device = input.device
        dtype = input.dtype
        
        output = -jacobian(lambda x: (F.grid_sample(x, grid,align_corners=False) - input).pow(2).mul(0.5).sum(), torch.zeros(shape, dtype=dtype, device=device))
        
        ctx.save_for_backward(input, grid, output)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, grid, output = ctx.saved_tensors
        
        B, C = input.shape[:2]
        input_dims = input.shape[2:]
        output_dims = grad_output.shape[2:]
    
        y = jacobian(lambda x: F.grid_sample(grad_output.unsqueeze(2).view(B*C, 1, *output_dims), x,align_corners=False).mean(), grid.unsqueeze(1).repeat(1, C, *([1]*(len(input_dims)+1))).view(B*C, *input_dims, len(input_dims))).view(B, C, *input_dims, len(input_dims))
        
        grad_grid = (input.numel()*input.unsqueeze(-1)*y).sum(1)
        
        grad_input = F.grid_sample(grad_output, grid,align_corners=False)
        
        return grad_input, grad_grid, None

splat = DiVRoC().apply    
smooth_c = nn.Sequential(GaussianSmoothing(.7),nn.Sigmoid())

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

def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]
    
    padding = torch.zeros(6,)
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N//2
    padding = padding.long().tolist()
    
    view = torch.ones(5,)
    view[dim + 2] = -1
    view = view.long().tolist()
    
    return F.conv3d(F.pad(img.view(B*C, 1, D, H, W), padding, mode=padding_mode), weight.view(view)).view(B, C, D, H, W)

def smooth(img, sigma):
    device = img.device
    
    sigma = torch.tensor([sigma]).to(device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1
    
    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N).to(device), 2) / (2 * torch.pow(sigma, 2)))
    weight /= weight.sum()
    
    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)
    
    return img

    
class Sigmoid5(nn.Module):
    def __init__(self):
        super(Sigmoid5, self).__init__()
    def forward(self, x):
        x = torch.tanh(x/5)*5
        return x
