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

class Sigmoid5(nn.Module):
    def __init__(self):
        super(Sigmoid5, self).__init__()
    def forward(self, x):
        x = torch.tanh(x/5)*5
        return x