from PIL import Image
import torch
import numpy as np
#import matplotlib.pyplot as plt

def interpImage(x, n1, n2):
    return np.array(Image.fromarray(x).resize((n2, n1),Image.BICUBIC).getdata()).reshape(n1, n2)

def downsampleImage(x, dx, dy): #Box downsampling
    y = torch.zeros(x.shape[0], x.shape[1]//dx, x.shape[2]//dy)
    for ii in range(dx):
        for jj in range(dy):
            y += x[:, ii::dx, jj::dy]
    return y

def downsampleImageNP(x, dx, dy): #Box downsampling
    y = np.zeros((x.shape[0], x.shape[1]//dx, x.shape[2]//dy), dtype=x.dtype)
    for ii in range(dx):
        for jj in range(dy):
            y += x[:, ii::dx, jj::dy]
    return y

def denormalize(x,max,min,up,down):
    if (len(x.shape) - len(max.shape)) == 2:
        return ((x-down)/up)*(max[:,None,None]-min[:,None,None])+min[:,None,None]
    else:
        return ((x-down)/up)*(max[:,None,None,None]-min[:,None,None,None])+min[:,None,None,None]
    
def normalize(x,max,min,up,down):
    if (len(x.shape) - len(max.shape)) == 2:
        return (x - min[:,None,None])/(max[:,None,None]-min[:,None,None])*up + down
    else:
        return (x - min[:,None,None,None])/(max[:,None,None,None]-min[:,None,None,None])*up + down
    
def noisePowerNormalize(x,max,min,up,down):
    if (len(x.shape) - len(max.shape)) == 2:
        return (x)/(max[:,None,None]-min[:,None,None])*up
    else:
        return (x)/(max[:,None,None,None]-min[:,None,None,None])*up
