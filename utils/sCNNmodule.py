# import torch
import torch.nn as nn
# import numpy as np

# from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
# from torch.nn.modules.utils import _pair
from math import log2

class convRelu(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(convRelu, self).__init__()
        self.conv = nn.Conv2d(in_channel,out_channel,3,1,1)
        self.lrelu = nn.LeakyReLU()
    def forward(self, x):
        return self.lrelu(self.conv(x))
    
        
class UpSampleBlock(nn.Module):
    def __init__(self,input_channels,scale_factor=2):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(input_channels,input_channels*scale_factor**2,kernel_size=3,stride=1,padding=1)
        self.shuffler = nn.PixelShuffle(scale_factor)
        self.lrelu = nn.LeakyReLU()
    def forward(self,x):
        x = self.conv(x)
        x = self.shuffler(x)
        return self.lrelu(x)

class sCNN(nn.Module):
    '''1 -> featureSize -> ...depth times...  ->featureSize -> upsampler/wPixel Shuffler(scale):featureSize'''
    def __init__(self,inpChannel, featureSize, depth, scale):
        super(sCNN, self).__init__()
        layers = []
        layers.append(convRelu(inpChannel,featureSize))
        for _ in range(depth):
            layers.append(convRelu(featureSize,featureSize))
        for _ in range(int(log2(scale))):
            layers.append(UpSampleBlock(featureSize,2))
        self.network = nn.Sequential(*layers)
    def forward(self,x):
        return self.network(x)
    
class sCNN_Complete(nn.Module):
    '''1 -> featureSize -> ...depth times...  ->featureSize -> upsampler/wPixel Shuffler(scale):featureSize'''
    def __init__(self,inpChannel, featureSize, depth, scale):
        super(sCNN_Complete, self).__init__()
        layers = []
        layers.append(convRelu(inpChannel,featureSize))
        for _ in range(depth):
            layers.append(convRelu(featureSize,featureSize))
        for _ in range(int(log2(scale))):
            layers.append(UpSampleBlock(featureSize,2))
        layers.append(nn.Conv2d(featureSize,1,kernel_size=3,stride=1,padding=1))
        self.network = nn.Sequential(*layers)
    def forward(self,x):
        return self.network(x)