from utils.sCNNmodule import *
from utils.cvtModule import *
from utils.rdnModule import *
import torch
import torch.nn as nn
from math import log2
    
class par_cvt_rdn(nn.Module):
    """config = {
    'initialConvFeatures' : 32, #### free param
    'scaleFactor':scale_factor, 
    'rdn_nb_of_features' : 24,
    'rdn_nb_of_blocks' : 4,
    'rdn_layer_in_each_block' : 5, #### untitled.png
    'rdn_growth_rate' : 6,
    'img_size' : 8, #e.g 8 or (6,8) input image size ####
    "cvt_out_channels": 32,#C1 ####
    "cvt_dim" : 32
    "convAfterConcatLayerFeatures" : 32 #C3 ###
    }"""
    def __init__(self, config):
        super(par_cvt_rdn, self).__init__()
        self.initialConv = nn.Conv2d(1,config['initialConvFeatures'],3,1,1)
        self.rdn = rdn1x(input_channels = config['initialConvFeatures'], 
                         nb_of_features = config['rdn_nb_of_features'], 
                         nb_of_blocks = config['rdn_nb_of_blocks'],
                        layer_in_each_block = config["rdn_layer_in_each_block"],
                        growth_rate = config["rdn_growth_rate"])
        self.transformer = CvT(image_size = config["img_size"], in_channels = config['initialConvFeatures'], out_channels = config["cvt_out_channels"],dim =config["cvt_dim"] )
        self.convAfterConcat = nn.Conv2d(config['rdn_nb_of_features']+config['cvt_out_channels'],
                                      config["convAfterConcatLayerFeatures"],3,1,1)
        upSamplersList = []
        for _ in range(int(log2(config['scaleFactor']))):
            upSamplersList.append(UpSampleBlock(config["convAfterConcatLayerFeatures"],2))
        self.upSampler = nn.Sequential(*upSamplersList)
        self.lastConv = nn.Conv2d(config['convAfterConcatLayerFeatures'],1,3,1,1)
    def forward(self,x):
        x = self.initialConv(x)
        rdnSkip = self.rdn(x)
        x = self.transformer(x)
        x = torch.cat([x, rdnSkip], dim=1)
        x = self.convAfterConcat(x)
        x = self.upSampler(x)
        x = self.lastConv(x)
        return x

class par_cvt_rdnDual(nn.Module):
    """config = {
    'initialConvFeatures' : 32, #### free param
    'scaleFactor':scale_factor, 
    'rdn_nb_of_features' : 24,
    'rdn_nb_of_blocks' : 4,
    'rdn_layer_in_each_block' : 5, #### untitled.png
    'rdn_growth_rate' : 6,
    'img_size' : 8, #e.g 8 or (6,8) input image size ####
    "cvt_out_channels": 32,#C1 ####
    "cvt_dim" : 32
    "convAfterConcatLayerFeatures" : 32 #C3 ###
    }"""
    def __init__(self, config):
        super(par_cvt_rdnDual, self).__init__()
        self.initialConv = nn.Conv2d(config['inChannel'],config['initialConvFeatures'],3,1,1)
        self.rdn = rdn1x(input_channels = config['initialConvFeatures'], 
                         nb_of_features = config['rdn_nb_of_features'], 
                         nb_of_blocks = config['rdn_nb_of_blocks'],
                        layer_in_each_block = config["rdn_layer_in_each_block"],
                        growth_rate = config["rdn_growth_rate"])
        self.transformer = CvT(image_size = config["img_size"], in_channels = config['initialConvFeatures'], out_channels = config["cvt_out_channels"],dim =config["cvt_dim"] )
        self.convAfterConcat = nn.Conv2d(config['rdn_nb_of_features']+config['cvt_out_channels'],
                                      config["convAfterConcatLayerFeatures"],3,1,1)
        upSamplersList = []
        for _ in range(int(log2(config['scaleFactor']))):
            upSamplersList.append(UpSampleBlock(config["convAfterConcatLayerFeatures"],2))
        self.upSampler = nn.Sequential(*upSamplersList)
        self.lastConv = nn.Conv2d(config['convAfterConcatLayerFeatures'],config['outChannel'],3,1,1)
    def forward(self,x):
        x = self.initialConv(x)
        rdnSkip = self.rdn(x)
        x = self.transformer(x)
        x = torch.cat([x, rdnSkip], dim=1)
        x = self.convAfterConcat(x)
        x = self.upSampler(x)
        x = self.lastConv(x)
        return x


class par_cvt_rdnDualNonSq(nn.Module):
    """config = {
    'initialConvFeatures' : 32, #### free param
    'scaleFactor':scale_factor, 
    'rdn_nb_of_features' : 24,
    'rdn_nb_of_blocks' : 4,
    'rdn_layer_in_each_block' : 5, #### untitled.png
    'rdn_growth_rate' : 6,
    'img_size1' : 8, #e.g 8 or (6,8) input image size ####
    'img_size2' : 8, #e.g 8 or (6,8) input image size ####
    "cvt_out_channels": 32,#C1 ####
    "cvt_dim" : 32
    "convAfterConcatLayerFeatures" : 32 #C3 ###
    }"""
    def __init__(self, config):
        super(par_cvt_rdnDualNonSq, self).__init__()
        self.initialConv = nn.Conv2d(config['inChannel'],config['initialConvFeatures'],3,1,1)
        self.rdn = rdn1x(input_channels = config['initialConvFeatures'], 
                         nb_of_features = config['rdn_nb_of_features'], 
                         nb_of_blocks = config['rdn_nb_of_blocks'],
                        layer_in_each_block = config["rdn_layer_in_each_block"],
                        growth_rate = config["rdn_growth_rate"])
        self.transformer = CvTNonSquare(image_size1 = config["img_size1"], image_size2 = config["img_size2"], in_channels = config['initialConvFeatures'], out_channels = config["cvt_out_channels"],dim =config["cvt_dim"] )
        self.convAfterConcat = nn.Conv2d(config['rdn_nb_of_features']+config['cvt_out_channels'],
                                      config["convAfterConcatLayerFeatures"],3,1,1)
        upSamplersList = []
        for _ in range(int(log2(config['scaleFactor']))):
            upSamplersList.append(UpSampleBlock(config["convAfterConcatLayerFeatures"],2))
        self.upSampler = nn.Sequential(*upSamplersList)
        self.lastConv = nn.Conv2d(config['convAfterConcatLayerFeatures'],config['outChannel'],3,1,1)
    def forward(self,x):
        x = self.initialConv(x)
        rdnSkip = self.rdn(x)
        x = self.transformer(x)
        x = torch.cat([x, rdnSkip], dim=1)
        x = self.convAfterConcat(x)
        x = self.upSampler(x)
        x = self.lastConv(x)
        return x
    
    
class par_cvt_only(nn.Module):
    """config = {
    'initialConvFeatures' : 32, #### free param
    'scaleFactor':scale_factor, 
    'rdn_nb_of_features' : 24,
    'rdn_nb_of_blocks' : 4,
    'rdn_layer_in_each_block' : 5, #### untitled.png
    'rdn_growth_rate' : 6,
    'img_size' : 8, #e.g 8 or (6,8) input image size ####
    "cvt_out_channels": 32,#C1 ####
    "cvt_dim" : 32
    "convAfterConcatLayerFeatures" : 32 #C3 ###
    }"""
    def __init__(self, config):
        super(par_cvt_only, self).__init__()
        self.initialConv = nn.Conv2d(1,config['initialConvFeatures'],3,1,1)
#         self.rdn = rdn1x(input_channels = config['initialConvFeatures'], 
#                          nb_of_features = config['rdn_nb_of_features'], 
#                          nb_of_blocks = config['rdn_nb_of_blocks'],
#                         layer_in_each_block = config["rdn_layer_in_each_block"],
#                         growth_rate = config["rdn_growth_rate"])
        self.transformer = CvT(image_size = config["img_size"], in_channels = config['initialConvFeatures'], out_channels = config["cvt_out_channels"],dim =config["cvt_dim"] )
        self.convAfterConcat = nn.Conv2d(config['cvt_out_channels'],
                                      config["convAfterConcatLayerFeatures"],3,1,1)
        upSamplersList = []
        for _ in range(int(log2(config['scaleFactor']))):
            upSamplersList.append(UpSampleBlock(config["convAfterConcatLayerFeatures"],2))
        self.upSampler = nn.Sequential(*upSamplersList)
        self.lastConv = nn.Conv2d(config['convAfterConcatLayerFeatures'],1,3,1,1)
    def forward(self,x):
        x = self.initialConv(x)
#         rdnSkip = self.rdn(x)
        x = self.transformer(x)
#         x = torch.cat([x, rdnSkip], dim=1)
        x = self.convAfterConcat(x)
        x = self.upSampler(x)
        x = self.lastConv(x)
        return x