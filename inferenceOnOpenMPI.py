# import libraries
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat, savemat

import glob
import os
import sys

from utils.mpiSuperResUtils import *
from utils.trainModel import *
from utils.loadData import *
from utils.cvtModels import *
import gc
from utils.smrNet import *

from utils.vdsr import *
from utils.srcnnModels import *
from utils.srcnnModels import SRCNN2channels

import argparse

# scale_factor, snrThreshold, useNoisyProjection, useGPUno, namePrefix, lrInpList, bsInpList, schedChoice, wdInpList

parser = argparse.ArgumentParser(description="TranSMS parameters")
parser.add_argument("--useGPUno", type=int, default=3,
                    help="Selected GPU no")
parser.add_argument("--useNoisyProjection", type=int, default=1,
                    help="0: Don't use noise projection, 1: use noise projection")
parser.add_argument("--bs", type=int, default=1024,
                    help="Batch Size")
parser.add_argument("--n1", type=int, default=32,
                    help="System Matrix Dimension")
parser.add_argument("--n2", type=int, default=32,
                    help="System Matrix Dimension")
parser.add_argument("--modelFolder", type=str, default="./outs/",
                    help="System Matrix Dimension")
parser.add_argument("--saveOutFolder", type=str, default="./newerTrial/",
                    help="System Matrix Dimension")
parser.add_argument("--testFolder", type=str, default="./test/",
                    help="System Matrix Dimension")
parser.add_argument("--interpolationMatrixPath", type=str, default="interpolaters.mat",
                    help="System Matrix Dimension")
opt = parser.parse_args()
print(opt)


namePrefix = "test_"

# load interpolation matrix for SRCNN & VDSR
interpolaters = loadmat(opt.interpolationMatrixPath)

test_file_dir = opt.testFolder
readDir = opt.modelFolder
saveDir = opt.saveOutFolder


def interpolateWithMatrix(x, scale_factor):
    interpolater = interpolaters["interpolater{0}x".format(scale_factor)]
    if isinstance(x, torch.Tensor):
        interpolater = torch.tensor(interpolater).float()
        if x.is_cuda:
            interpolater = interpolater.cuda()
    shapeX = x.shape
    xFlattened = x.reshape(*shapeX[:-2],-1)
    yFlattened = xFlattened @ interpolater.float()

    n1 = x.shape[2] * scale_factor
    n2 = x.shape[3] * scale_factor

    return yFlattened.reshape(*shapeX[:-2],n1,n2)

def test_vdsrcnn(model, testDS, scale_factor, batchSize = 1024):
    with torch.no_grad():
        model.eval()
        x = interpolateWithMatrix(testDS.LR, scale_factor).cuda()
        y_ground = testDS.HR.cuda()
        vdsrOutNormalized = torch.zeros_like(y_ground[:,:,:,:])
        iii = 0
        while(iii<vdsrOutNormalized.shape[0]-(vdsrOutNormalized.shape[0]%batchSize)):
            vdsrOutNormalized[iii:iii+batchSize,:,:,:] = model(x[iii:iii+batchSize,:,:,:])
            iii += batchSize
        vdsrOutNormalized[iii:,:,:,:] = model(x[iii:,:,:,:])
        nrmse =  torch.norm(testDS.denormalize(vdsrOutNormalized)-testDS.denormalize(y_ground))/torch.norm(testDS.denormalize(y_ground))

    return nrmse, vdsrOutNormalized


def inferenceFunction(opt):
    n1, n2 = opt.n1, opt.n2

    # Model Parameters
    useNoisyProjection = opt.useNoisyProjection
    bs = opt.bs
    
    # Select GPU
    useGPUno = opt.useGPUno
    torch.cuda.set_device(useGPUno)

    # regular downsampling experiments
    for scale_factor in [2, 4, 8]:
        testLoader = loadMtxFromOpenMPI(test_file_dir, scale_factor, n1, n2, True, True)
        testLoader.preprocessAndScaleSysMtx()

        totNoisePerElem = (2 * n1 * n2) ** (1/2)

        print("Total Noise Per normalized SM row: ", totNoisePerElem)
        
        test_lr = torch.cat(tuple(testLoader.Lr_list),0)
        test_hr = torch.cat(tuple(testLoader.Hr_list),0)
        test_max_arr = np.reshape(testLoader.mxList,(-1,1))
        test_min_arr = np.reshape(testLoader.mnList,(-1,1))

        test_copylr = torch.clone(test_lr)
        test_copyhr = torch.clone(test_hr)
        test_copymax = torch.clone(torch.Tensor(test_max_arr))
        test_copymin = torch.clone(torch.Tensor(test_min_arr))

        testDS = myDataset(test_copyhr, test_copylr, maxhr = test_copymax, minhr = test_copymin, batch_size = bs, up = 0.7, down = 0.15)

        totNoisePerElem = (2 * n1 * n2) ** (1/2)

        for file in sorted(os.listdir(readDir)):
            with torch.no_grad():
                if (file[-3:] == ".pt"):
                    if ("{0}x".format(scale_factor) in file):
                        useNoisyProjection = 0

                        if ("CvTRDN" in file):
                            useNoisyProjection = 1
                            if (scale_factor == 2):
                                config={'inChannel': 2, 'outChannel': 2, 'initialConvFeatures': 24, 'scaleFactor': 2, 'rdn_nb_of_features': 24, 'rdn_nb_of_blocks': 4, 'rdn_layer_in_each_block': 5, 'rdn_growth_rate': 6, 'img_size1': n1 // 2, 'img_size2': n2 // 2, 'cvt_out_channels': 64, 'cvt_dim': 64, 'num_attention_heads': 8, 'convAfterConcatLayerFeatures': 48}
                            elif (scale_factor == 4):
                                config = {'inChannel': 2, 'outChannel': 2, 'initialConvFeatures': 24, 'scaleFactor': 4, 'rdn_nb_of_features': 24, 'rdn_nb_of_blocks': 4, 'rdn_layer_in_each_block': 8, 'rdn_growth_rate': 6, 'img_size1': n1 // 4, 'img_size2': n2 // 4, 'cvt_out_channels': 64, 'cvt_dim': 64, 'num_attention_heads': 8, 'convAfterConcatLayerFeatures': 48}
                            elif (scale_factor == 8):
                                config = {'inChannel': 2, 'outChannel': 2, 'initialConvFeatures': 64, 'scaleFactor': 8, 'rdn_nb_of_features': 24, 'rdn_nb_of_blocks': 4, 'rdn_layer_in_each_block': 9, 'rdn_growth_rate': 6, 'img_size1': n1 // 8, 'img_size2': n2 // 8, 'cvt_out_channels': 64, 'cvt_dim': 64, 'num_attention_heads': 8, 'convAfterConcatLayerFeatures': 48}

                            model = par_cvt_rdnDualNonSq(config).cuda()
                        elif ("SRCNN2channels" in file):
                            model = SRCNN2channels(mode='test').cuda() #Train için paddingsiz
                        elif ("vdsr2channel" in file):
                            model = vdsr2channels().cuda()
                        else:
                            continue
                        model.eval()
                        for para in model.parameters():
                            para.requires_grad = False
                        print("Processing: ",file)

                        model.load_state_dict(torch.load(readDir +file,map_location=torch.device('cuda:{0}'.format(useGPUno))))
                        nsPwrProjection = totNoisePerElem / scale_factor ** 2

                        gc.collect()
                        torch.cuda.empty_cache()
                        if (useNoisyProjection):

                            temp_test_nrmseDivider = float(torch.norm(testDS.denormalize(testDS.HR)))
                            temp_test_nrmse, y_model_out = test_model_wbatch_DS(model, testDS)
                            oldBs = testDS.bs
                            testDS.bs = testDS.LR.shape[0]
                            nsEpsTest = testDS.noisePowerNormalize(nsPwrProjection, 0).reshape(-1, 1)

                            y_model_NsPrj = torch.zeros_like(testDS.HR)
                            iii = 0
                            while(iii<y_model_NsPrj.shape[0]-(y_model_NsPrj.shape[0]%bs)):
                                y_model_NsPrj[iii:iii+bs] = projectToNoiseLevel(y_model_out[iii:iii+bs], testDS.LR[iii:iii+bs], nsEpsTest[iii:iii+bs], int(scale_factor), int(scale_factor))
                                iii += bs
                            y_model_NsPrj[iii:] = projectToNoiseLevel(y_model_out[iii:], testDS.LR[iii:], nsEpsTest[iii:], int(scale_factor), int(scale_factor))

                            temp_test_nrmse_new = float(torch.norm(testDS.noisePowerDeNormalize(y_model_NsPrj - testDS.HR, 0)))
                            testDS.bs = oldBs

                            temp_test_nrmse = temp_test_nrmse_new / temp_test_nrmseDivider
                            print("nRMSE for HR measurement: {0:.4f}".format(float((test_copyhr.shape[0]/2 * 32*32)**(1/2) / temp_test_nrmseDivider)))
                        else:
                            temp_test_nrmse, y_model_NsPrj = test_vdsrcnn(model, testDS, scale_factor, batchSize = bs)

                        torch.cuda.empty_cache()
                        gc.collect()
                        torch.cuda.empty_cache()

                        print("estimated nRMSE for all unfiltered rows: ", float(temp_test_nrmse))
                        savemat(saveDir + file[:-3].replace(":","_") +"_save_snr0.mat" ,{'y_model_NsPrj':testDS.denormalize(y_model_NsPrj).cpu().detach().numpy()})
    
    # 2d-SMRnet downsampling experiments
    for scale_factor in [2, 4, 8]:
        testLoader = loadMtxFromOpenMPI(test_file_dir, scale_factor, n1, n2, False, True)
        testLoader.preprocessAndScaleSysMtx()

        totNoisePerElem = (2 * n1 * n2) ** (1/2)
        
        test_lr = torch.cat(tuple(testLoader.Lr_list),0)
        test_hr = torch.cat(tuple(testLoader.Hr_list),0)
        test_max_arr = np.reshape(testLoader.mxList,(-1,1))
        test_min_arr = np.reshape(testLoader.mnList,(-1,1))

        test_copylr = torch.clone(test_lr)
        test_copyhr = torch.clone(test_hr)
        test_copymax = torch.clone(torch.Tensor(test_max_arr))
        test_copymin = torch.clone(torch.Tensor(test_min_arr))

        testDS = myDataset(test_copyhr, test_copylr, maxhr = test_copymax, minhr = test_copymin, batch_size = 1024, up = 0.7, down = 0.15)

        opt = {"which_model_G": "RRDB_net", 
        "norm_type": None, 
        "mode": "CNA", 
        "nf": 64,
        "nb": 9,
        "in_nc": 2,
        "out_nc": 2,
        "gc": 32,
        "group": 1,
        "scale": scale_factor,
        "niter": 20e4,
        "lr_G": 1e-4,
        "weight_decay_G": 1e-4,
        "beta1_G": 0.9,
        "beta2_G": 0.999,}

        for file in sorted(os.listdir(readDir)):
            with torch.no_grad():
                if (file[-3:] == ".pt"):
                    if ("{0}x".format(scale_factor) in file):

                        if ("RRDB" in file):
                            model = RRDBNet(in_nc=opt['in_nc'], out_nc=opt['out_nc'], nf=opt['nf'],
                            nb=opt['nb'], gc=opt['gc'], upscale=opt['scale'], norm_type=opt['norm_type'],
                            act_type='leakyrelu', mode=opt['mode'], upsample_mode='upconv').cuda()
                        else:
                            continue

                        for para in model.parameters():
                            para.requires_grad = False

                        print("Processing: ",file)

                        model.load_state_dict(torch.load(readDir+file,map_location=torch.device('cuda:{0}'.format(useGPUno))))
                        model.eval()

                        gc.collect()
                        torch.cuda.empty_cache()
                        temp_test_nrmse, y_model_NsPrj = test_model_wbatch_DS(model, testDS)
                        torch.cuda.empty_cache()

                        print("estimated nRMSE for all unfiltered rows: ", float(temp_test_nrmse))
                        savemat(saveDir+file[:-3].replace(":","_") +"_snr0.mat" ,{'y_model_NsPrj':testDS.denormalize(y_model_NsPrj).cpu().detach().numpy()})

if __name__ == "__main__":
    inferenceFunction(opt)

