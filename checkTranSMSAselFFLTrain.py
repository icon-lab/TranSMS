# import libraries
import torch
from torch import nn
import numpy as np
# import matplotlib.pyplot as plt
# from skimage.transform import rescale, resize, downscale_local_mean
# from torch.autograd import Variable
# from scipy.io import loadmat, savemat
# import datetime
# from PIL import Image
# import glob
# import os
# import sys
# sys.path.append("/root/hostShare/") 
# sys.path.append("/root/hostShare/utils")
# from utils import *
# del sys.modules["utils.mpiSuperResUtils"]
#from utils.genericFnc import *
# del sys.modules["utils.cvtModels"]
import gc
import math

from utils.mpiSuperResUtils import *
from utils.trainModel import *
from utils.loadData import loadMtxFromOpenMPI
from utils.cvtModels import par_cvt_rdnDualNonSq
# from utils.smrNet import *
import argparse

# scale_factor, snrThreshold, useNoisyProjection, useGPUno, namePrefix, lrInpList, bsInpList, schedChoice, wdInpList

parser = argparse.ArgumentParser(description="TranSMS parameters")
parser.add_argument("--useGPUno", type=int, default=0,
                    help="Selected GPU no")
parser.add_argument("--wd", type=float,
                    default=0, help='weight decay')
parser.add_argument("--lr", type=float,
                    default=1e-4, help='path of log files')
parser.add_argument("--scale_factor", type=int,
                    default=4, help="Scale Factor for SR")
parser.add_argument("--snrThreshold", type=float, default=5,
                    help="SNR Threshold for System Matrix filtering")
parser.add_argument("--useNoisyProjection", type=int, default=1,
                    help="0: Don't use noise projection, 1: use noise projection")
parser.add_argument("--bs", type=int, default=64,
                    help="Batch Size")
parser.add_argument("--resultFolder", type=str, default=".",
                    help="Batch Size")
parser.add_argument("--n1", type=int, default=32,
                    help="System Matrix Dimension")
parser.add_argument("--n2", type=int, default=32,
                    help="System Matrix Dimension")
parser.add_argument("--trainFolder", type=str, default="./train",
                    help="System Matrix Dimension")
parser.add_argument("--testFolder", type=str, default="./val",
                    help="System Matrix Dimension")
opt = parser.parse_args()
print(opt)

#filePrefix = "/root/hostShare"
file_dir = opt.trainFolder #filePrefix + "/openMPI_mnp_half/train"
test_file_dir = opt.testFolder #filePrefix + "/openMPI_mnp_half/test"
directory = opt.resultFolder

def trainingFunction(opt, expName):
    n1, n2 = opt.n1, opt.n2

    # Model Parameters
    scale_factor, snrThreshold, useNoisyProjection = opt.scale_factor, opt.snrThreshold, opt.useNoisyProjection
    
    # Select GPU
    useGPUno = opt.useGPUno

    # Optimizer Parameters
    lr, bs, weight_decay = opt.lr, opt.bs, opt.wd

    namePrefix = expName + "_" + str(useGPUno) + "_"

    namePrefix += "model_CvTRDN_ext_"+ str(scale_factor) +  "x_using_thr_" + str(snrThreshold)

    if (useNoisyProjection):
        namePrefix += "NsProj"
    else:
        namePrefix += ""


    torch.cuda.set_device(useGPUno)

    trainLoader = loadMtxFromOpenMPI(file_dir, scale_factor, n1, n2, True, True)

    trainLoader.preprocessAndScaleSysMtx()
    # trainLoader.getBicubicInterpolation()

    # how_many_data_file = trainLoader.how_many_data_file

    testLoader = loadMtxFromOpenMPI(test_file_dir, scale_factor, n1, n2, True, True)

    testLoader.preprocessAndScaleSysMtx()
    testLoader.getBicubicInterpolation()

    test_how_many_data_file = testLoader.how_many_data_file

    totNoisePerElem = (2 * n1 * n2) ** (1/2)

    print("Total Noise Per System Matrix: ",totNoisePerElem)

    trainlr = torch.cat(tuple(trainLoader.Lr_list),0)
    trainhr = torch.cat(tuple(trainLoader.Hr_list),0)
    # trainBi = torch.cat(tuple(trainLoader.Bicubic_list), 0)
    max_arr = np.reshape(trainLoader.mxList,(-1,1))
    min_arr = np.reshape(trainLoader.mnList,(-1,1))
    nsStdEst_arr = torch.cat(tuple(trainLoader.nsKestirimi_list), 0)

#     snrEst_arr = np.concatenate((trainLoader.snrUnsc_list[0], trainLoader.snrUnsc_list[1]))

    train_hrDenorm = denormalize(trainhr, max_arr, min_arr, 0.7, 0.15)
    sigPow = torch.sum(torch.sum(torch.sum(torch.abs(train_hrDenorm)**2, axis = 3), axis = 2), axis = 1).squeeze().sqrt()

    # print(torch.norm(denormalize(trainBi, max_arr, min_arr, 0.7, 0.15) - train_hrDenorm) / torch.norm(train_hrDenorm))
    print("Train Samples: {0}".format(trainlr.shape[0]))

    snrEst_arr = sigPow / totNoisePerElem

    selectedElems = (snrEst_arr > snrThreshold).squeeze()

    trainlr = trainlr[selectedElems, :, :, :]
    trainhr = trainhr[selectedElems, :, :, :]
    # trainBi = trainBi[selectedElems, :, :, :]
    max_arr = max_arr[selectedElems, :]
    min_arr = min_arr[selectedElems, :]
    nsStdEst_arr = nsStdEst_arr[selectedElems, :]
    snrEst_arrCropped = snrEst_arr[selectedElems]

    train_hrDenorm = denormalize(trainhr, max_arr, min_arr, 0.7, 0.15)
    sigPows = torch.sum(torch.sum(torch.abs(train_hrDenorm)**2, axis = 3), axis = 2).squeeze().sqrt()
    # print(torch.norm(denormalize(trainBi, max_arr, min_arr, 0.7, 0.15) - train_hrDenorm) / torch.norm(train_hrDenorm))
    print("Train Samples after Filtering: {0}".format(trainlr.shape[0]))



    ordercopy = np.linspace(0,trainlr.shape[0]-1,trainlr.shape[0],dtype=int)
    np.random.seed(64)
    np.random.shuffle(ordercopy)
    trainhr = trainhr[ordercopy,:,:,:]
    trainlr = trainlr[ordercopy,:,:,:]
    max_arr = max_arr[ordercopy]
    min_arr = min_arr[ordercopy]
    nsStdEst_arr = nsStdEst_arr[ordercopy]
    copylr = torch.clone(trainlr)
    copyhr = torch.clone(trainhr)
    copymax = torch.clone(torch.Tensor(max_arr))
    copymin = torch.clone(torch.Tensor(min_arr))
    copyNsStd = torch.clone(nsStdEst_arr)
    #del trainHr_list, trainLr_list, trainBicubic_list

    test_lr = torch.cat(tuple(testLoader.Lr_list),0)
    test_hr = torch.cat(tuple(testLoader.Hr_list),0)
    test_Bi = torch.cat(tuple(testLoader.Bicubic_list), 0)
    test_nsStdEst_arr = torch.cat(tuple(testLoader.nsKestirimi_list), 0)
    test_max_arr = np.reshape(testLoader.mxList,(-1,1))
    test_min_arr = np.reshape(testLoader.mnList,(-1,1))
    test_snrEst_arr = np.concatenate((testLoader.snrUnsc_list[0], testLoader.snrUnsc_list[0]))

    test_hrDenorm = denormalize(test_hr, test_max_arr, test_min_arr, 0.7, 0.15)
    sigPowTest = torch.sum(torch.sum(torch.sum(torch.abs(test_hrDenorm)**2, axis = 3), axis = 2), axis = 1).squeeze().sqrt()

    test_snrEst_arr = sigPowTest / totNoisePerElem

    selectedElemsTest = (test_snrEst_arr > snrThreshold).squeeze()

    print("Bicubic Test nRMSE (No Filter): {0}".format(torch.norm(denormalize(test_Bi, test_max_arr, test_min_arr, 0.7, 0.15) - test_hrDenorm) / torch.norm(test_hrDenorm)))
    print("Test Samples (No Filter): {0}".format(test_lr.shape[0]))

    test_lr = test_lr[selectedElemsTest, :, :, :]
    test_hr = test_hr[selectedElemsTest, :, :, :]
    test_Bi = test_Bi[selectedElemsTest, :, :, :]
    test_max_arr = test_max_arr[selectedElemsTest, :]
    test_min_arr = test_min_arr[selectedElemsTest, :]
    test_nsStdEst_arr = test_nsStdEst_arr[selectedElemsTest, :]
    test_snrEst_arrCropped = test_snrEst_arr[selectedElemsTest]

    test_hrDenorm = denormalize(test_hr, test_max_arr, test_min_arr, 0.7, 0.15)
    sigPowTests = torch.sum(torch.sum(torch.sum(torch.abs(test_hrDenorm)**2, axis = 3), axis = 2), axis = 1).squeeze().sqrt()
    print("Bicubic nRMSE after Filtering: {0}".format(torch.norm(denormalize(test_Bi, test_max_arr, test_min_arr, 0.7, 0.15) - test_hrDenorm) / torch.norm(test_hrDenorm)))
    print("Test Samples after Filtering: {0}".format(test_lr.shape[0]))

    print("Min. nRMSE Due to Inherent Noise in Experimental Data: {0}".format(float((test_lr.shape[0]/2)**(1/2) * totNoisePerElem / torch.norm(test_hrDenorm))))

    test_BiNoOrder = test_Bi

    test_copylr = torch.clone(test_lr)
    test_copyhr = torch.clone(test_hr)
    test_copymax = torch.clone(torch.Tensor(test_max_arr))
    test_copymin = torch.clone(torch.Tensor(test_min_arr))


    # generate noisy LR & HR data
    SNRmtx = 0 #SNR for noise of matrix

    if SNRmtx > 0:
        nsPwr = 10**(-SNRmtx/20) # 20 dB SNR = 1/10
        nsTerm = 6e-11 * nsPwr * torch.randn_like(test_copylr)
        nsNrmlz = noisePowerNormalize(nsTerm, test_copymax, test_copymin, 0.7, 0.15)

        inpLR = test_copylr + nsNrmlz
        inpHR = test_copyhr + (scale_factor ** 2) * noisePowerNormalize(6e-11 * nsPwr * torch.randn_like(test_copyhr), test_copymax, test_copymin, 0.7, 0.15)
    else:
        nsPwr = 0 # 20 dB SNR = 1/10

        inpLR = test_copylr
        inpHR = test_copyhr

    y_bicub = test_BiNoOrder.cuda()

    testDS = myDataset(test_copyhr, inpLR, maxhr = test_copymax, minhr = test_copymin, batch_size = 1024, up = 0.7, down = 0.15)

    print(torch.norm(testDS.denormalize(y_bicub) - testDS.denormalize(testDS.HR)) / torch.norm(testDS.denormalize(testDS.HR)))

    epoch_nb = 500

    saveDict = dict()
    bicubNRMSE = float(torch.norm(testDS.denormalize(y_bicub) - testDS.denormalize(testDS.HR)) / torch.norm(testDS.denormalize(testDS.HR)))

    nsPwr = 0
    useAggr = 1

    if (scale_factor == 2):
        config={'inChannel': 2, 'outChannel': 2, 'initialConvFeatures': 24, 'scaleFactor': 2, 'rdn_nb_of_features': 24, 'rdn_nb_of_blocks': 4, 'rdn_layer_in_each_block': 5, 'rdn_growth_rate': 6, 'img_size1': n1 // 2, 'img_size2': n2 // 2, 'cvt_out_channels': 64, 'cvt_dim': 64, 'num_attention_heads': 8, 'convAfterConcatLayerFeatures': 48}
    elif (scale_factor == 4):
        config = {'inChannel': 2, 'outChannel': 2, 'initialConvFeatures': 24, 'scaleFactor': 4, 'rdn_nb_of_features': 24, 'rdn_nb_of_blocks': 4, 'rdn_layer_in_each_block': 8, 'rdn_growth_rate': 6, 'img_size1': n1 // 4, 'img_size2': n2 // 4, 'cvt_out_channels': 64, 'cvt_dim': 64, 'num_attention_heads': 8, 'convAfterConcatLayerFeatures': 48}
    elif (scale_factor == 8):
        config = {'inChannel': 2, 'outChannel': 2, 'initialConvFeatures': 64, 'scaleFactor': 8, 'rdn_nb_of_features': 24, 'rdn_nb_of_blocks': 4, 'rdn_layer_in_each_block': 9, 'rdn_growth_rate': 6, 'img_size1': n1 // 8, 'img_size2': n2 // 8, 'cvt_out_channels': 64, 'cvt_dim': 64, 'num_attention_heads': 8, 'convAfterConcatLayerFeatures': 48}

    model = par_cvt_rdnDualNonSq(config).cuda()

    print("# of parameters for model: ",sum(p.numel() for p in model.parameters() if p.requires_grad))
    print("Model Config: ",str(config))


    ordertrain = np.linspace(0,trainlr.shape[0]-1,trainlr.shape[0],dtype=int)
    trainhr = copyhr
    trainlr = copylr
    maxhr = copymax.cuda()
    minhr = copymin.cuda()

    if (useAggr):
        trainhrAggr = torch.cat((trainhr, torch.flip(trainhr, [2]), torch.flip(trainhr, [3]), torch.flip(trainhr, [2, 3])) , dim = 0)
        trainlrAggr = torch.cat((trainlr, torch.flip(trainlr, [2]), torch.flip(trainlr, [3]), torch.flip(trainlr, [2, 3])) , dim = 0)
        maxhrAggr = torch.cat( (maxhr, maxhr, maxhr, maxhr), dim = 0 )
        minhrAggr = torch.cat( (minhr, minhr, minhr, minhr), dim = 0 )

        ordertrainAggr = np.linspace(0,trainlrAggr.shape[0]-1,trainlrAggr.shape[0],dtype=int)

    batch_size = bs
    trainDS = myDataset(trainhr, trainlr, maxhr, minhr, batch_size, 0.7, 0.15, ordertrain)
    trainDS.shuffleAll()
    train_size = np.ceil(trainlr.shape[0]//batch_size) # from code
    epoch_nb = int(math.ceil(20e4 / train_size)) # from code

    if (useAggr):
        trainDS = myDataset(trainhrAggr, trainlrAggr, maxhrAggr, minhrAggr, batch_size, 0.7, 0.15, ordertrainAggr)
        trainDS.shuffleAll()
        train_size = np.ceil(trainlrAggr.shape[0]//batch_size) # from code
        epoch_nb = int(math.ceil(20e4 / train_size)) # from code
    
    print("Num Epochs:",epoch_nb)

    model = par_cvt_rdnDualNonSq(config).cuda()
    loss = nn.L1Loss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = epoch_nb // 5, gamma=0.5)
        
    modelSpecificInfo = str(model.__class__.__name__)+"lr_"+str(lr)+"wd_"+str(weight_decay)+"scl"+str(scale_factor)+"_bs"+str(batch_size)
    print("Experiment will be saved with: ",namePrefix)
    print("Model Specific Info: ", modelSpecificInfo)
    if (useNoisyProjection):
        nsStdEst_arr[:,0] = 1
        test_nsStdEst_arr[:,0] = 1

        nsPwrProjection = totNoisePerElem / scale_factor ** 2 #32 / scale_factor ** 3

        #training
        model, lossesArray, _ = trainMyModelNsProjectedListWithPower(model, epoch_nb+1, loss, optimizer, scheduler, trainDS, testDS, ordertrain, nsPwrProjection, batch_size, expName, 0, directory)

        model.eval()

        gc.collect()
        torch.cuda.empty_cache()

        with torch.no_grad():
            temp_test_nrmseDivider = float(torch.norm(testDS.denormalize(testDS.HR)))
            temp_test_nrmse, y_model_out = test_model_wbatch_DS(model, testDS)
            oldBs = testDS.bs
            testDS.bs = testDS.LR.shape[0]
            nsEpsTest = testDS.noisePowerNormalize(nsPwrProjection, 0).reshape(-1, 1)
            y_model_NsPrj = projectToNoiseLevel(y_model_out, testDS.LR, nsEpsTest, int(scale_factor), int(scale_factor))
            temp_test_nrmse_new = float(torch.norm(testDS.noisePowerDeNormalize(y_model_NsPrj - testDS.HR, 0)))
            testDS.bs = oldBs
        torch.cuda.empty_cache()

        temp_test_nrmse = temp_test_nrmse_new / temp_test_nrmseDivider
        print("nRMSE for HR measurement: {0}".format(float((test_copyhr.shape[0]/2 * 32*32)**(1/2) / temp_test_nrmseDivider)))

    else:
        #training
        model, lossesArray, _ = trainMyModel(model, epoch_nb+1, loss, optimizer, scheduler, trainDS, testDS, ordertrain, nsPwr, batch_size, expName, 0, directory)

        model.eval()

        gc.collect()
        torch.cuda.empty_cache()

        temp_test_nrmse, y_model_out = test_model_wbatch_DS(model, testDS)
        torch.cuda.empty_cache()

    print(temp_test_nrmse)
    modelSpecificInfo += "_err_" + str(temp_test_nrmse)
    torch.save(model.state_dict(), directory+"/"+namePrefix+modelSpecificInfo+".pt")

if __name__ == "__main__":
    trainingFunction(opt, expName = "")

