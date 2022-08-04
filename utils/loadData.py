# import libraries
import torch
import numpy as np
from scipy.io import loadmat
import os
from utils.mpiSuperResUtils import downsampleImageNP, interpImage, normalize


class loadMtxFromOpenMPI:
    def __init__(self, file_dir, scale_factor, n1, n2, useBoxDownsampling = True, useTwoChannelInput = False):
        self.file_dir = file_dir
        self.scale_factor = scale_factor
        self.n1 = n1
        self.n2 = n2
        
        self.HrUnsc_list = list()
        self.LrUnsc_list = list()
        self.nsKestirimiUnsc_list = list()
        self.snrUnsc_list = list()
        self.BiCubicUnsc_list = list()
        LrUnsc_list = list()
        BiCubicUnsc_list = list()
        
        self.useTwoChannelInput = useTwoChannelInput
        self.n1Down = self.n1 // scale_factor
        self.n2Down = self.n2 // scale_factor
        
                    
        for file in sorted(os.listdir(file_dir+"/")):
            if (file[-4:] == ".mat"):
                readImg = loadmat(file_dir+"/"+file)
                hrMtx = readImg['myMtx']
                nsMtx1 = readImg['estNsKestirimi']
                snrMtx = readImg['snrList']
                
                numFreqs = nsMtx1.shape[0]
                
                numZslice = hrMtx.shape[0] // numFreqs
                
                hrMtx = hrMtx.reshape(numFreqs, numZslice, n1, n2)
                nsMtx1 = nsMtx1.repeat(numZslice,axis = 1)
                snrMtx = snrMtx.repeat(numZslice,axis = 1)
        
                self.HrUnsc_list.append(hrMtx.reshape(numZslice * numFreqs, n1, n2)) # Bütün freq'ler
            
                if (useBoxDownsampling):
                    lrMtx = downsampleImageNP(hrMtx.reshape(-1, n1, n2), scale_factor, scale_factor).reshape(numZslice * numFreqs, self.n1Down, self.n2Down) / scale_factor ** 2
                else:
                    lrMtx = (hrMtx.reshape(-1, n1, n2)[:,0::scale_factor, 0::scale_factor]).reshape(numZslice * numFreqs, self.n1Down, self.n2Down)
                self.LrUnsc_list.append(lrMtx)
                
                self.nsKestirimiUnsc_list.append(nsMtx1.reshape(-1))
                self.snrUnsc_list.append(snrMtx.reshape(-1))
                
        self.newSize = (-1, self.n1, self.n2)
        self.newSizeLR = (-1, self.n1Down, self.n2Down)
        self.how_many_data_file = len(self.HrUnsc_list)
        
        self.Hr_list = list()
        self.Lr_list = list()
        self.nsKestirimi_list = list()
        self.Bicubic_list = list()
    
    
    def preprocessAndScaleSysMtx(self):
        self.mxList = np.zeros((0))
        self.mnList = np.zeros((0))

        for myReadImNum in range(self.how_many_data_file):
            HRIm = self.HrUnsc_list[myReadImNum]
            LRIm = self.LrUnsc_list[myReadImNum]
            nsIm = self.nsKestirimiUnsc_list[myReadImNum]
            
            if (self.useTwoChannelInput):
                _HRList = np.concatenate((HRIm.real.reshape(-1, 1, self.n1, self.n2), HRIm.imag.reshape(-1, 1, self.n1, self.n2)), axis = 1)
                _LRList = np.concatenate((LRIm.real.reshape(-1, 1, self.n1Down, self.n2Down), LRIm.imag.reshape(-1, 1, self.n1Down, self.n2Down)), axis = 1)
                numCh = 2
            else:
                _HRList = np.concatenate((HRIm.real.reshape(self.newSize), HRIm.imag.reshape(self.newSize)))
                _LRList = np.concatenate((LRIm.real.reshape(self.newSizeLR), LRIm.imag.reshape(self.newSizeLR)))
                numCh = 1

            mx = _LRList.reshape(-1, numCh * self.n1Down*self.n2Down).max(axis = 1)
            mn = _LRList.reshape(-1, numCh * self.n1Down*self.n2Down).min(axis = 1)

            _HRListSc = normalize(_HRList,mx,mn,0.7,0.15)
            _LRListSc = normalize(_LRList,mx,mn,0.7,0.15)
            _nsKestirimiSc = nsIm.repeat(2 // numCh, axis = 0) / (mx - mn) * 0.7
            
            self.mxList = np.concatenate((self.mxList, mx), axis = 0)
            self.mnList = np.concatenate((self.mnList, mn), axis = 0)
            
            if (self.useTwoChannelInput):
                self.Hr_list.append((torch.as_tensor(_HRListSc,dtype=torch.double)).float())
                self.Lr_list.append((torch.as_tensor(_LRListSc,dtype=torch.double)).float())
            else:
                self.Hr_list.append(torch.unsqueeze(torch.as_tensor(_HRListSc,dtype=torch.double), 1).float())
                self.Lr_list.append(torch.unsqueeze(torch.as_tensor(_LRListSc,dtype=torch.double), 1).float())
            
            self.nsKestirimi_list.append(torch.unsqueeze(torch.as_tensor(_nsKestirimiSc,dtype=torch.double), 1).float())
                
    def getBicubicInterpolation(self):
        n1 = self.n1
        n2 = self.n2
        for myReadImNum in range(self.how_many_data_file):
            LRIm = self.Lr_list[myReadImNum].numpy()
            numData = LRIm.shape[0]
            
            dnmInterp = [torch.from_numpy(interpImage(LRIm[i,0], n1, n2)) for i in range(numData)]
            dnmInterp2 = torch.cat(tuple(dnmInterp),0).reshape(numData, 1, n1, n2)
            
            if (self.useTwoChannelInput):
                dnmInterp = [torch.from_numpy(interpImage(LRIm[i,1], n1, n2)) for i in range(numData)]
                dnmInterp3 = torch.cat(tuple(dnmInterp),0).reshape(numData, 1, n1, n2)
                dnmInterp2 = torch.cat((dnmInterp2, dnmInterp3), 1)
                
            self.Bicubic_list.append(dnmInterp2)
