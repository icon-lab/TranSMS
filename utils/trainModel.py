# import libraries
import torch
from torch import nn
import numpy as np
# from skimage.transform import rescale, resize, downscale_local_mean
import datetime
from utils.mpiSuperResUtils import *

class myDataset:
    def __init__(self, HRimages, LRimages, maxhr, minhr, batch_size, up = 0.7, down = 0.15, ordertrain = 0):
        self.HR = HRimages.cuda()
        self.LR = LRimages.cuda()
        self.bs = batch_size
        self.max = maxhr.cuda()
        self.min = minhr.cuda()
        self.up = up
        self.down = down
        self.ordertrain = ordertrain
    
    def shuffleAll(self):
        np.random.shuffle(self.ordertrain)
        self.LR = self.LR[self.ordertrain,:,:,:]
        self.HR = self.HR[self.ordertrain,:,:,:]
        self.max = self.max[self.ordertrain,:]
        self.min = self.min[self.ordertrain,:]
        
    def normalize(self, x):
        return (x - self.min[:,None,None])/(self.max[:,None,None]-self.min[:,None,None])*self.up + self.down
    
    def denormalize(self, x):
        return ((x-self.down)/self.up)*(self.max[:,None,None]-self.min[:,None,None])+self.min[:,None,None]
    
    def noisePowerNormalizeWObatch(self, x):
        return (x)/(self.max[:,None,None]-self.min[:,None,None])*self.up
        
    def noisePowerDeNormalizeWObatch(self, x):
        return (x)*(self.max[:,None,None]-self.min[:,None,None])/self.up
    
    def noisePowerNormalize(self, x, batch_start_idx):
        return (x)/(self.max[batch_start_idx:batch_start_idx+self.bs,None,None]-self.min[batch_start_idx:batch_start_idx+self.bs,None,None])*self.up
    
    def noisePowerDeNormalize(self, x, batch_start_idx):
        return (x)*(self.max[batch_start_idx:batch_start_idx+self.bs,None,None]-self.min[batch_start_idx:batch_start_idx+self.bs,None,None])/self.up
    
def trainMyModel(model, epoch_nb, loss, optimizer, scheduler, trainDS, validDS, ordertrain, nsPwr, batch_size, optionalMessage = '', saveModelEpoch = 0, directory = ''):
    batch_size = trainDS.bs
    nrmses = list()
    losses = np.zeros(epoch_nb)
    with torch.no_grad():
        temp_test_nrmseDivider = float(torch.norm(validDS.denormalize(validDS.HR)))
    for epoch in range(int(epoch_nb)):
        model.train()
        tempLosses = list()
        if (saveModelEpoch > 0) :
            if (epoch % saveModelEpoch == 0):
                torch.save(model.state_dict(), directory+r"/_time"+datetime.datetime.now().strftime("%Y.%m.%d")+"_model_save_kfold_"+optionalMessage+"_epoch" +str(epoch)+model.__class__.__name__ +".pt")
        
        for batch_start_idx in range(0,trainDS.LR.shape[0] - (trainDS.LR.shape[0] % batch_size), batch_size):
            imghr = trainDS.HR[batch_start_idx:batch_start_idx+batch_size,:,:,:]
            imglr = trainDS.LR[batch_start_idx:batch_start_idx+batch_size,:,:,:]
            
            if (nsPwr > 0):
                nsTerm = 6e-11 * nsPwr * torch.randn_like(imglr)
                nsNrmlz = trainDS.noisePowerNormalize(nsTerm, batch_start_idx)
                imglr = imglr + nsNrmlz

            fakehr = model(imglr)
            model.zero_grad()
            model_loss = loss(fakehr,imghr)
            model_loss.backward()

            optimizer.step()
            tempLosses.append(float(model_loss))
            
            with torch.no_grad():
                if (((batch_start_idx//batch_size)%200==0)and(epoch%10==0)):

                    train_nRMSE = float(torch.norm(fakehr - imghr) / torch.norm(imghr))
                    info_str = "Epoch: {0}, [{1}/{2}], Train Loss = {3:.5f}, Train norm(error)/norm(real)={4:.5f} \n ".format(epoch, batch_size+batch_start_idx,trainDS.LR.shape[0],float(model_loss), train_nRMSE)
                    print(optionalMessage+"----------"+info_str)
            
        # back to epoch        
        scheduler.step()
        
        trainDS.shuffleAll()
        losses[epoch] = sum(tempLosses)/len(tempLosses)
        model.eval()
        if epoch%15 ==0:
            with torch.no_grad():
                model.eval()
                x = validDS.LR
                y_ground = validDS.HR
                y_model = torch.zeros_like(y_ground)
                iii = 0 
                while(iii<y_model.shape[0]-(y_model.shape[0]%validDS.bs)):
                    y_model[iii:iii+validDS.bs,:,:,:] = model(x[iii:iii+validDS.bs,:,:,:])
                    iii += validDS.bs
                y_model[iii:,:,:,:] = model(x[iii:,:,:,:])
                temp_test_nrmse = torch.norm(validDS.denormalize(y_model)-validDS.denormalize(y_ground))/torch.norm(validDS.denormalize(y_ground))
                
                print(optionalMessage, ' Epoch:', epoch, 'test set nrmse: ', temp_test_nrmse)
                nrmses.append(temp_test_nrmse)
                
    return model, losses, nrmses
        
def downsampleImageGPU(x, dx, dy): #Box downsampling
#    y = torch.zeros(x.shape[0], x.shape[1]//dx, x.shape[2]//dy)
    y = torch.cuda.FloatTensor(x.shape[0], x.shape[1], x.shape[2]//dx, x.shape[3]//dy).fill_(0)
    for ii in range(dx):
        for jj in range(dy):
            y += x[:,:, ii::dx, jj::dy]
    return y / (dx * dy)

def upsampleImageGPU(x, dx, dy): #Box upsampling
#    y = torch.zeros(x.shape[0], x.shape[1] * dx, x.shape[2] * dy)
    y = torch.cuda.FloatTensor(x.shape[0], x.shape[1], x.shape[2] * dx, x.shape[3] * dy)
    for ii in range(dx):
        for jj in range(dy):
            y[:, :, ii::dx, jj::dy] = x
    return y

def projectToNoiseLevel(x, y, epsList, dx ,dy):
    y_hat = y - downsampleImageGPU(x, dx ,dy)
    yNorm = (y_hat**2).sum(axis = 3).sum(axis = 2).sum(axis = 1).sqrt()
    workWithIndices = yNorm > epsList[:,0]
    x_est = x.clone()
    x_est[workWithIndices, :, :, :] += (1 - epsList[workWithIndices,:] / yNorm[workWithIndices,None])[:,:,None,None] * upsampleImageGPU(y_hat[workWithIndices,:,:,:], dx ,dy)
    return x_est

def projectToNoiseLevelOnBatch(x, y, nsPwr, dx ,dy):
    y_hat = y - downsampleImageGPU(x, dx ,dy)
    yNorm = (y_hat**2).sum(axis = 1).sum(axis = 0).sqrt()
    theYnorm = upsampleImageGPU(yNorm[None,None,:,:], dx, dy)
    theEps = nsPwr * ((x.shape[0]) ** (1/2))
    workWithIndices = theYnorm > theEps
    
    x_est = x.clone()
    x_est += workWithIndices * (1 - theEps / theYnorm) * upsampleImageGPU(y_hat, dx ,dy)
    return x_est.reshape(x.shape)

def trainMyModelNsProjectedListWithPower(model, epoch_nb, loss, optimizer, scheduler, trainDS, validDS, ordertrain, nsPwr, batch_size, optionalMessage = '', saveModelEpoch = 0, directory = ''):
    batch_size = trainDS.bs
    scaleX = int(trainDS.HR.shape[2] / trainDS.LR.shape[2])
    scaleY = int(trainDS.HR.shape[3] / trainDS.LR.shape[3])
    nrmses = list()
    losses = np.zeros(epoch_nb)
    with torch.no_grad():
        temp_test_nrmseDivider = float(torch.norm(validDS.denormalize(validDS.HR)))
        
    for epoch in range(int(epoch_nb)):
        model.train()
        tempLosses = list()
        
        if (saveModelEpoch > 0) and (epoch % saveModelEpoch == 0):
            torch.save(model.state_dict(), directory+r"/_time"+datetime.datetime.now().strftime("%Y.%m.%d")+"_model_save_kfold_"+optionalMessage+"_epoch" +str(epoch)+model.__class__.__name__ +".pt")
        
        for batch_start_idx in range(0,trainDS.LR.shape[0]- batch_size, batch_size):
            imghr = trainDS.HR[batch_start_idx:batch_start_idx+batch_size,:,:,:]
            imglr = trainDS.LR[batch_start_idx:batch_start_idx+batch_size,:,:,:]
#            nsEps = nsPwrListTrain[batch_start_idx:batch_start_idx+batch_size, :]
            with torch.no_grad():
                nsEps = trainDS.noisePowerNormalize(nsPwr, batch_start_idx).reshape(-1, 1)

            model_out = model(imglr)
            fakehr = projectToNoiseLevel(model_out, imglr, nsEps, scaleX, scaleY)

            # project fakeHR
            
            model.zero_grad()
            model_loss = loss(fakehr,imghr)
            model_loss.backward()

            optimizer.step()
            tempLosses.append(float(model_loss))
            
            with torch.no_grad():
                if (((batch_start_idx//batch_size)%200==0)and(epoch%10==0)):

                    train_nRMSE = float(torch.norm(fakehr - imghr) / torch.norm(imghr))
                    info_str = "Epoch: {0}, [{1}/{2}], Train Loss = {3:.5f}, Train norm(error)/norm(real)={4:.5f} \n ".format(epoch, batch_size+batch_start_idx,trainDS.LR.shape[0],float(model_loss), train_nRMSE)
                    print(optionalMessage+"----------"+info_str)

        # back to epoch        
        scheduler.step()
        
        trainDS.shuffleAll()
        losses[epoch] = sum(tempLosses)/len(tempLosses)
        
        model.eval()
        if epoch%30 ==0:
            with torch.no_grad():
                
                temp_test_nrmse = float(0)
                for batch_start_idx in range(0,validDS.LR.shape[0]- validDS.bs, validDS.bs):
                    valimghr = validDS.HR[batch_start_idx:batch_start_idx+validDS.bs,:,:,:]
                    valimglr = validDS.LR[batch_start_idx:batch_start_idx+validDS.bs,:,:,:]
                    y_modelOut = model(valimglr)
                    
                    with torch.no_grad():
                        nsEpsTest = validDS.noisePowerNormalize(nsPwr, batch_start_idx).reshape(-1, 1)
                    
                    y_model = projectToNoiseLevel(y_modelOut, valimglr, nsEpsTest, scaleX, scaleY)

                    temp_test_nrmse += float(torch.norm(validDS.noisePowerDeNormalize(y_model - valimghr, batch_start_idx)))**2
                temp_test_nrmse = (temp_test_nrmse)**(1/2) / temp_test_nrmseDivider

                print(optionalMessage, ' Epoch:', epoch, 'test set nrmse: ', temp_test_nrmse)
                nrmses.append(temp_test_nrmse)

    return model, losses, nrmses
        

    
def test_model_wbatch_DS(model, testDS):
    with torch.no_grad():
        model.eval()
        x = testDS.LR
        y_ground = testDS.HR
        y_model = torch.zeros_like(y_ground)
        iii = 0
        while(iii<y_model.shape[0]-(y_model.shape[0]%testDS.bs)):
            y_model[iii:iii+testDS.bs,:,:,:] = model(x[iii:iii+testDS.bs,:,:,:])
            iii += testDS.bs
        y_model[iii:,:,:,:] = model(x[iii:,:,:,:])
        return torch.norm(testDS.denormalize(y_model)-testDS.denormalize(y_ground))/torch.norm(testDS.denormalize(y_ground)), y_model

    
def trainMyModelSRCNN(model, epoch_nb, loss, optimizer, scheduler, trainDS, validDS, ordertrain, nsPwr, batch_size, interpolateWithMatrice, optionalMessage = '', saveModelEpoch = 0, directory = ''):
    batch_size = trainDS.bs
    nrmses = list()
    losses = np.zeros(epoch_nb)
    with torch.no_grad():
        temp_test_nrmseDivider = float(torch.norm(validDS.denormalize(validDS.HR)))
    for epoch in range(int(epoch_nb)):
        model.train()
        tempLosses = list()
        if (saveModelEpoch > 0) :
            if (epoch % saveModelEpoch == 0):
                torch.save(model.state_dict(), directory+r"/_time"+datetime.datetime.now().strftime("%Y.%m.%d")+"_model_save_kfold_"+optionalMessage+"_epoch" +str(epoch)+model.__class__.__name__ +".pt")
        
        for batch_start_idx in range(0,trainDS.LR.shape[0] - (trainDS.LR.shape[0] % batch_size), batch_size):
            imghr = trainDS.HR[batch_start_idx:batch_start_idx+batch_size,:,:,:]
            imglr = trainDS.LR[batch_start_idx:batch_start_idx+batch_size,:,:,:]
            
            if (nsPwr > 0):
                nsTerm = 6e-11 * nsPwr * torch.randn_like(imglr)
                nsNrmlz = trainDS.noisePowerNormalize(nsTerm, batch_start_idx)
                imglr = imglr + nsNrmlz

            imgbi = interpolateWithMatrice(imglr)
            fakehr = model(imgbi)
            imghr = imghr[:,:,6:imgbi.shape[2]-6,6:imgbi.shape[3]-6]
            model.zero_grad()
            model_loss = loss(fakehr,imghr)
            model_loss.backward()

            optimizer.step()
            tempLosses.append(float(model_loss))
            
            with torch.no_grad():
                if (((batch_start_idx//batch_size)%200==0)and(epoch%10==0)):

                    train_nRMSE = float(torch.norm(fakehr - imghr) / torch.norm(imghr))
                    info_str = "Epoch: {0}, [{1}/{2}], Train Loss = {3:.5f}, Train norm(error)/norm(real)={4:.5f} \n ".format(epoch, batch_size+batch_start_idx,trainDS.LR.shape[0],float(model_loss), train_nRMSE)
                    print(optionalMessage+"----------"+info_str)
            
        # back to epoch        
        scheduler.step()
        
        trainDS.shuffleAll()
        losses[epoch] = sum(tempLosses)/len(tempLosses)
        model.eval()
        if epoch%15 ==0:
            with torch.no_grad():
                model.eval()
                x = validDS.LR
                y_ground = validDS.HR[:,:,6:imgbi.shape[2]-6,6:imgbi.shape[3]-6]
                y_model = torch.zeros_like(y_ground)
                iii = 0 
                while(iii<y_model.shape[0]-(y_model.shape[0]%validDS.bs)):
                    y_model[iii:iii+validDS.bs,:,:,:] = model(interpolateWithMatrice(x[iii:iii+validDS.bs,:,:,:]))
                    iii += validDS.bs
                y_model[iii:,:,:,:] = model(interpolateWithMatrice(x[iii:,:,:,:]))
                temp_test_nrmse = torch.norm(validDS.denormalize(y_model)-validDS.denormalize(y_ground))/torch.norm(validDS.denormalize(y_ground))

                
                print(optionalMessage, ' Epoch:', epoch, 'test set nrmse: ', temp_test_nrmse)
                nrmses.append(temp_test_nrmse)
                
    return model, losses, nrmses
    

def trainMyModelVDSR(model, epoch_nb, loss, optimizer, scheduler, trainDS, validDS, ordertrain, nsPwr, batch_size, interpolateWithMatrice, optionalMessage = '', saveModelEpoch = 0, directory = ''):
    batch_size = trainDS.bs
    nrmses = list()
    losses = np.zeros(epoch_nb)
    with torch.no_grad():
        temp_test_nrmseDivider = float(torch.norm(validDS.denormalize(validDS.HR)))
    for epoch in range(int(epoch_nb)):
        model.train()
        tempLosses = list()
        if (saveModelEpoch > 0) :
            if (epoch % saveModelEpoch == 0):
                torch.save(model.state_dict(), directory+r"/_time"+datetime.datetime.now().strftime("%Y.%m.%d")+"_model_save_kfold_"+optionalMessage+"_epoch" +str(epoch)+model.__class__.__name__ +".pt")
        
        for batch_start_idx in range(0,trainDS.LR.shape[0] - (trainDS.LR.shape[0] % batch_size), batch_size):
            imghr = trainDS.HR[batch_start_idx:batch_start_idx+batch_size,:,:,:]
            imglr = trainDS.LR[batch_start_idx:batch_start_idx+batch_size,:,:,:]
            
            if (nsPwr > 0):
                nsTerm = 6e-11 * nsPwr * torch.randn_like(imglr)
                nsNrmlz = trainDS.noisePowerNormalize(nsTerm, batch_start_idx)
                imglr = imglr + nsNrmlz

            imgbi = interpolateWithMatrice(imglr)
            fakehr = model(imgbi)
            model.zero_grad()
            model_loss = loss(fakehr,imghr)
            model_loss.backward()
            nn.utils.clip_grad_norm(model.parameters(), 0.4)

            optimizer.step()
            tempLosses.append(float(model_loss))
            
            with torch.no_grad():
                if (((batch_start_idx//batch_size)%200==0)and(epoch%10==0)):

                    train_nRMSE = float(torch.norm(fakehr - imghr) / torch.norm(imghr))
                    info_str = "Epoch: {0}, [{1}/{2}], Train Loss = {3:.5f}, Train norm(error)/norm(real)={4:.5f} \n ".format(epoch, batch_size+batch_start_idx,trainDS.LR.shape[0],float(model_loss), train_nRMSE)
                    print(optionalMessage+"----------"+info_str)
            
        # back to epoch        
        scheduler.step()
        
        trainDS.shuffleAll()
        losses[epoch] = sum(tempLosses)/len(tempLosses)
        model.eval()
        if epoch%15 ==0:
            with torch.no_grad():
                model.eval()
                x = validDS.LR
                y_ground = validDS.HR
                y_model = torch.zeros_like(y_ground)
                iii = 0 
                while(iii<y_model.shape[0]-(y_model.shape[0]%validDS.bs)):
                    y_model[iii:iii+validDS.bs,:,:,:] = model(interpolateWithMatrice(x[iii:iii+validDS.bs,:,:,:]))
                    iii += validDS.bs
                y_model[iii:,:,:,:] = model(interpolateWithMatrice(x[iii:,:,:,:]))
                temp_test_nrmse = torch.norm(validDS.denormalize(y_model)-validDS.denormalize(y_ground))/torch.norm(validDS.denormalize(y_ground))

                
                
                print(optionalMessage, ' Epoch:', epoch, 'test set nrmse: ', temp_test_nrmse)
                nrmses.append(temp_test_nrmse)
                
    return model, losses, nrmses
