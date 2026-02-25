# -*- coding: utf-8 -*-
"""
Created on Tue Oct  5 19:13:50 2021

@author: behnood
"""

#from __future__ import print_function
import matplotlib.pyplot as plt
#%matplotlib inline
# from numpy import linalg as LA
import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import math
import numpy as np
#from models import *
#import math
import torch
import torch.optim
import torch.nn as nn
import utilities
import hydra 
from omegaconf import DictConfig 
# from skimage.measure import compare_psnr
# from skimage.measure import compare_mse
#from utils.denoising_utils import *

# from skimage._shared import *
# from skimage.util import *
# from skimage.metrics.simple_metrics import _as_floats
# from skimage.metrics.simple_metrics import mean_squared_error

#from UtilityMine import add_noise
# from UtilityMine import find_endmember
# from UtilityMine import add_noise
from UtilityMine import *
# from VCA import *
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark =True
dtype = torch.cuda.FloatTensor

PLOT = True
#%% Load image
import scipy.io
import scipy.linalg
#%%
# fname2  = "HS Data/Samson/Y_clean.mat"
# mat2 = scipy.io.loadmat(fname2)
# img_np_gt = mat2["Y_clean"]
# img_np_gt = img_np_gt.transpose(2,0,1)
# [p1, nr1, nc1] = img_np_gt.shape
# print(img_np_gt.shape)


@hydra.main(version_base=None,config_path='config',config_name='samson')
def main(cfg:DictConfig):
    DATASET = cfg.dataset

    DATA_PATH = os.path.join("../hsi_datasets/",f"{DATASET}.mat")
    praveen_data = scipy.io.loadmat(DATA_PATH)

    Y = praveen_data['Y']
    A = praveen_data['A']
    M = praveen_data['M']
    H,W = map(int,list(praveen_data['HW'].ravel()))

    img_np_gt = Y.reshape(-1,H,W)
    [p1, nr1, nc1] = img_np_gt.shape
    # rmax=3#E_np.shape[1] 
    rmax = A.shape[0]
    #%%

    tol2=1
    save_result=False
    from tqdm import tqdm

    LR = cfg.lr
    EPOCH = cfg.epoch 
    LAMB = cfg.lamb 
    for fi in (range(1)):
        for fj in (range(tol2)):
                #%%
            #img_noisy_np = get_noisy_image(img_np_gt, 1/10)
            img_noisy_np = img_np_gt# add_noise(img_np_gt, 1/npar[0,fi])#11.55 20 dB, 36.7 30 dB, 116.5 40 dB
            #print(compare_snr(img_np_gt, img_noisy_np))
            img_resh=np.reshape(img_noisy_np,(p1,nr1*nc1))
            V, SS, U = scipy.linalg.svd(img_resh, full_matrices=False)
            PC=np.diag(SS)@U
            # img_resh_DN=V[:,:rmax]@PC[:rmax,:]
            img_resh_DN=V[:,:rmax]@V[:,:rmax].transpose(1,0)@img_resh
            img_resh_np_clip=np.clip(img_resh_DN, 0, 1)
            II,III = Endmember_extract(img_resh_np_clip,rmax)
            E_np1=img_resh_np_clip[:,II]
            #%% Set up Simulated 
            INPUT = 'noise' # 'meshgrid'
            pad = 'reflection'
            need_bias=True
            OPT_OVER = 'net' 
            
            # 
            LR1 = LR
            show_every = 100
            exp_weight=0.99
            
            num_iter1 = EPOCH
            input_depth =  img_noisy_np.shape[0]
            class CAE_EndEst(nn.Module):
                def __init__(self):
                    super(CAE_EndEst, self).__init__()
                    self.conv1 = nn.Sequential(
                        conv(input_depth, 256,3,1,bias=need_bias, pad=pad),
                        nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.LeakyReLU(0.1, inplace=True),
                    )
                    self.conv2 = nn.Sequential(
                        conv(256, 256,3,1,bias=need_bias, pad=pad),
                        nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.LeakyReLU(0.1, inplace=True),
                    )
                    self.conv3 = nn.Sequential(
                        conv(input_depth, 4, 1,1,bias=need_bias, pad=pad),
                        nn.BatchNorm2d(4,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.LeakyReLU(0.1, inplace=True),
                    )
                    self.dconv2 = nn.Sequential(
                        nn.Upsample(scale_factor=1),
                        conv(260, 256, 3,1,bias=need_bias, pad=pad),
                        nn.BatchNorm2d(256,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.LeakyReLU(0.1, inplace=True),
                    )
            
                    self.dconv3 = nn.Sequential(
                        nn.Upsample(scale_factor=1),
                        conv(256, rmax, 3,1,bias=need_bias, pad=pad),
                        nn.BatchNorm2d(rmax,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                        nn.Softmax(),
                    )
                    self.dconv4 = nn.Sequential(
                        nn.Linear(rmax, p1,bias=False),
                    )
                def forward(self, x):
                    x1 = self.conv3(x)
                    x = self.conv1(x)
                    x = torch.cat([x,x1], 1)
                    x = self.dconv2(x)
                    x2 = self.dconv3(x)
                    x3 = torch.transpose(x2.view((rmax,nr1*nc1)),0,1)
                    x3 = self.dconv4(x3)
                    return x2,x3

            net1 = CAE_EndEst()
            net1.cuda()
            
            # Loss
            def my_loss(target, End2, lamb, out_):
                loss1 = 0.5*torch.norm((out_.transpose(1,0).view(1,p1,nr1,nc1) - target), 'fro')**2
                O = torch.mean(target.view(p1,nr1*nc1),1).type(dtype).view(p1,1)
                B = torch.from_numpy(np.identity(rmax)).type(dtype)
                loss2 = torch.norm(torch.mm(End2,B.view((rmax,rmax)))-O, 'fro')**2
                return loss1+lamb*loss2
            img_noisy_torch = torch.from_numpy(img_resh_DN).view(1,p1,nr1,nc1).type(dtype)
            net_input1 = get_noise(input_depth, INPUT,
                (img_noisy_np.shape[1], img_noisy_np.shape[2])).type(dtype).detach()
            E_torch = torch.from_numpy(E_np1).type(dtype)
            #%%
            # net_input_saved = net_input1.detach().clone()
            # noise = net_input1.detach().clone()
            out_avg = True
            
            i = 0
            def closure1():
                
                nonlocal i,  out_avg, out_avg_np, Eest
                
                out_LR,out_spec = net1(net_input1)
                if out_avg is None:
                    out_avg = out_LR.detach()
                else:
                    out_avg = out_avg * exp_weight + out_LR.detach() * (1 - exp_weight)
                total_loss = my_loss(img_noisy_torch, net1.dconv4[0].weight,LAMB,out_spec)
                total_loss.backward()
                i += 1       
                return total_loss
            net1.dconv4[0].weight=torch.nn.Parameter(E_torch.view(p1,rmax))       
            p11 = get_params(OPT_OVER, net1, net_input1)
            optimizer = torch.optim.Adam(p11, lr=LR1, betas=(0.9, 0.999), eps=1e-8,
                    weight_decay= 0, amsgrad=False)
            for j in tqdm(range(num_iter1)):
                optimizer.zero_grad()
                closure1()  
                optimizer.step()
                net1.dconv4[0].weight.data[net1.dconv4[0].weight <= 0] = 0
                net1.dconv4[0].weight.data[net1.dconv4[0].weight >= 1] = 1
                if j>0:
                    Eest=net1.dconv4[0].weight.detach().cpu().squeeze().numpy()
                    # if PLOT and j % show_every== 0: 
                    #   plt.plot(Eest)
                    #   plt.show()
                    
            out_avg_np = out_avg.detach().cpu().squeeze().numpy()
            print(f"{Eest.shape=}, {out_avg_np.shape}")
            A_pred = out_avg_np.reshape(out_avg_np.shape[0],-1)
            M_pred = Eest
        

        #%%
            # if  save_result is True:
            #           scipy.io.savemat("Result/EestdB%01d%01d.mat" % (fi+2, fj+1),
            #                             {'Eest%01d%01d' % (fi+2, fj+1):Eest})
            #           scipy.io.savemat("Result/out_avg_npdB%01d%01d.mat" % (fi+2, fj+1),
            #                             {'out_avg_np%01d%01d' % (fi+2, fj+1):out_avg_np.transpose(1,2,0)})
            # #
            A_pred, M_pred  = utilities.correct_permuation(
                A_pred = A_pred,
                A_true= A,
                M_pred = M_pred,
                M_true=M
            )

            SAVE_PATH_PARENT = "outputs"
            os.makedirs(SAVE_PATH_PARENT,exist_ok=True)
            SAVE_PATH = os.path.join(SAVE_PATH_PARENT,DATASET)
            os.makedirs(SAVE_PATH,exist_ok=True)
            results = utilities.calculate_errors(
                A_pred = A_pred,
                A_true= A,
                M_pred = M_pred,
                M_true=M,
                save_path=SAVE_PATH
            )
            utilities.plot_figures(
                A_pred = A_pred,
                A_true= A,
                M_pred = M_pred,
                M_true=M,
                save_path=SAVE_PATH,
                H = H
            ) 
            final_result = results['total_rmse'].item()
            if(math.isnan(final_result)):
                final_result = 100
            else:
                final_result = float(final_result)

            print(final_result)
            return final_result


if __name__ == "__main__":
    main()
