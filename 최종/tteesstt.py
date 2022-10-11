# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 15:00:10 2021

@author: dudgo
"""
import torch 
import torch.nn as nn
import torch.nn.init as init 
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy as np    



class fwd_last_NN(nn.Module):
        
    
    def forward(self, X, Xexog, archi, dropout_prob=None):
        n = len(archi)
        macro_module = torch.nn.Sequential()
            
        for i in range(n):
            if i==0:
                macro_module.add_module('dropout'+str(i+1), nn.Dropout(dropout_prob))
                macro_module.add_module('linear'+str(i+1), nn.Linear(X_scaled_train.shape[1], archi[i]))
                macro_module.add_module('Relu'+str(i+1), nn.ReLU())
            else:
                macro_module.add_module('dropout'+str(i+1), nn.Dropout(dropout_prob))
                macro_module.add_module('linear'+str(i+1), nn.Linear(archi[i-1], archi[i]))
                macro_module.add_module('Relu'+str(i+1), nn.ReLU())

    