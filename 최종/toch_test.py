# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 00:39:18 2021

@author: dudgo
"""


import sklearn
import torch
import multiprocessing as mp
import pandas as pd
import numpy as np

# ==============================================================================
# 1. Load  yield,  t= 1/12,2/12, ...580/12 (1971.08~~2019.12)(monthly), maturity=1,2,3,4,5,6,7,8,9,10(yearly)
# ==============================================================================

#Read yield data, by using skiprows delete 'data source' etc...
y_raw_data = pd.read_excel('LW_monthly.xlsx',sheet_name='Sheet1',skiprows=7,header=1)

# re-set column name : data, 1m, 2m,3m .... 360m, since exist 'space' in excel -> '  4m', ' 10m' 
col_name=['date']
for i in range(1,y_raw_data.shape[1]):
    col_name.append(str(i)+'m')

y_raw_data.columns = col_name

# date를 index로 지정
indexed_y = y_raw_data.set_index(['date'])

# we need 12m,24m,36m, ... 120m
yearly_index=[]
for i in range(1,11):
    #yearly_index.append(str(12*i)+'m')
    yearly_index.append(12*i-1)


# 1971.08~~2019.12 /  12m,24m,36m....120m 추출 
yield_data = indexed_y.iloc[122:,yearly_index] # 122 : 1961.06~1971.07제외

del y_raw_data
del indexed_y



# ===================================================================================
#            2.    Load   Macro-data,   1971.08~~2019.12 (monthly),  Na > column average (수정가능)
# ===================================================================================

# read macro data 
m_raw_data = pd.read_csv('current.csv')
m_raw_data = m_raw_data.iloc[1:,:]   # raw 0 : transform method > delete

# 매월 1일 기준 data > 하루 빼서 yield data와 맞춤 
m_raw_data['sasdate'] = pd.to_datetime(m_raw_data['sasdate']) - pd.DateOffset(days=1)
m_raw_data['sasdate'] = m_raw_data['sasdate'].dt.date
indexed_m = m_raw_data.set_index(['sasdate'])

# 1971.08~~2019.12 까지 추출
macro_data = indexed_m.iloc[152:733,:]


# some variables have Na, > fill Na with average 
macro_na = pd.isna(macro_data).sum()
macro_data = macro_data.fillna(macro_data.mean())

del m_raw_data
del indexed_m


    
# =========================================================================
#        3.     yield > calculate forwad & excess-return data 
# =========================================================================


#from yield curve data, construct forward rate & excess return 
#Note that yield data is annually & conti-compounded.
#yld, forward-rate, excess-return are all described in annulized decimal notation  

for i in range(len(yearly_index)):
    yearly_index[i]=str(12*(i+1))+'m'
    
yld = 1/100*yield_data.values    # to np.ndarray


# log(price)
log_p = np.zeros(yld.shape) 

for i in range(log_p.shape[1]):
    log_p[:,i] = -(i+1) * yld[:,i]
   
    
# forward rate,   Note that the first column of forward-rate matrix is the same as that of yield curve 
fwd_rate = np.zeros(log_p.shape)

for i in range(log_p.shape[1]):
    if i ==0:
        fwd_rate[:,i] = yld[:,i]
    else:
        fwd_rate[:,i] = log_p[:,i-1] - log_p[:,i]   
        

# excess return , the 1st column & row are all 0 
xr_rate = np.zeros(log_p.shape)

for i in range(12,log_p.shape[0]):
    for j in range(1,log_p.shape[1]):
        xr_rate[i,j] = log_p[i,j-1] - log_p[i-12,j] - fwd_rate[i-12,0] 
 

fwd_rate = pd.DataFrame(data = fwd_rate, columns = yearly_index, index = macro_data.index)
xr_rate = pd.DataFrame(data=xr_rate, columns= yearly_index, index = macro_data.index)

# ------ 여기까지는 일단, macro / fwd rate 1971.08 ~ 2019.12월까지 ,  xr_rate는 : 1972.08~2020.12월까지 만듬




# ==============================================================================
#    4.  Setting sample period (** data of x(RHS) variable) 
# ==============================================================================

from datetime import datetime
from dateutil import relativedelta

# All sample period
All_sample_Start = '1971-08-31'   # Date of RHS variable where to start sample. choose 1971.08.31~~
All_sample_End = '2018-12-31'     # Date of RHS variable where to end sample    Choose  ~2018.12.31
delta_all = relativedelta.relativedelta(datetime.strptime(All_sample_End, "%Y-%m-%d"), datetime.strptime(All_sample_Start, "%Y-%m-%d"))
all_sp_len = 12*delta_all.years + delta_all.months + 1  


# OOS period 
OOS_Start = '1989-01-31'  # Date of RHS variable where to start OOS
OOS_End = All_sample_End    # Date of RHS variable where to end OOS
delta = relativedelta.relativedelta(datetime.strptime(OOS_End, "%Y-%m-%d"), datetime.strptime(OOS_Start, "%Y-%m-%d"))
OOS_len = 12*delta.years + delta.months + 1  



# ===========================================================================
#     5.  Variable Setting for estimation (according to above period)  
# ===========================================================================

# X-varaible : 1971.8 ~ 2018.12
Xexog = fwd_rate.iloc[-all_sp_len-12:-12,:]  # fwd rate   ** -12 : 2019.01 ~ 2019.12 제거 
X = macro_data.iloc[-all_sp_len-12:-12,:]    # macro   ** -12 : 2019.01 ~ 2019.12 제거 

# Y-varaible : 1972.8 ~ 2019.12,  (n) = 2,3,4,5,7,10 (p20 Table 1 in Bianchi,Buchner,Tamoni(2020))
maturity = [1,2,3,4,6,9] #(n) = 2,3,4,5,7,10
Y = xr_rate.iloc[-all_sp_len:, maturity]   # excess_return,  

y_true = Y.iloc[-OOS_len:, :]  # true y for OOS,  for calculating R2OOS


# Computational Ressources: Determine Number of available cores
ncpus = mp.cpu_count()
print("CPU count is: "+str(ncpus))

X=X.values
Xexog =Xexog.values
Y=Y.values

#-------------------------------------------------------

import torch 
import torch.nn as nn
import torch.nn.init as init 
import torch.optim as optim
import torch.nn.functional as F
from sklearn.preprocessing import MinMaxScaler
import numpy as np
#---------------------------------------------   
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=20, verbose=False, delta=1e-6, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 1e-6
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
    
    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss
        
#-----------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
#seed 
torch.manual_seed(1)
np.random.seed(1)
 

#Split Data for Test and Training
X_train = X[:-1,:]  
Xexog_train = Xexog[:-1,:]
Y_train = Y[:-1,:]

X_test = X[-1,:].reshape(1,-1)
Xexog_test = Xexog[-1,:].reshape(1,-1)


#Scale the predictors for training
Xscaler_train =  MinMaxScaler(feature_range=(-1,1))
X_scaled_train = Xscaler_train.fit_transform(X_train)

Xexog_scaler_train =  MinMaxScaler(feature_range=(-1,1))
Xexog_scaled_train = Xexog_scaler_train.fit_transform(Xexog_train)

# split train/ cross validation
N_train = int(np.round(np.size(X_train,axis=0)*0.85))
N_val = np.size(X_train,axis=0)-N_train

X_scaled_val = X_scaled_train[N_train:,:]
X_scaled_train = X_scaled_train[:N_train,:]
Xexog_scaled_val = Xexog_scaled_train[N_train:,:]
Xexog_scaled_train = Xexog_scaled_train[:N_train,:]


# Define Model Architecture
archi=[5,4,3]

    
# Base model for macro variables
n = len(archi)
macro_module = torch.nn.Sequential()
dropout_prob=0.5

for i in range(n):
    if i==0:
        macro_module.add_module('dropout'+str(i+1), nn.Dropout(dropout_prob))
        macro_module.add_module('linear'+str(i+1), nn.Linear(X_scaled_train.shape[1], archi[i]))
        macro_module.add_module('Relu'+str(i+1), nn.ReLU())
    else:
        macro_module.add_module('dropout'+str(i+1), nn.Dropout(dropout_prob))
        macro_module.add_module('linear'+str(i+1), nn.Linear(archi[i-1], archi[i]))
        macro_module.add_module('Relu'+str(i+1), nn.ReLU())

macro_module.add_module('linear555',nn.Linear(X_scaled_train.shape[1]+Xexog_scaled_train.shape[1],1))

# Using He-initilization 
for m in macro_module:
    if isinstance(m,nn.Linear):
         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
        
# for macro + fwrd model 
merge_module = torch.nn.Sequential()
merge_module.add_module('dropout_final', nn.Dropout(dropout_prob))
merge_module.add_module('BN', nn.BatchNorm1d(X_scaled_train.shape[1]+Xexog_scaled_train.shape[1]))
merge_module.add_module('linear_final', nn.Linear(X_scaled_train.shape[1]+Xexog_scaled_train.shape[1], 1))

# Using He-initilization 
for m in merge_module:
    if isinstance(m,nn.Linear):
         nn.init.kaiming_normal_(m.weight)
         

loss_ftn = torch.nn.MSELoss(reduction='sum')
torch.optim.SGD(list(macro_module.parameters()) + list(merge_module.parameters()), lr=0.01, momentum=0.9, weight_decay=0.01, nesterov=True)



"""
for i in range(n):
    if i==0:
        layers['ins_main'] = torch.from_numpy(X_scaled_train)
    elif i==1:
        layers['dropout'+str(i)] = nn.Dropout(.5)(layers['ins_main'])
        layers['linear'+str(i)] = nn.Linear(X_scaled_train.shape[1], archi[i-1])(layers['dropout'+str(i)].float())
        layers['Relu'+str(i)] = nn.ReLU()(layers['linear'+str(i)])       
    else:
        layers['dropout'+str(i)] = nn.Dropout(.5)(layers['Relu'+str(i-1)])
        layers['linear'+str(i)] = nn.Linear(archi[i-2], archi[i-1])(layers['dropout'+str(i)].float())
        layers['Relu'+str(i)] = nn.ReLU()(layers['linear'+str(i)])

# Model for yield variables
layers['ins_exog'] =  torch.from_numpy(X_scaled_train)
      
        
        
        
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
"""
#        nn.init.kaiming_normal_('linear'+str(i+1), nonlinearity='relu')