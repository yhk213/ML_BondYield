# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:19:38 2021

@author: yhkim
"""
#%%
import numpy as np
import pandas as pd 
import multiprocessing as mp
import FunLib_0524 as FL
import random

#%%
# 1. Load  yield,  t= 1/12,2/12, ...580/12 (1971.08~~2019.12)(monthly), maturity=1,2,3,4,5,6,7,8,9,10(yearly)

#Read yield data, by using skiprows delete 'data source' etc...
y_raw_data = pd.read_excel('LW_monthly.xlsx',sheet_name='Sheet1',skiprows=7,header=1)

# re-set column name : data, 1m, 2m,3m .... 360m, since exist 'space' in excel -> '  4m', ' 10m' 
col_name=['date']
for i in range(1,y_raw_data.shape[1]):
    col_name.append(str(i)+'m')

y_raw_data.columns = col_name

#%%
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

#%%
# 2. Load   Macro-data,   1971.08~~2019.12 (monthly),  Na > column average (수정가능)

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
#macro_data = macro_data.fillna(method = 'bfill')

del m_raw_data
del indexed_m


#%%
# 3. yield > calculate forwad & excess-return data 


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

# ------ 여기까지는 일단, macro / fwd rate 1971.08 ~ 2019.12월까지 ,  xr_rate는 : 1972.08~2019.12월까지 만듬


#%%
# 4.  Setting sample period (** data of x(RHS) variable) 

from datetime import datetime
from dateutil import relativedelta

# All sample period
All_sample_Start = '1971-08-31'   # Date of RHS variable where to start sample. choose 1971.08.31~~
All_sample_End = '2018-12-31'     # Date of RHS variable where to end sample    Choose  ~2018.12.31
delta_all = relativedelta.relativedelta(datetime.strptime(All_sample_End, "%Y-%m-%d"), datetime.strptime(All_sample_Start, "%Y-%m-%d"))
all_sp_len = 12*delta_all.years + delta_all.months + 1  


# OOS period 
OOS_Start = '1989-01-31'  # Date of RHS variable where to start OOS 1989-01-31~
OOS_End = All_sample_End    # Date of RHS variable where to end OOS
delta = relativedelta.relativedelta(datetime.strptime(OOS_End, "%Y-%m-%d"), datetime.strptime(OOS_Start, "%Y-%m-%d"))
OOS_len = 12*delta.years + delta.months + 1  

del delta, delta_all


#%%
# 5.  Variable Setting for estimation (according to above period)  

# X-varaible : 1971.8 ~ 2018.12
Xexog = fwd_rate.iloc[-all_sp_len-12:-12,:]  # fwd rate   ** -12 : 2019.01 ~ 2019.12 제거 
X = macro_data.iloc[-all_sp_len-12:-12,:]    # macro   ** -12 : 2019.01 ~ 2019.12 제거 

# Y-varaible : 1972.8 ~ 2019.12,  (n) = 2,3,4,5,7,10 (p20 Table 1 in Bianchi,Buchner,Tamoni(2020))
maturity = [1,2,3,4,6,9] #(n) = 2,3,4,5,7,10
#maturity = [9]
Y = xr_rate.iloc[-all_sp_len:, maturity]   # excess_return,  

y_true = Y.iloc[-OOS_len:, :]  # true y for OOS,  for calculating R2OOS


# Computational Ressources: Determine Number of available cores
ncpus = mp.cpu_count()
print("CPU count is: "+str(ncpus))


#%%
#  PCR : PCA & regression  (fwd-rate only),  using expanding windows 

num_pca = [3,5,10]  # 3,5,10   # of pca components 

predict_y_pca = np.full([len(num_pca), y_true.shape[0], y_true.shape[1]], np.nan)

for i in range(len(num_pca)):
    for j in range(OOS_len):
        
        if -(OOS_len-j)+1 !=0:
            predict_y_pca[i,j,:] = FL.Pca_regression(Xexog.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values, num_pca[i])
        else:
            predict_y_pca[i,j,:] = FL.Pca_regression(Xexog.values, Y.values, num_pca[i])

       
R2OOS_pca = np.full([len(num_pca), len(maturity)], np.nan)
for i in range(len(num_pca)):
    for j in range(len(maturity)):        
        R2OOS_pca[i,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_pca[i,:,j])
        
P_value_pca = np.full([len(num_pca), len(maturity)], np.nan)
for i in range(len(num_pca)):
    for j in range(len(maturity)):        
        P_value_pca[i,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_pca[i,:,j])

# We only print the case with the largest # of pca component & the largest maturity 
#print('R2OOS for PCA with fwrd-rate / ' 'number of pca component :', num_pca[-1],'/' ' maturity :', maturity[-1]+1)
#print(FL.R2OOS(y_true.iloc[:,-1].values, predict_y_pca[-1,:,-1]))



#%%
#  PLS  (fwd-rate only),  using expanding windows 

num_pls = [3,5]    # of pls components 

predict_y_pls = np.full([len(num_pls), y_true.shape[0], y_true.shape[1]], np.nan)

for i in range(len(num_pls)):
    for j in range(OOS_len):
        
        if -(OOS_len-j)+1 !=0:
            predict_y_pls[i,j,:] = FL.Pls_regression(Xexog.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values, num_pls[i])
        else:
            predict_y_pls[i,j,:] = FL.Pls_regression(Xexog.values, Y.values, num_pls[i])

R2OOS_pls = np.full([len(num_pls), len(maturity)], np.nan)
for i in range(len(num_pls)):
    for j in range(len(maturity)):        
        R2OOS_pls[i,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_pls[i,:,j])
        
P_value_pls = np.full([len(num_pls), len(maturity)], np.nan)
for i in range(len(num_pls)):
    for j in range(len(maturity)):        
        P_value_pls[i,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_pls[i,:,j])

# We only print the case with the largest # of pls component & the largest maturity 
#print('R2OOS for PLS with fwrd-rate / ' 'number of pls component :', num_pls[-1],'/' ' maturity :', maturity[-1]+1)
#print(FL.R2OOS(y_true.iloc[:,-1].values, predict_y_pls[-1,:,-1]))


#%%
# PCR : PCA & regression  (fwd-rate + macro variale),  using expanding windows 

num_pca_m = [8]     # of pca components 

predict_y_pca_m = np.full([len(num_pca_m), y_true.shape[0], y_true.shape[1]], np.nan)

for i in range(len(num_pca_m)):
    for j in range(OOS_len):
        
        if -(OOS_len-j)+1 !=0:
            predict_y_pca_m[i,j,:] = FL.m_Pca_regression(Xexog.iloc[:-(OOS_len-j)+1,:].values, X.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values, num_pca_m[i])
        else:
            predict_y_pca_m[i,j,:] = FL.m_Pca_regression(Xexog.values, X.values, Y.values, num_pca_m[i])


R2OOS_pca_m = np.full([len(num_pca_m), len(maturity)], np.nan)
for i in range(len(num_pca_m)):
    for j in range(len(maturity)):        
        R2OOS_pca_m[i,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_pca_m[i,:,j])
        
P_value_pca_m = np.full([len(num_pca_m), len(maturity)], np.nan)
for i in range(len(num_pca_m)):
    for j in range(len(maturity)):        
        P_value_pca_m[i,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_pca_m[i,:,j])

# We only print the case with the largest # of pca component & the largest maturity 
#print('R2OOS for PCA with fwrd-rate & macro / ' 'number of pca component :', num_pca_m[-1],'/' ' maturity :', maturity[-1]+1)
#print(FL.R2OOS(y_true.iloc[:,-1].values, predict_y_pca_m[-1,:,-1]))


#%%
# PLS : Partial least square (fwd-rate + Macro variable), using expanding windows 


num_pls_m = [8]     # of pls components 

predict_y_pls_m = np.full([len(num_pls_m), y_true.shape[0], y_true.shape[1]], np.nan)

for i in range(len(num_pls_m)):
    for j in range(OOS_len):
                    
        if -(OOS_len-j)+1 !=0:
            predict_y_pls_m[i,j,:] = FL.m_Pls_regression(Xexog.iloc[:-(OOS_len-j)+1,:].values, X.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values, num_pls_m[i])
        else:
            predict_y_pls_m[i,j,:] = FL.m_Pls_regression(Xexog.values, X.values, Y.values, num_pls_m[i])


R2OOS_pls_m = np.full([len(num_pls_m), len(maturity)], np.nan)
for i in range(len(num_pls_m)):
    for j in range(len(maturity)):        
        R2OOS_pls_m[i,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_pls_m[i,:,j])
        
P_value_pls_m = np.full([len(num_pls_m), len(maturity)], np.nan)
for i in range(len(num_pls_m)):
    for j in range(len(maturity)):        
        P_value_pls_m[i,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_pls_m[i,:,j])
# We only print the case with the largest # of pls component & the largest maturity 
#print('R2OOS for PLS with fwrd-rate & macro /' 'number of pls component :', num_pls_m[-1],'/' ' maturity :', maturity[-1]+1)
#print(FL.R2OOS(y_true.iloc[:,-1].values, predict_y_pls_m[-1,:,-1]))



#%% 
#  Ridge (fwd-rate only), (hyperparameter tuning) 

# hyper-parameter tuning in alphas = [.01, .05, .1, .5, 1, 2.5, 5, 10]
# can modify it in FunLib.Ridge_regression

predict_y_ridge = np.full([y_true.shape[0], y_true.shape[1]], np.nan)

for j in range(OOS_len):
    print(j)
        
    if -(OOS_len-j)+1 !=0:
        predict_y_ridge[j,:] = FL.Ridge_regression(Xexog.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values)
    else:
        predict_y_ridge[j,:] = FL.Ridge_regression(Xexog.values, Y.values)

R2OOS_ridge = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_ridge[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_ridge[:,j])
        
P_value_ridge = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    P_value_ridge[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_ridge[:,j])
        
        
predict_y_ridge = pd.DataFrame(predict_y_ridge)
predict_y_ridge.to_csv('predict_y_ridge.csv', index = False)

#%% 
#  Ridge (fwd-rate + macro), (hyperparameter tuning) 

# hyper-parameter tuning in alphas = [.01, .05, .1, .5, 1, 2.5, 5, 10]
# can modify it in FunLib.Ridge_regression

predict_y_ridge_m = np.full([y_true.shape[0], y_true.shape[1]], np.nan)

print('ridge_m')
for j in range(OOS_len):
    print(j)    
    if -(OOS_len-j)+1 !=0:
        predict_y_ridge_m[j,:] = FL.m_Ridge_regression(Xexog.iloc[:-(OOS_len-j)+1,:].values, X.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values)
    else:
        predict_y_ridge_m[j,:] = FL.m_Ridge_regression(Xexog.values, X.values, Y.values)


R2OOS_ridge_m = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_ridge_m[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_ridge_m[:,j])
        
P_value_ridge_m = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    P_value_ridge_m[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_ridge_m[:,j])
    
# We only print the case with the largest maturity 
#print('R2OOS for Ridge-regression with fwrd-rate /' ' maturity :', maturity[-1]+1)
#print(FL.R2OOS(y_true.iloc[:,-1].values, predict_y_ridge[:,-1]))
predict_y_ridge_m = pd.DataFrame(predict_y_ridge_m)
predict_y_ridge_m.to_csv('predict_y_ridge_m.csv', index = False)


#%% 
#  Lasso (fwd-rate), (hyperparameter tuning) 
# hyper-parameter tuning in alphas = [.01, .05, .1, .5, 1, 2.5, 5, 10]
# can modify it in FunLib.Lasso_regression

predict_y_lasso = np.full([y_true.shape[0], y_true.shape[1]], np.nan)

for j in range(OOS_len):
    print(j)    
    if -(OOS_len-j)+1 !=0:
        predict_y_lasso[j,:] = FL.Lasso_regression(Xexog.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values)
    else:
        predict_y_lasso[j,:] = FL.Lasso_regression(Xexog.values, Y.values)

R2OOS_lasso = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_lasso[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_lasso[:,j])
        
P_value_lasso = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    P_value_lasso[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_lasso[:,j])
    
predict_y_lasso = pd.DataFrame(predict_y_lasso)
predict_y_lasso.to_csv('predict_y_lasso.csv', index = False)
    
    

#%%
#   Lasso (fwd-rate + Macro variable), (hyperparameter tuning) 

# hyper-parameter tuning in alphas automatically 
# can modify it in FunLib.m_Lasso

predict_y_lasso_m = np.full([y_true.shape[0], y_true.shape[1]], np.nan)

print('lasso_m')
for j in range(OOS_len):
    print(j)    
    if -(OOS_len-j)+1 !=0:
        predict_y_lasso_m[j,:] = FL.m_Lasso(Xexog.iloc[:-(OOS_len-j)+1,:].values, X.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values)
    else:
        predict_y_lasso_m[j,:] = FL.m_Lasso(Xexog.values, X.values, Y.values)

R2OOS_lasso_m = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_lasso_m[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_lasso_m[:,j])
        
P_value_lasso_m = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    P_value_lasso_m[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_lasso_m[:,j])
    

predict_y_lasso_m = pd.DataFrame(predict_y_lasso_m)
predict_y_lasso_m.to_csv('predict_y_lasso_m.csv', index = False)

#%% Elastic_net (fwd) (hyperparameter tuning) 

predict_y_elas = np.full([y_true.shape[0], y_true.shape[1]], np.nan)

for j in range(OOS_len):
    print(j)    
    if -(OOS_len-j)+1 !=0:
        predict_y_elas[j,:] = FL.ElasticNet(Xexog.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values)
    else:
        predict_y_elas[j,:] = FL.ElasticNet(Xexog.values, Y.values)

R2OOS_elas = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_elas[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_elas[:,j])
        
P_value_elas = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    P_value_elas[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_elas[:,j])
    
    

predict_y_elas = pd.DataFrame(predict_y_elas)    
predict_y_elas.to_csv('predict_y_elas.csv', index = False)
    
    
    
#%%  Elastic_net  (fwd-rate + macro), (hyperparameter tuning) 

# we have 2 hyper-parameters (l1 ratio, alpha)
# can modify it in FunLib.m_ElasticNet

predict_y_elas_m = np.full([y_true.shape[0], y_true.shape[1]], np.nan)

print('elas_m')
for j in range(OOS_len):
    print(j)
        
    if -(OOS_len-j)+1 !=0:
        predict_y_elas_m[j,:] = FL.m_ElasticNet(Xexog.iloc[:-(OOS_len-j)+1,:].values, X.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values)
    else:
        predict_y_elas_m[j,:] = FL.m_ElasticNet(Xexog.values, X.values, Y.values)

R2OOS_elas_m = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_elas_m[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_elas_m[:,j])
        
P_value_elas_m = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    P_value_elas_m[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_elas_m[:,j])

predict_y_elas_m = pd.DataFrame(predict_y_elas_m)
predict_y_elas_m.to_csv('predict_y_elas_m.csv', index = False)



#%% Gradient boosting Regression Tree (fwd-rate)

predict_y_gbrt = np.full([y_true.shape[0], y_true.shape[1]], np.nan)

for j in range(OOS_len):
    print(j)    
    if -(OOS_len-j)+1 !=0:
        predict_y_gbrt[j,:] = FL.GBRT(Xexog.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values)
    else:
        predict_y_gbrt[j,:] = FL.GBRT(Xexog.values, Y.values)


R2OOS_gbrt = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_gbrt[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_gbrt[:,j])
        
P_value_gbrt = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    P_value_gbrt[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_gbrt[:,j])

predict_y_gbrt = pd.DataFrame(predict_y_gbrt)
predict_y_gbrt.to_csv('predict_y_gbrt.csv', index = False)


#%% Gradient boosting Regression Tree (fwd-rate +macro)

predict_y_gbrt_m = np.full([y_true.shape[0], y_true.shape[1]], np.nan)

print('gbrt_m')
for j in range(OOS_len):
    print(j)    
    if -(OOS_len-j)+1 !=0:
        predict_y_gbrt_m[j,:] = FL.m_GBRT(Xexog.iloc[:-(OOS_len-j)+1,:].values, X.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values)
    else:
        predict_y_gbrt_m[j,:] = FL.m_GBRT(Xexog.values, X.values, Y.values)


R2OOS_gbrt_m = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_gbrt_m[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_gbrt_m[:,j])
        
P_value_gbrt_m = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    P_value_gbrt_m[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_gbrt_m[:,j])

predict_y_gbrt_m = pd.DataFrame(predict_y_gbrt_m)
predict_y_gbrt_m.to_csv('predict_y_gbrt_m.csv', index = False)



#%%  Random-forest  (fwd-rate), (hyper-parameter tuning)

# consider 1(n_estimators), 2(max_depth), 3(max_features) for hyper-parameter
# use Gridsearch for hyperparameter tuning  
# can modify it in FunLib.m_h_RandomForest


predict_y_rf = np.full([y_true.shape[0], y_true.shape[1]], np.nan)

for j in range(OOS_len):
    print(j)
    if -(OOS_len-j)+1 !=0:
        predict_y_rf[j,:] = FL.RandomForest(Xexog.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values)
    else:
        predict_y_rf[j,:] = FL.RandomForest(Xexog.values, Y.values)

R2OOS_rf = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_rf[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_rf[:,j])
        
P_value_rf = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    P_value_rf[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_rf[:,j])


predict_y_rf = pd.DataFrame(predict_y_rf)
predict_y_rf.to_csv('predict_y_rf.csv', index = False)

#%%  Random-forest  (fwd-rate + macro), (hyper-parameter tuning)

# consider 1(n_estimators), 2(max_depth), 3(max_features) for hyper-parameter
# use Gridsearch for hyperparameter tuning 
# can modify it in FunLib.m_h_RandomForest


predict_y_rf_m = np.full([y_true.shape[0], y_true.shape[1]], np.nan)

for j in range(OOS_len):
    print(j)
    if -(OOS_len-j)+1 !=0:
        predict_y_rf_m[j,:] = FL.m_RandomForest(Xexog.iloc[:-(OOS_len-j)+1,:].values, X.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values)
    else:
        predict_y_rf_m[j,:] = FL.m_RandomForest(Xexog.values, X.values, Y.values)

R2OOS_rf_m = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_rf_m[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_rf_m[:,j])
        
P_value_rf_m = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    P_value_rf_m[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_rf_m[:,j])
    
predict_y_rf_m = pd.DataFrame(predict_y_rf_m)
predict_y_rf_m.to_csv('predict_y_rf_m.csv', index = False)


#%%  Neural-Net, fwd only, Figure2 pare 12 in BBT(2020)

# Refer to Figure2 page12 in BBT(2020)
    
# Use mini-batch, drop last (if 1 training set left in a batch while training, BatchNormalization is impossible)
# Use early-stopping with calculating MSE in validation set after 1-epoch training 

archi = [5,5]
epoch = 200
dropout_prob = 0.5
weight_decay = [1e-5, 1e-4, 0.001, 0.01, 0.05, 0.1, 0.5]

predict_y_temp = np.full([len(weight_decay), y_true.shape[0], y_true.shape[1]], np.nan)
predict_y_NN_fwd = np.full([y_true.shape[0], y_true.shape[1]], np.nan)

for j in range(OOS_len):
    print(j)
    val_loss_collect = []  
    for k in range(len(weight_decay)): 
        if -(OOS_len-j)+1 !=0:
            predict_y_temp[k,j,:], min_val_loss = FL.NN_fwd(Xexog.iloc[:-(OOS_len-j)+1,:].values, Y.iloc[:-(OOS_len-j)+1,:].values,
                                                            archi, epoch, dropout_prob, weight_decay[k])
        else:
            predict_y_temp[k,j,:],  min_val_loss = FL.NN_fwd(Xexog.values, Y.values, archi, epoch, dropout_prob, weight_decay[k])
        
        val_loss_collect.append(min_val_loss)
    
    chosen_index = np.argmin(val_loss_collect)
    print(chosen_index)
    
    predict_y_NN_fwd[j,:] = predict_y_temp[chosen_index,j,:]

R2OOS_NN_fwd = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_NN_fwd[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_NN_fwd[:,j])
        
P_value_NN_fwd = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    P_value_NN_fwd[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_NN_fwd[:,j])

predict_y_NN_fwd = pd.DataFrame(predict_y_NN_fwd)
predict_y_NN_fwd.to_csv('predict_y_NN_fwd.csv', index = False)


#%%

#  Neural-Net, Figure3-(a) in BBT(2020), macro + fwd direct in the last layer

# X,Xexog,Y are np.arraay value
# archi is the # of neurons in hidden layers for macro variables (list) ex) [32,16]
# drop-out is for fast training ex)0.25

# After the hidden layers(archi), the (macro) outcome and fwd-rate are linearly combined to output layer
# Refer to Figure3-(a) page13 in BBT(2020)
# If archi is [32, 16], 128(macro variables) > 32 > 16 + 10(fwd-rate) (=26) > 1~9(excess return)
    
# Use mini-batch, drop last (since if 1 training set left, BatchNormalization is impossible)
# Use early-stopping with calculating MSE in validation set after 1-epoch training 

archi = [32,16,8]
epoch = 500
dropout_prob = 0.5
weight_decay = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]

predict_y_temp = np.full([len(weight_decay), y_true.shape[0], y_true.shape[1]], np.nan)
predict_y_NN_m_fwdlast = np.full([y_true.shape[0], y_true.shape[1]], np.nan)
                        
for j in range(OOS_len):
    print(j)
    val_loss_collect = []    
    for k in range(len(weight_decay)):    
        if -(OOS_len-j)+1 !=0:
            predict_y_temp[k,j,:], min_val_loss = FL.m_NN_fwd_last(Xexog.iloc[:-(OOS_len-j)+1,:].values, X.iloc[:-(OOS_len-j)+1,:].values, 
                                                                Y.iloc[:-(OOS_len-j)+1,:].values, archi, epoch, dropout_prob, weight_decay[k])
        else:
            predict_y_temp[k,j,:], min_val_loss = FL.m_NN_fwd_last(Xexog.values, X.values, Y.values, archi, epoch, dropout_prob, weight_decay[k])
        
        val_loss_collect.append(min_val_loss)    
    
    chosen_index = np.argmin(val_loss_collect)
    print(chosen_index)
    predict_y_NN_m_fwdlast[j,:]=predict_y_temp[chosen_index,j,:]
        
R2OOS_NN_m_fwdlast = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_NN_m_fwdlast[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_NN_m_fwdlast[:,j])
        
P_value_NN_m_fwdlast = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    P_value_NN_m_fwdlast[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_NN_m_fwdlast[:,j])
    
predict_y_NN_m_fwdlast = pd.DataFrame(predict_y_NN_m_fwdlast)
predict_y_NN_m_fwdlast.to_csv('predict_y_NN_m_fwdlast.csv', index = False)



#%%

#  Neural-Net, Figure3-(b) in BBT(2020), macro + fwd direct in the last layer

# X,Xexog,Y are np.arraay value
# archi is the # of neurons in hidden layers for macro variables (list) ex) [32,16]
# drop-out is for fast training ex)0.25

# After the hidden layers(archi), the (macro) outcome and fwd-rate are linearly combined to output layer
# Refer to Figure3-(b) page13 in BBT(2020)
# If archi is [32, 16], 128(macro variables) > 32 > 16 + 10(fwd-rate) (=26) > 1~9(excess return)
    
# Use mini-batch, drop last (since if 1 training set left, BatchNormalization is impossible)
# Use early-stopping with calculating MSE in validation set after 1-epoch training 

archi = [32,16]
epoch = 300
dropout_prob = 0.5
weight_decay = [1e-5, 1e-4, 0.001, 0.01, 0.05, 0.1, 0.5]

predict_y_temp = np.full([len(weight_decay), y_true.shape[0], y_true.shape[1]], np.nan)
predict_y_NN_m_fwd_hidden = np.full([y_true.shape[0], y_true.shape[1]], np.nan)
                        
for j in range(OOS_len):
    print(j)
    val_loss_collect = []    
    for k in range(len(weight_decay)):    
        if -(OOS_len-j)+1 !=0:
            predict_y_temp[k,j,:], min_val_loss = FL.m_NN_fwd_hidden(Xexog.iloc[:-(OOS_len-j)+1,:].values, X.iloc[:-(OOS_len-j)+1,:].values, 
                                                                Y.iloc[:-(OOS_len-j)+1,:].values, archi, epoch, dropout_prob, weight_decay[k])
        else:
            predict_y_temp[k,j,:], min_val_loss = FL.m_NN_fwd_hidden(Xexog.values, X.values, Y.values, archi, epoch, dropout_prob, weight_decay[k])
        
        val_loss_collect.append(min_val_loss)    
    
    chosen_index = np.argmin(val_loss_collect)
    print(chosen_index)
    predict_y_NN_m_fwd_hidden[j,:]=predict_y_temp[chosen_index,j,:]
        
R2OOS_NN_m_fwd_hidden = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_NN_m_fwd_hidden[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_NN_m_fwd_hidden[:,j])
        
P_value_NN_m_fwd_hidden = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    P_value_NN_m_fwd_hidden[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_NN_m_fwd_hidden[:,j])
    
predict_y_NN_m_fwd_hidden = pd.DataFrame(predict_y_NN_m_fwd_hidden)
predict_y_NN_m_fwd_hidden.to_csv('predict_y_NN_m_fwd_hidden.csv', index = False)





#%% Diebold - Mariano test (yield vs yield+macro) (same model)

print('DM test, PCA, p-value')
print(FL.DM_test(y_true.to_numpy(),predict_y_pca[0,:,:],predict_y_pca_m[0,:,:]))


print('DM test, PLS, p-value')
print(FL.DM_test(y_true.to_numpy(),predict_y_pls[0,:,:],predict_y_pls_m[0,:,:]))


predict_y_lasso = pd.read_csv('predict_y_lasso.csv').to_numpy()
predict_y_lasso_m = pd.read_csv('predict_y_lasso_m.csv').to_numpy()
print('DM test, Lasso, p-value')
print(FL.DM_test(y_true.to_numpy(),predict_y_lasso,predict_y_lasso_m))


predict_y_ridge = pd.read_csv('predict_y_ridge.csv').to_numpy()
predict_y_ridge_m = pd.read_csv('predict_y_ridge_m.csv').to_numpy()
print('DM test, Ridge, p-value')
print(FL.DM_test(y_true,predict_y_ridge,predict_y_ridge_m))


predict_y_elas = pd.read_csv('predict_y_elas.csv').to_numpy()
predict_y_elas_m = pd.read_csv('predict_y_elas_m.csv').to_numpy()
print('DM test, elastic-net, p-value')
print(FL.DM_test(y_true,predict_y_elas,predict_y_elas_m))


predict_y_gbrt = pd.read_csv('predict_y_gbrt.csv').to_numpy()
predict_y_gbrt_m = pd.read_csv('predict_y_gbrt_m.csv').to_numpy()
print('DM test, GBRT, p-value')
print(FL.DM_test(y_true,predict_y_gbrt,predict_y_gbrt_m))


predict_y_rf = pd.read_csv('predict_y_rf.csv').to_numpy()
predict_y_rf_m = pd.read_csv('predict_y_rf_m.csv').to_numpy()
print('DM test, Random Forest, p-value')
print(FL.DM_test(y_true,predict_y_rf,predict_y_rf_m))


predict_y_NN_fwd = pd.read_csv('predict_y_NN_fwd.csv').to_numpy()
predict_y_NN_m_fwdlast = pd.read_csv('predict_y_NN_m_fwdlast.csv').to_numpy()
print('DM test, NN, p-value')
print(FL.DM_test(y_true,predict_y_NN_fwd,predict_y_NN_m_fwdlast))

predict_y_NN_fwd = pd.read_csv('predict_y_NN_fwd.csv').to_numpy()
predict_y_NN_m_fwd_hidden = pd.read_csv('predict_y_NN_m_fwd_hidden.csv').to_numpy()
print('DM test, NN (with yield hidden), p-value')
print(FL.DM_test(y_true,predict_y_NN_fwd,predict_y_NN_m_fwd_hidden))

#%% Diebold - Mariano (yield only) (use same information, check whether non-linearity helpful)

# pca, pls, lasso, ridge, elastic-net, GBRT, RF, NN 
num_model_yield = 8
p_value_dm_yield = np.full([num_model_yield, num_model_yield], np.nan)
predict_y_yield = np.concatenate([predict_y_pca[0,:,:], predict_y_pls[0,:,:], predict_y_lasso, predict_y_ridge,
                                  predict_y_elas, predict_y_gbrt, predict_y_rf, predict_y_NN_fwd], axis=1)

for i in range(num_model_yield-1):
    for j in range(i+1,num_model_yield):
        p_value_dm_yield[i,j] = FL.DM_test(y_true,predict_y_yield[:,(6*i):(6*i+6)],predict_y_yield[:,(6*j):(6*j+6)])
        
        
#%% Diebold - Mariano (yield+macro) (use same information, check whether non-linearity helpful)

# pca, pls, lasso, ridge, elastic-net, GBRT, RF, NN 
num_model_macro = 9
p_value_dm_macro = np.full([num_model_macro, num_model_macro], np.nan)
predict_y_macro = np.concatenate([predict_y_pca_m[0,:,:], predict_y_pls_m[0,:,:], predict_y_lasso_m, predict_y_ridge_m,
                                  predict_y_elas_m, predict_y_gbrt_m, predict_y_rf_m, predict_y_NN_m_fwdlast, predict_y_NN_m_fwd_hidden], axis=1)

for i in range(num_model_macro-1):
    for j in range(i+1,num_model_macro):
        p_value_dm_macro[i,j] = FL.DM_test(y_true,predict_y_macro[:,(6*i):(6*i+6)],predict_y_macro[:,(6*j):(6*j+6)])


#%% Clark and west (already calculated above) (just recalculte when load predicted y)

# pca
R2OOS_pca = np.full([len(num_pca), len(maturity)], np.nan)
for i in range(len(num_pca)):
    for j in range(len(maturity)):        
        R2OOS_pca[i,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_pca[i,:,j])        
P_value_pca = np.full([len(num_pca), len(maturity)], np.nan)
for i in range(len(num_pca)):
    for j in range(len(maturity)):        
        P_value_pca[i,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_pca[i,:,j])
        

R2OOS_pca_m = np.full([len(num_pca_m), len(maturity)], np.nan)
for i in range(len(num_pca_m)):
    for j in range(len(maturity)):        
        R2OOS_pca_m[i,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_pca_m[i,:,j])
        
P_value_pca_m = np.full([len(num_pca_m), len(maturity)], np.nan)
for i in range(len(num_pca_m)):
    for j in range(len(maturity)):        
        P_value_pca_m[i,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_pca_m[i,:,j])
        


# pls      
R2OOS_pls = np.full([len(num_pls), len(maturity)], np.nan)
for i in range(len(num_pls)):
    for j in range(len(maturity)):        
        R2OOS_pls[i,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_pls[i,:,j])
        
P_value_pls = np.full([len(num_pls), len(maturity)], np.nan)
for i in range(len(num_pls)):
    for j in range(len(maturity)):        
        P_value_pls[i,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_pls[i,:,j])
        
R2OOS_pls_m = np.full([len(num_pls_m), len(maturity)], np.nan)
for i in range(len(num_pls_m)):
    for j in range(len(maturity)):        
        R2OOS_pls_m[i,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_pls_m[i,:,j])
        
P_value_pls_m = np.full([len(num_pls_m), len(maturity)], np.nan)
for i in range(len(num_pls_m)):
    for j in range(len(maturity)):        
        P_value_pls_m[i,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_pls_m[i,:,j])
        
        
# lasso 
R2OOS_lasso = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    R2OOS_lasso[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_lasso[:,j])
        
P_value_lasso = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    P_value_lasso[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_lasso[:,j])
    

R2OOS_lasso_m = np.full([1, len(maturity)], np.nan)

for j in range(len(maturity)):        
    R2OOS_lasso_m[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_lasso_m[:,j])   
    
P_value_lasso_m = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    P_value_lasso_m[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_lasso_m[:,j])
    
    
# Ridge
R2OOS_ridge = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    R2OOS_ridge[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_ridge[:,j])
        
P_value_ridge = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    P_value_ridge[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_ridge[:,j])
    
    
R2OOS_ridge_m = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    R2OOS_ridge_m[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_ridge_m[:,j])
        
P_value_ridge_m = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    P_value_ridge_m[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_ridge_m[:,j])
    
    
# elastic net
R2OOS_elas = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    R2OOS_elas[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_elas[:,j])
        
P_value_elas = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    P_value_elas[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_elas[:,j])
    
    
R2OOS_elas_m = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    R2OOS_elas_m[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_elas_m[:,j])
        
P_value_elas_m = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    P_value_elas_m[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_elas_m[:,j])
    
    
# GBRT
R2OOS_gbrt = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    R2OOS_gbrt[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_gbrt[:,j])
        
P_value_gbrt = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    P_value_gbrt[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_gbrt[:,j])
    
    
R2OOS_gbrt_m = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    R2OOS_gbrt_m[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_gbrt_m[:,j])
        
P_value_gbrt_m = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    P_value_gbrt_m[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_gbrt_m[:,j])
    
    
# Random forest
R2OOS_rf = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    R2OOS_rf[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_rf[:,j])
        
P_value_rf = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    P_value_rf[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_rf[:,j])
    
    
R2OOS_rf_m = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    R2OOS_rf_m[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_rf_m[:,j])
        
P_value_rf_m = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    P_value_rf_m[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_rf_m[:,j])
    
    
    
# Neural network (fwd only)
R2OOS_NN_fwd = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    R2OOS_NN_fwd[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_NN_fwd[:,j])
        
P_value_NN_fwd = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    P_value_NN_fwd[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_NN_fwd[:,j])
    
    
    
# Neural network (macro+ fwd direct)
R2OOS_NN_m_fwdlast = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    R2OOS_NN_m_fwdlast[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_NN_m_fwdlast[:,j])
        
P_value_NN_m_fwdlast = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    P_value_NN_m_fwdlast[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_NN_m_fwdlast[:,j])
    
    
# Neural network (macro + fwd 1 hidden)
R2OOS_NN_m_fwd_hidden = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    R2OOS_NN_m_fwd_hidden[0,j]=FL.R2OOS(y_true.iloc[:,j].values, predict_y_NN_m_fwd_hidden[:,j])
        
P_value_NN_m_fwd_hidden = np.full([1, len(maturity)], np.nan)
for j in range(len(maturity)):        
    P_value_NN_m_fwd_hidden[0,j]=FL.RSZ_Signif(y_true.iloc[:,j].values, predict_y_NN_m_fwd_hidden[:,j])