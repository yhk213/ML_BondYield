# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 18:02:36 2021

@author: yhkim
"""

import numpy as np
import pandas as pd 
import sklearn
import torch
import multiprocessing as mp
import os

"""
Load yield & macro data 
import  1971.08~~2019.12 (monthly)
"""

# ==============================================================================
# Load  yield,  t= 1/12,2/12, ...580/12 (1971.08~~2019.12)(monthly), maturity=1,2,3,4,5,6,7,8,9,10(yearly)
# ==============================================================================


#Read yield data, by using skiprows delete data source etc...
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
yield_data = indexed_y.iloc[122:,yearly_index] # 123 : 1961.06~1971.07제외

del y_raw_data
del indexed_y


# ===================================================================================
#                Load   Macro-data,   1971.09~~2019.12 (monthly),  Na > average 
# ===================================================================================

# read macro data 
m_raw_data = pd.read_csv('current.csv')
m_raw_data = m_raw_data.iloc[1:,:]   # raw 0 : transform method > delete

# 매월 1일 기준 data > 하루 빼서 yield data와 맞춤 & 년/월/일만 추출
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
#             yield > forwad & excess-return data 
# =========================================================================



"""
from yield curve data, construct forward rate & excess return 
Note that yield data is annually & conti-compounded.
yld, forward-rate, excess-return are all described in annulized decimal notation  
"""

time_idx = np.linspace(1/12,1/12*yield_data.shape[0],yield_data.shape[0])    # t= 1/12, 2/12 3/12..

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




# =========================================================================
#           R2oos 
# =========================================================================

def R2OOS(y_true, y_forecast):
    import numpy as np

    # Compute conidtional mean forecast
    y_condmean = np.divide(y_true.cumsum(), (np.arange(y_true.size)+1))

    # lag by one period
    y_condmean = np.insert(y_condmean, 0, np.nan)    # size +1, nan추가
    y_condmean = y_condmean[:-1]                     # 맨뒤에 날리고
    y_condmean[np.isnan(y_forecast)] = np.nan        # isnan -> Ture/false, forecast 값이 없는 것 true condmean도 nan으로  

    # Sum of Squared Resids
    SSres = np.nansum(np.square(y_true-y_forecast))
    SStot = np.nansum(np.square(y_true-y_condmean))

    return 1-SSres/SStot




"""
# =========================================================================
#                   PCR : PCA & regression  (fwd-rate only)
#                   forecasting  starts 1990.01, using expanding windows 
# =========================================================================

# yield-only 

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler


num_pca = 3  # 3,5,10   # of pca components 
maturity_n = 10 #2,3,4,5,7,10   bond maturity when buying it

Xexog = fwd_rate.iloc[:-12,:]   # feature(forward-rate) : 1971.8 ~ 2018.12    size: 569*10
excess_return = xr_rate.iloc[12:,:]  # y_value : 1972.8~ 2019.12   size: 569*10   0-th column : data none

y_true = excess_return.iloc[209:, maturity_n-1].values  #forecasting starts 1990.1 > 1990.1~2019.12

y_forecast_pca=[]   

# Note that 1971.08~1988.12 : 209
for i in range(209,Xexog.shape[0]):       # traing using ~88.12 > using 89.01 forward and forecast 90.01 
#for i in range(208,209):
#for i in range(Xexog.shape[0],Xexog.shape[0]+1):
    pca = PCA(n_components = num_pca)
    
    #for scaling training_set

    Xexog_scaler = StandardScaler()
    Xexog_scaled = Xexog_scaler.fit_transform(Xexog.iloc[:i,:])   
    principalComponents = pca.fit_transform(Xexog_scaled)
    
    x_new_scaled = Xexog_scaler.transform(fwd_rate.iloc[i,:].values.reshape(1,-1))  #start 1989.01  
    x_new = pca.transform(x_new_scaled)
    
  
    # linear-regression with principal component 

    y = excess_return.iloc[:i,maturity_n-1]
        
    line_fitter = LinearRegression()
    line_fitter.fit(principalComponents, y)
    y_forecast_pca.append(line_fitter.predict(x_new))
    
y_forecast_pca = np.array(y_forecast_pca).reshape(-1)


print('R2oos with pca' + ' (#pca compo) ' +str(num_pca) + ' (maturity) ' +str(maturity_n)+' : ' +str(R2OOS(y_true, y_forecast_pca)))



# =========================================================================
#                   PLS : Partial least square (fwd-rate only)
#                   forecasting  starts 1990.01, using expanding windows 
# =========================================================================

from sklearn.cross_decomposition import PLSRegression

num_pls = num_pca   # for comparison 


y_forecast_pls=[]  
for i in range(209,Xexog.shape[0]): 
    pls = PLSRegression(n_components = num_pls)
    
    
    pls.fit(Xexog.iloc[:i,:], excess_return.iloc[:i,maturity_n-1])
    y_forecast_pls.append(pls.predict(Xexog.iloc[i,:].values.reshape(1,-1)))
    
y_forecast_pls = np.array(y_forecast_pls).reshape(-1)  

print('R2oos with pls' + ' (#pca compo) ' +str(num_pca) + ' (maturity) ' +str(maturity_n)+' : ' +str(R2OOS(y_true, y_forecast_pls)))





# =========================================================================
#                   PCR : PCA & regression  (fwd-rate + macro variale)
#                   forecasting  starts 1990.01, using expanding windows 
# =========================================================================


num_pca_macro = 8     # of pca components 
maturity_n_macro = 10 #2,3,4,5,7,10   bond maturity when buying it

#Xexog = fwd_rate.iloc[:-12,:]   # feature(forward-rate) : 1971.8 ~ 2018.12    size: 569*10
X = macro_data.iloc[:-12,:]     # maro data : 1971.8 ~ 2018.12  size 569*128
Xexog_X = pd.concat([Xexog, X],axis=1)   # Xexog + X : size 569*138


#excess_return = xr_rate.iloc[12:,:]  # y_value : 1972.8~ 2019.12   size: 569*10   0-th column : data none
y_true_macro = excess_return.iloc[209:, maturity_n_macro - 1].values  #forecasting starts 1990.1 > 1990.1~2019.12

y_forecast_pca_macro=[]   

# Note that 1971.08~1988.12 : 209
for i in range(209,Xexog_X.shape[0]):       # traing using ~88.12 > using 89.01 data and forecast 90.01 
    pca_macro = PCA(n_components = num_pca_macro)
    
    #for scaling training_set
    Xexog_X_scaler = StandardScaler()
    Xexog_X_scaled = Xexog_X_scaler.fit_transform(Xexog_X.iloc[:i,:])   
    principalComponents_macro = pca_macro.fit_transform(Xexog_X_scaled)
        
    x_new_macro_scaled = Xexog_X_scaler.transform(Xexog_X.iloc[i,:].values.reshape(1,-1))  #start 1989.01  
    x_new_macro = pca_macro.transform(x_new_macro_scaled)
    

    # linear-regression with principal component 
    y_macro = excess_return.iloc[:i,maturity_n_macro-1]
        
    line_macro_fitter = LinearRegression()
    line_macro_fitter.fit(principalComponents_macro, y_macro)
    y_forecast_pca_macro.append(line_macro_fitter.predict(x_new_macro))
    
y_forecast_pca_macro = np.array(y_forecast_pca_macro).reshape(-1)

print('R2oos with pca, with macro' + ' (#pca compo) ' +str(num_pca_macro) + ' (maturity) ' +str(maturity_n_macro)+' : ' +str(R2OOS(y_true_macro, y_forecast_pca_macro)))




# =========================================================================
#                   PLS : Partial least square (fwd-rate + Macro variable)
#                   forecasting  starts 1990.01, using expanding windows 
# =========================================================================

from sklearn.cross_decomposition import PLSRegression

num_pls_macro = num_pca_macro   # for comparison 


y_forecast_pls_macro=[]  
for i in range(209,Xexog.shape[0]): 
    pls = PLSRegression(n_components = num_pls_macro)
    
    
    pls.fit(Xexog.iloc[:i,:], excess_return.iloc[:i,maturity_n-1])
    y_forecast_pls_macro.append(pls.predict(Xexog.iloc[i,:].values.reshape(1,-1)))
    
y_forecast_pls_macro = np.array(y_forecast_pls_macro).reshape(-1)  

print('R2oos with pls, with macro' + ' (#pca compo) ' +str(num_pls_macro) + ' (maturity) ' +str(maturity_n_macro)+' : ' +str(R2OOS(y_true, y_forecast_pls_macro)))
"""



# =========================================================================
#                   PCR : PCA & regression  (fwd-rate only)
#                   forecasting  starts 1990.01, using expanding windows 
# =========================================================================

import FunLib as FL


Xexog = fwd_rate.iloc[:-12,:]   # feature(forward-rate) : 1971.8 ~ 2018.12    size: 569*10
X = macro_data.iloc[:-12,:]     # maro data : 1971.8 ~ 2018.12  size 569*128
y= xr_rate.iloc[12:,1:]
y_forecast_elas= np.full([360, y.shape[1]], np.nan)

# Note that 1971.08~1989.1 : 210
for i in range(209,X.shape[0]):    
    Ypred = FL.ElasticNet_Exog_Plain(X.iloc[:i+1,:].values, Xexog.iloc[:i+1,:].values, y.iloc[:i+1,:].values)
    y_forecast_elas[i-209,:] = Ypred