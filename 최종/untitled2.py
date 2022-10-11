# -*- coding: utf-8 -*-
"""
Created on Tue Jan 19 22:06:50 2021

@author: dudgo
"""


#


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

# 1971.09~~2019.12 까지 추출
macro_data = indexed_m.iloc[153:733,:]


# some variables have Na, > fill Na with average 
na = pd.isna(macro_data).sum()
macro_data = macro_data.fillna(macro_data.mean())

del m_raw_data
del indexed_m



# =========================================================================
#             yield > forwad / excess-return data 
# =========================================================================



"""
from yield curve data, construct forward rate & excess return 
Note that yield data is annually & conti-compounded.
yld, forward-rate, excess-return are all described in annulized decimal notation  
"""

time_idx = np.linspace(1/12,1/12*yield_data.shape[0],yield_data.shape[0])    # t= 1/12, 2/12 3/12...
maturity_idx = np.linspace(1/12,10,120)                                      # n=1/12, 2/12, 3/12 ...

yld = 1/100*yield_data.values    # to np.ndarray
log_p = np.zeros(yld.shape)


# log(price) 
for i in range(log_p.shape[1]):
    log_p[:,i] = -(i+1)/12 * yld[:,i]
    
    
# forward rate,   Note that the first column of forward-rate matrix is the same as that of yield curve 
fwd_rate = np.zeros(log_p.shape)

for i in range(log_p.shape[1]):
    if i ==0:
        fwd_rate[:,i] = yld[:,i]
    else:
        fwd_rate[:,i] = 12*(log_p[:,i-1] - log_p[:,i])    #annualized 
        
fwd_rate = pd.DataFrame(data=fwd_rate, columns= col_name[1:121], index = macro_data.index)
    
        
# excess return , the 1st column & row are all 0 
xr_rate = np.zeros(log_p.shape)

for i in range(1,log_p.shape[0]):
    for j in range(1,log_p.shape[1]):
        xr_rate[i,j] = 12*(log_p[i,j-1] - log_p[i-1,j]) - yld[i-1,0]     #annualized
        

xr_rate = pd.DataFrame(data=xr_rate, columns= col_name[1:121], index = macro_data.index)




        

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




# =========================================================================
#                   PCR : PCA & regression  (fwd-rate only / fwd-rate + macro)
# =========================================================================

# yield-only 

from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

num_pca = [3]   # 3,5,10

for num in num_pca: 
    # no normalization with only fwd-data 
    pca = PCA(n_components = num)
    principalComponents = pca.fit_transform(fwd_rate.iloc[:-1,:])    

y=xr_rate.iloc[1:,12]

line_fitter = LinearRegression()
line_fitter.fit(principalComponents, y)
print(line_fitter.coef_)

line_fitter.predict(principalComponents)













    
    for j in [0]:                # [0,1,2,3,5,8]:   # n=2,3,4,5,7,10 in paper 18-page
        y = excess_return.iloc[:i,j]
        
        line_fitter = LinearRegression()
        line_fitter.fit(principalComponents, y)
        
        y_forecast_pca.append(line_fitter.predict(fwd_rate.iloc[i+1,:]))