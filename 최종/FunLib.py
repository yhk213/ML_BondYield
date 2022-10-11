# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 12:07:15 2021

@author: yhkim
"""


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
#                   PCR : PCA & regression  (fwd-rate only)
# =========================================================================

def Pca_regression(Xexog,Y,numpc):
    # numpc : # of principal component  
    # Xeog : fwd-data only (without macro-var)
    # y : excess-return 
    # no need to split train/validation in PCA
    
    import numpy as np 
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # Split data into training and test
    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]
    
    Xexog_test = Xexog[-1,:].reshape(1,-1)
       
    # Scale Inputs for Training
    Xexog_scaler = StandardScaler()
    Xexog_train_scaled = Xexog_scaler.fit_transform(Xexog_train)
    
    Xexog_test_scaled = Xexog_scaler.transform(Xexog_test)
    
    #PCA
    pca = PCA(n_components = numpc)
    principalComponents = pca.fit_transform(Xexog_train_scaled)
    
    Xexog_test_weighted = pca.transform(Xexog_test_scaled)
    
    # Regress & Predict
    Ypred = np.full([1, Y_train.shape[1]],np.nan)
    # Loop over maturities
    for i in range(Y_train.shape[1]):

        line_fitter = LinearRegression()
        line_fitter.fit(principalComponents, Y_train[:,i])
        Ypred[0,i]=line_fitter.predict(Xexog_test_weighted)

    return Ypred


    
# =========================================================================
#                   PLS : Partial least square (fwd-rate only)
# ========================================================================= 
    
def Pls_regression(Xexog,Y,numpls):
    # numpls : # of component  
    # Xeog : fwd-data only (without macro-var)
    # y : excess-return 
    # no need to split train/validation in PLS
    
    import numpy as np
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    
    # Split data into training and test
    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]
    
    Xexog_test = Xexog[-1,:].reshape(1,-1)
    
    # Scale Inputs for Training
    Xexog_scaler = StandardScaler()
    Xexog_train_scaled = Xexog_scaler.fit_transform(Xexog_train)
    
    Xexog_test_scaled = Xexog_scaler.transform(Xexog_test)
    
    # PLS regression 
    Ypred = np.full([1, Y_train.shape[1]],np.nan)
    # Loop over maturities
    for i in range(Y_train.shape[1]):

        pls = PLSRegression(n_components = numpls)
        pls.fit(Xexog_train_scaled, Y_train[:,i])
        Ypred[0,i]=pls.predict(Xexog_test_scaled)

    return Ypred


# =========================================================================
#                   PCR : PCA & regression  (fwd-rate + macro)
# =========================================================================

def m_Pca_regression(Xexog,X,Y,numpc):
    # numpc : # of principal component  
    # Xeog : fwd-data only (without macro-var)
    # X : macro varaible(128-variable)
    # y : excess-return 
    # no need to split train/validation in PCA
    
    import numpy as np 
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    
    # Split data into training and test
    X_train = X[:-1,:]
    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]
    X_test = X[-1,:]
    X_test = X_test.reshape(1, -1)
    Xexog_test = Xexog[-1,:]
    Xexog_test = Xexog_test.reshape(1, -1)
    
    
    # Scale Inputs for Training 
    Xscaler_train = StandardScaler()
    X_train = Xscaler_train.fit_transform(X_train)
    X_test = Xscaler_train.transform(X_test)

    Xexogscaler_train = StandardScaler()
    Xexog_train = Xexogscaler_train.fit_transform(Xexog_train)
    Xexog_test = Xexogscaler_train.transform(Xexog_test)
    
    X_Xexog_train = np.concatenate((Xexog_train, X_train),axis = 1)
    X_Xexog_test = np.concatenate((Xexog_test, X_test),axis = 1)
    
    #PCA
    pca = PCA(n_components = numpc)
    principalComponents = pca.fit_transform(X_Xexog_train)
    
    X_Xexog_test_weighted = pca.transform(X_Xexog_test)
    
    # Regress & Predict
    Ypred = np.full([1, Y_train.shape[1]],np.nan)
    # Loop over maturities
    for i in range(Y_train.shape[1]):

        line_fitter = LinearRegression()
        line_fitter.fit(principalComponents, Y_train[:,i])
        Ypred[0,i]=line_fitter.predict(X_Xexog_test_weighted)

    return Ypred



# =========================================================================
#                   PLS : Partial least square (fwd-rate + macro)
# ========================================================================= 
    
def m_Pls_regression(Xexog,X,Y,numpls):
    # numpls : # of component  
    # Xeog : fwd-data only (without macro-var)
    # X : macro varaible(128-variable)
    # y : excess-return 
    # no need to split train/validation in PLS
    
    import numpy as np
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.preprocessing import StandardScaler
    
    # Split data into training and test
    X_train = X[:-1,:]
    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]
    
    X_test = X[-1,:]
    X_test = X_test.reshape(1, -1)
    Xexog_test = Xexog[-1,:]
    Xexog_test = Xexog_test.reshape(1, -1)
    
    
    # Scale Inputs for Training
    Xscaler_train = StandardScaler()
    X_train = Xscaler_train.fit_transform(X_train)
    X_test = Xscaler_train.transform(X_test)

    Xexogscaler_train = StandardScaler()
    Xexog_train = Xexogscaler_train.fit_transform(Xexog_train)
    Xexog_test = Xexogscaler_train.transform(Xexog_test)
    
    X_Xexog_train = np.concatenate((Xexog_train, X_train),axis = 1)
    X_Xexog_test = np.concatenate((Xexog_test, X_test),axis = 1)
    
    # PLS regression 
    Ypred = np.full([1, Y_train.shape[1]],np.nan)
    # Loop over maturities
    for i in range(Y_train.shape[1]):

        pls = PLSRegression(n_components = numpls)
        pls.fit(X_Xexog_train, Y_train[:,i])
        Ypred[0,i]=pls.predict(X_Xexog_test)

    return Ypred



# =========================================================================
#                   Ridge regression (fwd-rate only) (hyperparameter tuning)
# ========================================================================= 

# if we don't input alphas = [.01, .05, .1, .5, 1, 2.5, 5, 10] in >> model = RidgeCV(....alphas = ...)
# alpha =[.1, 1, 10] is a default 

def Ridge_regression(Xexog,Y):  
    # Xeog : fwd-data only (without macro-var)
    # y : excess-return 
    
    import numpy as np
    from sklearn.linear_model import RidgeCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import PredefinedSplit
    
    # Split data into training and test
    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]
    
    Xexog_test = Xexog[-1,:].reshape(1,-1)
    
    # Scale Inputs for Training
    Xexog_scaler = StandardScaler()
    Xexog_train_scaled = Xexog_scaler.fit_transform(Xexog_train)
    
    Xexog_test_scaled = Xexog_scaler.transform(Xexog_test)
    

    # Construct Validation sample as last 15% of sample
    N_train = int(np.round(np.size(Xexog_train_scaled,axis=0)*0.85))
    N_val = np.size(Xexog_train_scaled,axis=0)-N_train
    test_fold =  np.concatenate(((np.full((N_train),-1),np.full((N_val),0))))
    ps = PredefinedSplit(test_fold.tolist())
    

    # Regress & Predict
    Ypred = np.full([1, Y_train.shape[1]],np.nan)
    
    # Loop over maturities
    for i in range(Y_train.shape[1]):
        model = RidgeCV(cv=ps,alphas = [.01, .05, .1, .5, 1, 2.5, 5, 10])
        model = model.fit(Xexog_train_scaled, Y_train[:,i])
        Ypred[0,i]=model.predict(Xexog_test_scaled)

    return Ypred





# =========================================================================
#                   Lasso  (fwd-rate + macro) (hyperparameter tuning)
# =========================================================================

# we 'can' input >> model = LassoCV(alphas=[.01, .1, .2, .5, 1, 1.5, 2, 2.5, 3, 5, 10] .... ) 
# if not, LassoCV search alpha automatically  > need more time 

def m_Lasso(Xexog,X,Y):
    # Xeog : fwd-data only (without macro-var)
    # X : macro varaible(128-variable)
    # y : excess-return 
    
    import numpy as np
    from sklearn.linear_model import LassoCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import PredefinedSplit

    # Split data into training and test
    X_train = X[:-1,:]
    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]
    X_test = X[-1,:]
    X_test = X_test.reshape(1, -1)
    Xexog_test = Xexog[-1,:]
    Xexog_test = Xexog_test.reshape(1, -1)

    # Scale Inputs for Training
    Xscaler_train = StandardScaler()
    X_train = Xscaler_train.fit_transform(X_train)
    X_test = Xscaler_train.transform(X_test)

    Xexogscaler_train = StandardScaler()
    Xexog_train = Xexogscaler_train.fit_transform(Xexog_train)
    Xexog_test = Xexogscaler_train.transform(Xexog_test)


    # Construct Validation sample as last 15% of sample
    N_train = int(np.round(np.size(X_train,axis=0)*0.85))
    N_val = np.size(X_train,axis=0)-N_train
    test_fold =  np.concatenate(((np.full((N_train),-1),np.full((N_val),0))))
    ps = PredefinedSplit(test_fold.tolist())
    

    # Regress & Predict
    Ypred = np.full([1, Y_train.shape[1]],np.nan)
    
    # Loop over maturities
    for i in range(Y_train.shape[1]):
        model = LassoCV(cv=ps, max_iter=1000, n_jobs=-1,
                             random_state=42)
        model = model.fit(np.concatenate((Xexog_train, X_train),axis=1),
                          Y_train[:,i])
        Ypred[0,i]=model.predict(np.concatenate((Xexog_test,X_test),axis=1))

    return Ypred




# =========================================================================
#                   Elastic-net  (fwd-rate + macro) (hyperparameter tuning)
# =========================================================================

# in elastic-net, we have 2 hyper-parameters (l1 ratio, alpha)
# we already set l1 ratio in function, >>> model = ElasticNetCV(....l1_ratio=[.1, .3, .5, .7, .9],...)
# but alpha is not designated >> so search automatically & need more time 

def m_ElasticNet(Xexog,X,Y):
    # Xeog : fwd-data only (without macro-var)
    # X : macro varaible(128-variable)
    # y : excess-return 
    
    import numpy as np
    from sklearn.linear_model import ElasticNetCV
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import PredefinedSplit

    # Split data into training and test
    X_train = X[:-1,:]
    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]
    X_test = X[-1,:]
    X_test = X_test.reshape(1, -1)
    Xexog_test = Xexog[-1,:]
    Xexog_test = Xexog_test.reshape(1, -1)

    # Scale Inputs for Training
    Xscaler_train = StandardScaler()
    X_train = Xscaler_train.fit_transform(X_train)
    X_test = Xscaler_train.transform(X_test)

    Xexogscaler_train = StandardScaler()
    Xexog_train = Xexogscaler_train.fit_transform(Xexog_train)
    Xexog_test = Xexogscaler_train.transform(Xexog_test)


    # Construct Validation sample as last 15% of sample
    N_train = int(np.round(np.size(X_train,axis=0)*0.85))
    N_val = np.size(X_train,axis=0)-N_train
    test_fold =  np.concatenate(((np.full((N_train),-1),np.full((N_val),0))))
    ps = PredefinedSplit(test_fold.tolist())

    # Regress & Predict
    Ypred = np.full([1, Y_train.shape[1]],np.nan)
    # Loop over maturities
    for i in range(Y_train.shape[1]):

        model = ElasticNetCV(cv=ps, max_iter=5000, n_jobs=-1,
                             l1_ratio=[.1, .3, .5, .7, .9],
                             random_state=42)
        model = model.fit(np.concatenate((Xexog_train, X_train),axis=1),
                          Y_train[:,i])
        Ypred[0,i]=model.predict(np.concatenate((Xexog_test, X_test),axis=1))

    return Ypred



# =========================================================================
#                   Gradient Boosted Regression Tree   (fwd-rate) 
# =========================================================================

# in random GBRT, no need to scale x-variable 

def GBRT(Xexog,Y):
    # Xeog : fwd-data only (without macro-var)
    # y : excess-return 
    
    # Loss ftn : least-square, learning-rate : .1, # of boosting stage : 100
    # initial estimator = 0, max_features : auto(all features)
    # n_iter_no_change = None (no early stopping)

    import numpy as np
    from sklearn.ensemble import GradientBoostingRegressor
    
    # Split data into training and test
    Xexog_train = Xexog[:-1,:] 
    Y_train = Y[:-1,:]
   
    Xexog_test = Xexog[-1,:]
    Xexog_test = Xexog_test.reshape(1, -1)
           
    Ypred = np.full([1, Y_train.shape[1]],np.nan)
    
    # Loop over maturities
    for i in range(Y_train.shape[1]):
        GBR = GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators = 100, 
                                        max_depth = 5, init='zero', random_state=42, 
                                        max_features = 'auto', n_iter_no_change = None)
        GBR.fit(Xexog_train, Y_train[:,i])
        Ypred[0,i]=GBR.predict(Xexog_test)
        
    return Ypred



# =========================================================================
#                   Random-forest  (fwd-rate + macro) 
# =========================================================================

# in random forest, no need to scale x-variable 

def m_RandomForest(Xexog,X,Y):
    # Xeog : fwd-data only (without macro-var)
    # X : macro varaible(128-variable)
    # y : excess-return 
    
    # 1. the number of trees(n_estimators)  2. the depth of the individual trees(max_depth),  
    # 3. the size of the randomly selected sub-set of predictors (max_features) 
    # 4. max samples 
    
    # n_estimators = 100, max_depth = 4, bootstrap = True, max_features = 'sqrt', max_samples = 0.4
    
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    
    # Split data into training and test
    X_train = X[:-1,:]
    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]
    X_test = X[-1,:]
    X_test = X_test.reshape(1, -1)
    Xexog_test = Xexog[-1,:]
    Xexog_test = Xexog_test.reshape(1, -1)
       
    Ypred = np.full([1, Y_train.shape[1]],np.nan)
    
    # Loop over maturities
    for i in range(Y_train.shape[1]):
        RFR = RandomForestRegressor(n_estimators = 100, max_depth = 5, bootstrap = True, max_features = 'sqrt',
                                    n_jobs=-1, random_state=42, max_samples = 0.4)
        RFR.fit(np.concatenate((Xexog_train, X_train),axis=1), Y_train[:,i])
        Ypred[0,i]=RFR.predict(np.concatenate((Xexog_test, X_test),axis=1))
        
    return Ypred




# =========================================================================
#                   Random-forest  (fwd-rate + macro) (hyperparameter tuning)
# =========================================================================

# in random forest, no need to scale x-variable 

def m_h_RandomForest(Xexog,X,Y):
    # Xeog : fwd-data only (without macro-var)
    # X : macro varaible(128-variable)
    # y : excess-return 
    
    # We can set many hyper-parameter in Random forest model. 
    # 1. the number of trees(n_estimators)  2. the depth of the individual trees(max_depth),  
    # 3. the size of the randomly selected sub-set of predictors (max_features) 
    # 4. min_samples_split   5. min_samples_leaf  6. max_samples  ..........etc....
    # 
    # Here we only consider 1(n_estimators), 2(max_depth), 3(max_features) for hyper-parameter tuning
    
    
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import GridSearchCV
    from sklearn.model_selection import PredefinedSplit
    
    # Split data into training and test
    X_train = X[:-1,:]
    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]
    X_test = X[-1,:]
    X_test = X_test.reshape(1, -1)
    Xexog_test = Xexog[-1,:]
    Xexog_test = Xexog_test.reshape(1, -1)
    
    # Construct Validation sample as last 15% of sample
    N_train = int(np.round(np.size(X_train,axis=0)*0.85))
    N_val = np.size(X_train,axis=0)-N_train
    test_fold =  np.concatenate(((np.full((N_train),-1),np.full((N_val),0))))
    ps = PredefinedSplit(test_fold.tolist())
    
    # Set hyper-parameter candidate 
    n_estimators = [100, 250, 500, 1000]
    max_depth = [3,4,5]
    max_features = ["auto", "sqrt", "log2"]
    
    grid_param = {'n_estimators':n_estimators, 'max_depth':max_depth, 'max_features':max_features}
    
    Ypred = np.full([1, Y_train.shape[1]],np.nan)
    
    # Loop over maturities
    for i in range(Y_train.shape[1]):      

        RFR = RandomForestRegressor(bootstrap = True, n_jobs=-1, random_state=42, max_samples = 0.5)
        
        RFR_grid = GridSearchCV(estimator=RFR, param_grid=grid_param, n_jobs=-1, cv=ps)
        RFR_grid.fit(np.concatenate((Xexog_train, X_train),axis=1), Y_train[:,i])
        
        Ypred[0,i]=RFR_grid.predict(np.concatenate((Xexog_test, X_test),axis=1))
        
    return Ypred

        
    
    
# =========================================================================
#                 Early stopping (to prevent overfitting in NN)
# =========================================================================


import numpy as np
import torch

class EarlyStopping:
    # early stops the training if validation loss doesn't improve after a given patience   
    def __init__(self, patience=20, verbose=False, delta=1e-6, path='checkpoint.pt', trace_func=print):
        #    patience (int): How long to wait after last time validation loss improved.
        #                    Default: 20
        #    verbose (bool): If True, prints a message for each validation loss improvement. 
        #                    Default: False
        #    delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        #                    Default: 1e-6
        #    path (str): Path for the checkpoint to be saved to.
        #                    Default: 'checkpoint.pt'
        #    trace_func (function): trace print function.
        #                    Default: print            
        
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




# =========================================================================
#                  Neural-Net, fwd only
# =========================================================================

def NN_fwd(Xexog, Y, archi, epoch, dropout_prob):
    
    # X,Y are np.arraay value
    #archi is # of neurons in layer. (ex [3,4,5])
    
    import torch 
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np  
    from torch.utils.data import TensorDataset
    from torch.utils.data import DataLoader
    
    
    #seed 
    torch.manual_seed(11)
    np.random.seed(11)
 
    #Split Data for Test and Training  
    Xexog_train = Xexog[:-1,:]
    Y_train = Y[:-1,:]
    Xexog_test = Xexog[-1,:].reshape(1,-1)

    #Scale the predictors for training
    Xexog_scaler_train =  MinMaxScaler(feature_range=(-1,1))
    Xexog_scaled_train = Xexog_scaler_train.fit_transform(Xexog_train)
    Xexog_scaled_test = Xexog_scaler_train.transform(Xexog_test)
    
    
    # split train/ cross validation
    N_train = int(np.round(np.size(Xexog_train,axis=0)*0.85))
    N_val = np.size(Xexog_train,axis=0)-N_train

    Xexog_scaled_val = Xexog_scaled_train[N_train:,:]
    Xexog_scaled_train = Xexog_scaled_train[:N_train,:]
    
    Y_val = Y_train[N_train:,:]
    Y_train = Y_train[:N_train,:]
    
    
    # from np.array > torch tensor     
    Xexog_scaled_val = torch.tensor(Xexog_scaled_val)
    Xexog_scaled_train = torch.tensor(Xexog_scaled_train)
    Xexog_scaled_test = torch.tensor(Xexog_scaled_test).reshape(1,-1)
   
    Y_val = torch.from_numpy(Y_val)
    Y_train = torch.from_numpy(Y_train)
    
    # dataset
    train_dataset = TensorDataset(Xexog_scaled_train, Y_train)   
    valid_dataset = TensorDataset(Xexog_scaled_val, Y_val)
    
    trainloader = DataLoader(train_dataset, batch_size=16, shuffle=True, drop_last=False)
    validloader = DataLoader(valid_dataset, batch_size=16, shuffle=True, drop_last=False)
    
    
    # define Network 
    class NN_fwd_model(nn.Module):
        
        def __init__(self, Xexog_scaled_train, Y_train, archi, dropout_prob):
            
            super(NN_fwd_model, self).__init__()
            
            self.Xexog_scaled_train = Xexog_scaled_train
            self.Y_train = Y_train
            self.archi = archi
            self.dropout_prob = dropout_prob
            
            self.fwd_module = torch.nn.Sequential()
            
            n = len(self.archi)
            for i in range(n+1):
                if i==0:
                    
                    self.fwd_module.add_module('linear'+str(i+1), nn.Linear(Xexog_scaled_train.shape[1], archi[i]))
                    self.fwd_module.add_module('Relu'+str(i+1), nn.ReLU())
                    #self.fwd_module.add_module('dropout'+str(i+1), nn.Dropout(dropout_prob, inplace=True))
                    
                elif i>0 and i<n:
                    
                    self.fwd_module.add_module('linear'+str(i+1), nn.Linear(archi[i-1], archi[i]))
                    self.fwd_module.add_module('Relu'+str(i+1), nn.ReLU())
                    #self.fwd_module.add_module('dropout'+str(i+1), nn.Dropout(dropout_prob,inplace=True))
                    
                else:
                    self.fwd_module.add_module('BN', nn.BatchNorm1d(archi[i-1]))
                    self.fwd_module.add_module('linear'+str(i+1), nn.Linear(archi[i-1], Y_train.shape[1]))
                    
                    
            
            # Using He-initilization 
            for m in self.fwd_module:
                if isinstance(m,nn.Linear):
                    nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                    
         
        def forward(self, Xexog_scaled_train):
           y_hat = self.fwd_module(Xexog_scaled_train)
           
           return y_hat
    
    
    model = NN_fwd_model(Xexog_scaled_train.float(), Y_train.float(), archi, dropout_prob)
    print(model)
    #print(list(model.parameters()))
    #print(model)   
    # define loss ftn & optimizer
    loss_ftn = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.9,weight_decay=0.01, nesterov=True)
    early_stopping = EarlyStopping(patience=40, delta=1e-6)
    
    
    losses=[]
    for i in range(epoch):
        
        # to track the training loss as the model trains
        train_losses = []
        # to track the validation loss as the model trains
        valid_losses = []
        # to track the average training loss per epoch as the model trains
        avg_train_losses = []
        # to track the average validation loss per epoch as the model trains
        avg_valid_losses = []
  
        model.train()
        for (batch_X, batch_Y) in trainloader:
            
            optimizer.zero_grad()
           
            # compute the model output
            trained_y = model(batch_X.float())            
            
            # calculate loss
            loss = loss_ftn(trained_y, batch_Y.float())        
            
            # credit assignment
            loss.backward()
            
            # update model weights
            optimizer.step()
            
            train_losses.append(loss.item())
        
        model.eval()
        for (batch_X, batch_Y) in validloader:
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(batch_X.float())
            # calculate the loss
            loss = loss_ftn(output, batch_Y.float())
            # record validation loss
            valid_losses.append(loss.item())
            
            
        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)        
        
        train_losses = []
        valid_losses = []
        
        
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    model.eval()
    Ypred = model(Xexog_scaled_test.float())
    Ypred = Ypred.detach().numpy()
    print(Ypred.shape)
        
    return Ypred

















# =========================================================================
#                  Neural-Net, Figure3-(a) in BBT(2020)
# =========================================================================

def m_NN_fwd_last(Xexog, X, Y, archi, epoch, dropout_prob):
    
    # X,Xexog,Y are np.arraay value
    #archi is # of neurons in layer. (ex [3,4,5])
    
    import torch 
    import torch.nn as nn
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np    
    
    
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
    X_scaled_test = Xscaler_train.transform(X_test)

    Xexog_scaler_train =  MinMaxScaler(feature_range=(-1,1))
    Xexog_scaled_train = Xexog_scaler_train.fit_transform(Xexog_train)
    Xexog_scaled_test = Xexog_scaler_train.transform(Xexog_test)
    
    
    # split train/ cross validation
    N_train = int(np.round(np.size(X_train,axis=0)*0.85))
    N_val = np.size(X_train,axis=0)-N_train

    X_scaled_val = X_scaled_train[N_train:,:]
    X_scaled_train = X_scaled_train[:N_train,:]
    Xexog_scaled_val = Xexog_scaled_train[N_train:,:]
    Xexog_scaled_train = Xexog_scaled_train[:N_train,:]
    
    Y_val = Y_train[N_train:,:]
    Y_train = Y_train[:N_train,:]
    
    
    # from np.array > torch tensor 
    X_scaled_val = torch.from_numpy(X_scaled_val)
    X_scaled_train = torch.from_numpy(X_scaled_train)
    X_scaled_test = torch.from_numpy(X_scaled_test).reshape(1,-1)
    
    Xexog_scaled_val = torch.from_numpy(Xexog_scaled_val)
    Xexog_scaled_train = torch.from_numpy(Xexog_scaled_train)
    Xexog_scaled_test = torch.from_numpy(Xexog_scaled_test).reshape(1,-1)
    
    Y_val = torch.from_numpy(Y_val)
    Y_train = torch.from_numpy(Y_train)
    
    
    # define Network 
    class NNmodel(nn.Module):
        
        def __init__(self, Xexog_scaled_train, X_scaled_train, Y_train, archi, dropout_prob):
            
            super(NNmodel, self).__init__()
            n = len(archi)
            
            self.macro_module = torch.nn.Sequential()
            for i in range(n):
                if i==0:
                    self.macro_module.add_module('dropout'+str(i+1), nn.Dropout(dropout_prob))
                    self.macro_module.add_module('linear'+str(i+1), nn.Linear(X_scaled_train.shape[1], archi[i]))
                    self.macro_module.add_module('Relu'+str(i+1), nn.ReLU())
                else:
                    self.macro_module.add_module('dropout'+str(i+1), nn.Dropout(dropout_prob))
                    self.macro_module.add_module('linear'+str(i+1), nn.Linear(archi[i-1], archi[i]))
                    self.macro_module.add_module('Relu'+str(i+1), nn.ReLU())
            
            # Using He-initilization 
            for m in self.macro_module:
                if isinstance(m,nn.Linear):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity="relu")
                    
                    
            # for macro + fwrd model 
            self.merge_module = torch.nn.Sequential()
            self.merge_module.add_module('dropout_final', nn.Dropout(dropout_prob))  
            self.merge_module.add_module('BN', nn.BatchNorm1d(archi[-1]+Xexog_scaled_train.shape[1]))
            self.merge_module.add_module('linear_final', nn.Linear(archi[-1]+Xexog_scaled_train.shape[1], Y_train.shape[1] ))

            # Using He-initilization 
            for m in self.merge_module:
                if isinstance(m,nn.Linear): 
                    nn.init.kaiming_normal_(m.weight)
         
        def forward(self, Xexog_scaled_train, X_scaled_train):
           macro_final = self.macro_module(X_scaled_train)
           merge_final = self.merge_module(torch.cat((macro_final, Xexog_scaled_train), 1))
           
           return merge_final
    
    
    model = NNmodel(Xexog_scaled_train.float(), X_scaled_train.float(), Y_train.float(), archi, dropout_prob)
        
    # define loss ftn & optimizer
    loss_ftn = torch.nn.MSELoss(reduction='mean')
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.9, weight_decay=0.01, nesterov=True)
    early_stopping = EarlyStopping(patience=20, delta=1e-6)
    
    for i in range(epoch):
         
        for j in range(N_train):
                        
            optimizer.zero_grad()
            
            model.train()
            # compute the model output
            trained_y = model(Xexog_scaled_train.float(), X_scaled_train.float())
            
            # calculate loss
            loss = loss_ftn(trained_y, Y_train.float())
            print(loss)
            # credit assignment
            loss.backward()
            
            # update model weights
            optimizer.step()
            
        # for validation 
        model.eval()
        validation_y = model(Xexog_scaled_val.float(), X_scaled_val.float())
        val_loss = loss_ftn(validation_y, Y_val.float())
        #print(val_loss)
        
        early_stopping(val_loss, model)
        
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    model.eval()
    Ypred = model(Xexog_scaled_test.float(), X_scaled_test.float())
    Ypred = Ypred.detach().numpy()
        
    return Ypred
    
    
    
    
    
    

    