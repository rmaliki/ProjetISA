

from scipy.stats.stats import pearsonr   as corr
import numpy as np
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

import sklearn.feature_selection as sel
from matplotlib import cm as c
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from numpy.linalg import norm
from sklearn.neighbors import NearestNeighbors as NN


#an object of this class holds information about a local linear model. at the end, each point on the surfx, surf y plane has
# his own, ( or even several models)

class modData:
    QTILE =0.05 # not important
    qtile=QTILE# quantiles for chosing the intervals on each side
    params=# the vector of the linear function
    intercept=0
    resid=0 # residus of the model
    rmse=0
    qtile_range=0
    n_params=0
    interval=-1
    estimate=-10   # the estimate ( oil or gas value of the specific point)
    error=20
    x=0  # surf x of the point
    y=0  # surf y



### this function calulates a  linear model, given 
###  an array "a" ( sample, which is from a knn neighborhood of a specific point) with all the features including target
### TRE  treshold to cut off features after doing pca ( not important)
### gas = boolean whether to predict oil or gas
def get_model(a,TRE,gas=True):
#########################################   can be skipped  until **
    if len(a[:,0]) <= 10:
        AL=0.1
    else:
        AL=0.1+(len(a[:,0])-10)*0.25
  #################################################  **    
    zero_params=False  # an indicator whether the linear model has zero paramters ( only constant)

##  extract the target from the array a
   
    if(gas):
        yg=np.array(a[1:,42])
    else:
        yg=np.array(a[1:,43])
     
## cut off target   
    a=a[:,:42]
 # normalize the data  
    scaler = StandardScaler()
    x=np.array(scaler.fit_transform(a))
# by default, the point for whom the local linear model shall be constructed is the first row of the array.
# in order to make a crossfold, we are going to cut it off for making the model
    argument=np.array(x[0,:])
    x=x[1:,]
   
    ## PCA:   to be more efficient 

    pca = PCA()
    pca.fit(x)
    #cum=sum(pca.explained_variance_ratio_ *100)# keeps 96% of variance
    xx=pca.transform(x)
    selector = sel.VarianceThreshold(threshold=TRE) # delete useless principal components
    xx=selector.fit_transform(xx)

# do an f-test, which features are candidates for the linear regression ( to not overfit by using to many features)
    p_vals=sel.f_regression(xx,yg)[1]

    for i in range(len(p_vals)):## not important
        if np.isnan(p_vals[i]):
            p_vals[i]=2# secure handilng for n small
   
    sign_features_ind=[]
    sign_features_ind=[i for i in range(len(p_vals)) if p_vals[i] < AL] # indices of the significant features
    #print(sign_features_ind)
    xx2=xx[:,sign_features_ind]### use only significant features for linear model
    support_pca=pca.components_[sign_features_ind,:]## take the prinicipal components' original coordinates

    #inear model

    X=xx2
    X=sm.add_constant(X)
    model=sm.OLS(yg,X)
    result=model.fit()
    resids=result.resid
    params=result.params
    interc=params[0]
    
    #retranform the regression parameters from pca space to original
    if(len(params)>1):
        params_origninal_space=[support_pca[i]* params[i+1] for i in range(len(params)-1)]
        params_origninal_space=np.array(params_origninal_space)
        params_origninal_space=[np.sum(params_origninal_space[:,i]) for i in range(len(x[0]))]
        params_origninal_space=np.array(params_origninal_space)
        
        
    else:
        zero_params=True
        
    
   
    modD=modData() ## create and object to hold the model and fil in the data of the model ( can be skipped)
    if(not zero_params):
        modD.params=params_origninal_space
    modD.intercept=interc
    resids=np.sort(resids)
 
    modD.resid=resids
    modD.rmse=np.sqrt(np.sum(resids**2))
    modD.n_params=len(params)-1
    ####calculate the quantile range
    delta=int(len(x[:,0])*modD.qtile+0.25)# rounded up from 0.75
    
    upper=len(x[:,0])-delta
    modD.qtile_range=[resids[delta],resids[upper-1]]
    modD.interval=resids[upper-1]-resids[delta]
    if(not zero_params):
        modD.estimate=np.dot(params_origninal_space,argument)+modD.intercept
    else:
        modD.estimate=np.mean(yg)
        
    return modD
#############################################################
#END DEF



## can be skipped    
df=pd.read_csv('ready_data.csv',sep=';')
df=df.loc[:,'API':]
rows=df[~df['Nbr_Stages'].isnull()].index
df2=df.loc[rows,:]
ind=list(df2.columns)
ind.remove('Zone')
ind.remove('GasCum360')
ind.remove('OilCum360')
ind.append('GasCum360')
ind.append('OilCum360')
df2=df2.loc[:,ind]









d=df2.copy()
ind2=list(d.columns)
#ind2.remove('GasCum360')
#ind2.remove('OilCum360')
ind2.remove('API')
x=d.loc[:,ind2]
x=np.array(x) # hold the data in an array

#### KNN


X=x[:,0:2]# take only surf_x and surf_y for knn
nbrs = NN( algorithm='ball_tree')

nbrs.fit(X)# construct a tree for efficient knn calculation later
max_n=120
smallest_ens_size=3

## different sizes for nearest neighbor environment
sizes=[smallest_ens_size+1+ i*4 + int(i*i) for i in range(7)]# because point itself will not be use for estimation. CROSSFOLD

scores=np.zeros((len(d),len(sizes),max_n+20)) #array to hold the models and additional info  for each point
scores=scores+100# 
gas=False
####### #### now build for each point several local linear models ( here oil) and save all the data about them in the "scores array"
for ff in range(len(d)):
    
    
   
    
    nn=nbrs.kneighbors([X[ff]], max_n)# get the 120 nearest neighbors
    dists=nn[0]#[:,1:]# do not take point itself
    indices=nn[1]#[:,1:]# do not take point itself
  
    
    
    for i in range(len(sizes)):# for all the nearest neighbor environments of different sizes
        inds4=indices[0,0:sizes[i]]# indices of the knns
        ens=x[inds4,:]
        m=get_model(ens,0.5,gas=gas)##  make the linear model for the knns
        m.error=m.estimate-x[ff,43]## calulate the error of the model
        m.x=x[ff,0] ## fill the rest of the model data for 
        m.y=x[ff,1]
        scores[ff,i,0]=m.error
        scores[ff,i,1]=np.abs(m.error)
        scores[ff,i,2]=sizes[i]-1
        # std of environment        
      
        
        if gas:
            scores[ff,i,3]=np.std(x[inds4,42])
            
            
        else:
            scores[ff,i,3]=np.std(x[inds4,43])
           
            
        scores[ff,i,5]=m.x
        scores[ff,i,6]=m.y
        scores[ff,i,7:10]=inds4[1:4]the nearest neighbors if needed later
        scores[ff,i,11:(sizes[i]+11-1)]=m.resid# the residuus if needed later
       
        

      
## save the array for further analysis    
      

eval=open('error_predn3','wb')
np.save(eval,scores)
eval.close()

