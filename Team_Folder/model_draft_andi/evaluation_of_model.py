#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Nov 26 18:17:44 2017

@author: andreas
"""

#first dimesnion of the array: id of point
# second : id that identifies the environment size used to build the model ( 3 -9 -14 -20 -34 40- 54)

# data strucure of the third dimesion of the array:

#0 error
#1 abs error
#2 size
#3 oil var
#4 oil var 2
#5 x
#6 y
#7 - 12 NNs
# rest : residuus of the model

import matplotlib.pyplot as plt
from matplotlib import cm as c
import numpy as np

cm=plt.get_cmap('nipy_spectral') # for the plot
import matplotlib.pyplot as plt
f=open('error_pred','rb')
s=np.load(f)


size_ind=3 #not imp

n3=int(s[0,3,2])# environment size 
resid3=s[:,3,13:n3+13]  # residuuus for all models with environment size of n3 neighbors

n0=int(s[0,0,2])
resid0=s[:,0,13:n0+13]

n5=int(s[0,5,2])
resid5=s[:,5,13:5+13]

nn=s[:,2,7:12]
abser=s[:,:,1]# extract absolute error for models of all points and all environment sizes
oil_var=s[:,:,3]  ## this is the std deviation of oil in the environment the model was built

mean_abser=[np.mean(abser[:,i]) for i in range(len(oil_var[0,:]))]
x=s[:,size_ind,5]
y=s[:,size_ind,6]


e_pred=np.zeros((len(x)))  ## this part is not important
for i in range(len(x)):
    nns=[int(e) for e in nn[i] ]
    e_nns=[abser[j] for j in nns[:-2]]
    e_pred[i]=np.mean(e_nns)

#

plt.scatter(oil_var[:,0],abser[:,0],s=8)  # plot the mean abs error for the models with size = 5 neighbors
# against the std dev of oil in the environment ( get a good indicator how good the model is)

