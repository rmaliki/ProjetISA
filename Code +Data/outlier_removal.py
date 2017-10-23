# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:38:28 2017

@author: andreas
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
##### This function is replacing outliers in a column of the data set by 
# the mean.
#df data frame
#colum= name of the column 
#factor iqr= factor to determine outliers ( x inter quartile range)
# visu= boolean, enable visualization ot not
def replOutl(df,column,factor_iqr,visu=True):
    if(visu==True): #plottting 
        df.boxplot(column,return_type='dict',whis=factor_iqr)
        plt.title('outliers')
        f, axarr = plt.subplots(ncols=2,sharey=True)
        y=df[column]
        x=range(1, len(y)+1)
        axarr[0].plot(x,y,'.')
        axarr[0].set_title('original')
        
    #quartiles
    Q1=np.percentile(df.dropna()[column],25)
    Q3=np.percentile(df.dropna()[column],75)
    iqr=Q3-Q1
    
    #Replace outliers by mean
    mean=np.mean(df[column])
    df[df[column]< Q1 - factor_iqr*iqr ]=mean
    df[df[column]> Q3 + factor_iqr*iqr ]=mean
    if(visu==True):
        y2=df[column]
        x2=range(1, len(y)+1)
        axarr[1].plot(x2,y2,'.')
        axarr[1].set_title('replaced outliers')
        plt.show()