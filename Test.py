# -*- coding: utf-8 -*-
"""
Created on Sun Oct 22 10:41:32 2017

@author: andreas
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import outlier_removal as out

df=pd.read_csv('TrainSample.csv',sep=';' )
out.replOutl(df,'ShutInPressure_Fil (KPa)',1.5)