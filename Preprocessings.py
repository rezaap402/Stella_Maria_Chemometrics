import numpy as np
import pandas as pd
import statsmodels.api as sm

from scipy.signal import savgol_filter

def snv(input_data):
    # Define a new array and populate it with the corrected data  
    data_snv = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply standard normal variate correction
        data_snv[i,:] = (input_data[i,:] - np.mean(input_data[i,:])) / np.std(input_data[i,:]) 
    return data_snv

def msc(x):
    XS, YS = x.shape
    SpectralMean = np.mean(x, axis=0)
    YY = np.zeros((XS, YS))
    
    for ii in range(XS):
        SpectralMean1 = np.column_stack((np.ones(YS), SpectralMean))
        model = sm.OLS(x[ii, :], SpectralMean1).fit()
        b = model.params
        YY[ii, :] = (x[ii, :] - (b[0] * np.ones(YS))) / (b[1] * np.ones(YS))

def sg1(x):
    sg1 = savgol_filter(x, window_length=5, polyorder=3, deriv=1)
    return sg1
