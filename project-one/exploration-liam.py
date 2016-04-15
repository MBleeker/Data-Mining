# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 15:39:27 2016

@author: liam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

# Read in data to big pandas df and setup time variable
os.chdir('/home/liam/cloud/uni/dm/assign1/')
big_df = pd.read_csv('dataset_mood_smartphone.csv')
big_df['time'] = pd.to_datetime(big_df['time'])
feature_names = ['mood']+[v for v in big_df['variable'].unique() if v !='mood']

# Function to extract a particular feature for a particular individual
def get_feature(feature_name, indiv_id, big_df = big_df):
    feature = big_df[big_df['variable']==feature_name]
    feature = feature[feature['id']==indiv_id][['time','value']]
    return feature.set_index(['time'])

from scipy.stats import skew, kurtosis
mood_all = big_df[big_df['variable']=='mood'][['value']]
print mood_all.mean()
print skew(mood_all)
print kurtosis(mood_all)
mood_all.hist()


from statsmodels.tsa.stattools import acf, pacf
def get_feature_by_day(feature, current_indiv):
    y = get_feature(feature, current_indiv)
    avg_features = ['mood', 'circumplex.valence', 'circumplex.arousal']
    sum_features = [s for s in feature_names if s not in avg_features]
    if feature in avg_features:
        mn = 'mean'
    else:
        mn = 'sum'
    y = y.resample('d', how=mn)
    start, end = find_start_and_end_dates(y['value'], 10)
    y = y.reindex(pd.date_range(start = y.index[start], end = y.index[end]))
    y.index = y.index.rename('time')
    if feature in avg_features:
        y = y.interpolate()
    else:
        y = y.fillna(0)
    return y

def plot_acf_and_pacf(y):
    lag_acf = acf(y, nlags=20)
    lag_pacf = pacf(y, nlags=20, method='ols')
    
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(y)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(y)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(y)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(y)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()
    plt.close()
indiv_ids = big_df['id'].unique()
#%%
def plot_time_series(y):
    y.plot(ylim=(0,10))
    #plt.title('Mood Time Series for Subset of Individuals)
    plt.legend().set_visible(False)
    plt.show()
    plt.savefig('mood_time_series.png')
    #plt.close()
    
#for current_indiv in indiv_ids:
#    print current_indiv
#    y = get_feature_by_day('mood', current_indiv)
    #plot_acf_and_pacf(y)
    #plot_time_series(y)

num = len(indiv_ids)
y = get_feature_by_day('mood', indiv_ids[0])
y = y.reindex(pd.date_range(start = '2014-4-01', end = '2014-4-15', freq='D'))

cols = [indiv_ids[0]]
for current_indiv in indiv_ids[1:num]:
    y = pd.merge(y, get_feature_by_day('mood', current_indiv), 
                 left_index = True, 
               right_index = True, how = 'left')
    cols.append(current_indiv)
y.columns = cols
#y = y.reindex
plot_time_series(y)
