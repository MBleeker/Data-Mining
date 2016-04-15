# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 10:54:26 2016

@author: Maurits
"""
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

# Read in data to big pandas df and setup time variabl
os.chdir('/Users/Maurits/Documents/GitHub/School Projects/Data-Mining/project-one/data')
big_df = pd.read_csv('dataset_mood_smartphone.csv')
big_df['time'] = pd.to_datetime(big_df['time'])

pd.big_df['circumplex.arousal']
pd.big_df['circumplex.valence ']
#%%
 

# Function to extract a particular feature for a particular individual
def get_feature(feature_name, big_df = big_df):
    feature = big_df[big_df['variable']==feature_name]
    feature = feature[['time','value']]
    return feature.set_index(['time'])
    
    
    
valence = get_feature('circumplex.valence')
arousal = get_feature('circumplex.arousal')
#%%
mat = np.zeros((5,5))
valence = valence.fillna(0)
arousal = arousal.fillna(0)
for i in range(-2, 3):
    for j in range(-2,3):
        mat[i+2,j+2] = np.sum((arousal['value'].values==i)&(valence['value'].values==j))
  
m = pd.DataFrame(mat)
m.columns = [str(s) for s in range(-2, 3)]
m.index = [str(s) for s in range(-2, 3)]

fig, ax = plt.subplots()
from matplotlib import cm
cax = ax.imshow(mat, interpolation='nearest', cmap=cm.coolwarm)

cbar = fig.colorbar(cax, ticks=[0, 250, 1200])
cbar.ax.set_yticklabels([''])  # vertically oriented colorbar
fig.savefig('test.png')
#%%
cax = ax.matshow(corr)
fig.colorbar(cax, ticks=[-1, 0, 1])
#%%
fig = plt.matshow(mat)
fig.colorbar(fig, ticks=[-1, 0, 1])
#%%
ax.set_xlim([2,-2])
ax.set_ylim([-2,2])
plt.show
#%%
