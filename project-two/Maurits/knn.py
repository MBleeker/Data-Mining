# -*- coding: utf-8 -*-
"""
Created on Wed May  4 11:37:58 2016

@author: Maurits
"""
#%%
import numpy as np
import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
#%%
train_data_in = pd.read_csv('/Users/Maurits/Documents/GitHub/School Projects/Data-Mining/project-two/train_set_10pct_of_90pct.csv')
#%%

def fill_in_missing_data(train_data_in, col_name,k=1, default = -99999):
  comp_cols = ['comp1_rate','comp1_inv','comp1_rate_percent_diff',
              'comp2_rate','comp2_inv','comp2_rate_percent_diff',
              'comp3_rate','comp3_inv','comp3_rate_percent_diff',
              'comp4_rate','comp4_inv','comp4_rate_percent_diff',
              'comp5_rate','comp5_inv','comp5_rate_percent_diff',
              'comp6_rate','comp6_inv','comp6_rate_percent_diff',
              'comp7_rate','comp7_inv','comp7_rate_percent_diff',
              'comp8_rate','comp8_inv','comp8_rate_percent_diff']
              
  null_cols = ['visitor_hist_starrating','visitor_hist_adr_usd',
                     'prop_location_score2','srch_query_affinity_score',
                     'orig_destination_distance'] + comp_cols
                     
  zero_cols = ['prop_starrating', 'prop_review_score', 
                     'prop_log_historical_price']
  drop_cols = ['date_time']
  original_data = train_data_in
  train_data_in = train_data_in.drop(drop_cols , axis=1)
  
  if col_name in null_cols:
    null_cols.remove(col_name)
    train_data_in.loc[:,zero_cols] = train_data_in.loc[:,zero_cols].replace(0, default)
    train_data_in.loc[:,null_cols] = train_data_in.loc[:,null_cols].fillna(default)

    train_data = train_data_in.loc[~train_data_in.loc[:,col_name].isnull(),:]
    train_labeles = train_data_in.loc[~train_data_in.loc[:,col_name].isnull(), col_name]
    predicting_rows = train_data_in.loc[train_data_in.loc[:,col_name].isnull(), col_name]
    index = train_data_in.loc[train_data_in.loc[:,col_name].isnull(), col_name].index
  if col_name in zero_cols:
    zero_cols.remove(col_name)
    
    train_data_in.loc[:,zero_cols] = train_data_in.loc[:,zero_cols].replace(0, default)
    train_data_in.loc[:,null_cols] = train_data_in.loc[:,null_cols].fillna(default)

    train_data = train_data_in.loc[train_data_in[col_name] != 0]
    train_labeles = train_data_in.loc[train_data_in[col_name] != 0, col_name]
    predicting_rows = train_data_in.loc[train_data_in[col_name] == 0]
    index = train_data_in.loc[train_data_in[col_name] == 0].index


  train_data_in = train_data_in.drop(col_name, axis=1)
  neigh = KNeighborsClassifier(n_neighbors=k)
  neigh.fit(train_data.values, train_labeles.values) 
           
  y = neigh.predict(predicting_rows)
           
  data = original_data.loc[index, col_name] = y[col_name]
  return data

#%%

'visitor_hist_starrating'