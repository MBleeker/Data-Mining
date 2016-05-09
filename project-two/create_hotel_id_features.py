# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:51:37 2016

@author: liam
"""
import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from query import *
from lambda_rank import *
train_data_in = pd.read_csv('train_set_90pct.csv')
#%%
def cat_to_prob(data, col_name):
    grouped = data.groupby(col_name)
    ctr = grouped.apply(lambda x: sum(x['click_bool'])*1./len(x))
    data = data.merge(pd.DataFrame(ctr, columns = [col_name + '_ctr']), 
                       right_index = True, left_on = col_name)
    btr = grouped.apply(lambda x: sum(x['booking_bool'])*1./len(x))
    data = data.merge(pd.DataFrame(btr, columns = [col_name + '_btr']), 
                       right_index = True, left_on = col_name)
    return data
train_data_in = cat_to_prob(train_data_in, 'prop_id')
#%%
def stats_of_col_by_group(data, grouping_col_name, val_col_name):
    grouped = data.groupby(grouping_col_name)
    val = grouped.apply(lambda x: x[val_col_name].mean())
    data = data.merge(pd.DataFrame(val, 
                            columns = [val_col_name + '_mean_by_' + \
                                        grouping_col_name]), 
                            right_index = True, left_on = grouping_col_name)
    val = grouped.apply(lambda x: x[val_col_name].std())
    data = data.merge(pd.DataFrame(val, 
                            columns = [val_col_name + '_std_by_' + \
                                        grouping_col_name]), 
                            right_index = True, left_on = grouping_col_name)
    val = grouped.apply(lambda x: x[val_col_name].median())
    data = data.merge(pd.DataFrame(val, 
                            columns = [val_col_name + '_median_by_' + \
                                        grouping_col_name]), 
                            right_index = True, left_on = grouping_col_name)
    return data
train_data_in = stats_of_col_by_group(train_data_in, 
                                      'prop_id', 'prop_location_score2')
#%%
dd = pd.DataFrame(sorted(train_data_in['prop_id'].unique()), 
                  columns = ['prop_id'])
keep_cols = [c for c in train_data_in.columns if 'prop_id' in c]
#%%
dout = pd.merge(dd, train_data_in[keep_cols]).drop_duplicates('prop_id')
dout.to_csv('features_by_prop_id.csv', index=False)