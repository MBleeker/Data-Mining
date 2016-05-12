# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:51:37 2016

@author: liam
"""
import pandas as pd
import numpy as np
train_data_in = pd.read_csv('train_set_90pct.csv')
#%%
def get_grouping_col_name_string(grouping_col_name):
    if type(grouping_col_name) == list:
        str_group = ''
        for st in range(len(grouping_col_name)):
            if st == len(grouping_col_name) - 1:
                str_group += grouping_col_name[st]
            else:
                str_group += grouping_col_name[st] + '_and_'
    else:
        str_group = grouping_col_name
    return str_group
# Use to make features like: 'stdev of prop_location_score2 by prop_id'
def stat_of_col_by_group(data, grouping_col_name, val_col_name, 
                          fn, fn_name, min_length=10):
    str_group = get_grouping_col_name_string(grouping_col_name)
    fn_name = '_' + fn_name + '_by_'
    grouped = data.groupby(grouping_col_name)
    val = grouped.apply(lambda x: fn(x[val_col_name]))
    data = data.merge(pd.DataFrame(val, 
                       columns = [val_col_name + fn_name + str_group]), 
                       right_index = True, left_on = grouping_col_name)
    return data
# Makes column applying every function in fns (if in fn_dict) to the values of
# val_col_name to each group specified by grouping_col_name
def stats_of_col_by_group(data, grouping_col_name, val_col_name, fns, 
                          min_length=10):
    def mean_if_long(x, min_length=10):
        if len(x) > min_length:
            return np.mean(x)
        else:
            return np.nan
    def std_if_long(x, min_length=10):
        if len(x) > min_length:
            return np.std(x)
        else:
            return np.nan
    def median_if_long(x, min_length=10):
        if len(x) > min_length:
            return np.median(x)
        else:
            return np.nan
    fn_dict = {'mean' : mean_if_long, 
               'std' : std_if_long, 
               'median' : median_if_long}
    if type(fns) == str:
        fns = [fns]
    for fn_name in fns:
        if fn_name in fn_dict:
            data = stat_of_col_by_group(data, grouping_col_name, val_col_name,
                                        fn_dict[fn_name], fn_name, min_length)
    return data
def output_results_by_feature(data, group_col):
    dd = pd.DataFrame(data[group_col].drop_duplicates().sort_index(), 
                      columns = [group_col])
    str_group = get_grouping_col_name_string(group_col)
    keep_cols = [c for c in data.columns if str_group in c] +\
                [c for c in group_col if type(group_col) == list]
    dout = pd.merge(dd, data[keep_cols]).drop_duplicates(group_col)
    dout.to_csv('features_by_' + str_group + '.csv', index=False)
#%%
group_col = 'prop_id'
min_length = 10
# Average CTR and BTR by property id
train_data_in = stats_of_col_by_group(train_data_in, 
                              group_col, 'click_bool', 'mean', min_length)
train_data_in = stats_of_col_by_group(train_data_in,
                              group_col, 'booking_bool', 'mean', min_length)
# Mean/median/stdev of location score by property ID
curr_col = 'prop_location_score2'
train_data_in = stats_of_col_by_group(train_data_in, group_col, curr_col, 
                                      ['mean', 'std', 'median'], min_length)
output_results_by_feature(train_data_in, group_col)
#%%
group_col = ['srch_destination_id', 'prop_id']
train_data_in = stats_of_col_by_group(train_data_in, 
                                      group_col, 'position', ['std', 'median'], 5)
output_results_by_feature(train_data_in, group_col)
#%%
group_col = ['srch_destination_id', 'prop_id']
train_data_in = stats_of_col_by_group(train_data_in, 
                                      group_col, 'position', 'mean', 5)
output_results_by_feature(train_data_in, group_col)