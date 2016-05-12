# -*- coding: utf-8 -*-
"""
Created on Fri May  6 12:51:37 2016

@author: liam
"""
import pandas as pd
import numpy as np
min_length = 10

def get_grouping_col_name_string(grouping_col_name):
    if type(grouping_col_name) == list:
        grouping_col_name = '_and_'.join(grouping_col_name)
    return grouping_col_name
# Use to make features like: 'stdev of prop_location_score2 by prop_id'
def stat_of_col_by_group(data, grouping_col_name, val_col_name, 
                          fns, fn_dict, min_length):
    str_group = get_grouping_col_name_string(grouping_col_name)
    if type(val_col_name) != list:
        val_col_name = [val_col_name]
    grouped = data.groupby(grouping_col_name)
    out = []
    start = True
    for c in val_col_name:
        for fn_name in fns:
            if fn_name in fn_dict:
                fn = fn_dict[fn_name]
                if start:
                    out = pd.DataFrame(grouped.apply(lambda x: fn(x[c], min_length)), columns = [c + '_' + fn_name + '_by_' + str_group])
                    start = False
                else:
                    curr = pd.DataFrame(grouped.apply(lambda x: fn(x[c], min_length)), columns = [c + '_' + fn_name + '_by_' + str_group])
                    out = out.merge(curr, left_index=True, right_index=True)
    return out
# Makes column applying every function in fns (if in fn_dict) to the values of
# val_col_name to each group specified by grouping_col_name
def stats_of_col_by_group(data, grouping_cols, val_cols, fns, min_length):
    def mean_if_long(x, min_length):
        if len(x) > min_length:
            return np.nanmean(x)
        else:
            return np.nan
    def std_if_long(x, min_length):
        if len(x) > min_length:
            return np.nanstd(x)
        else:
            return np.nan
    def median_if_long(x, min_length):
        if len(x) > min_length:
            return np.nanmedian(x)
        else:
            return np.nan
    fn_dict = {'mean' : mean_if_long, 
               'std' : std_if_long, 
               'median' : median_if_long}
    if type(fns) == str:
        fns = [fns]
    data = stat_of_col_by_group(data, grouping_cols, val_cols, fns, fn_dict, 
                                min_length)
    return data
def agg_vals_by_feature_to_csv(data, group_cols, val_cols, fns, min_length=10):
    stats_out = stats_of_col_by_group(train_data_in, group_cols, val_cols, 
                                      fns, min_length)
    str_group = get_grouping_col_name_string(group_col)
    stats_out.to_csv('features_by_' + str_group + '.csv', index=True)
#%%
# These features use the booking rate and click rate and position, so we should 
# not use the 'test' set when creating them...
train_data_in = pd.read_csv('train_set_90pct.csv')
# Average CTR and BTR by property id - will auto-out to .csv
group_col = ['prop_id']
val_cols = ['click_bool','booking_bool']
agg_vals_by_feature_to_csv(train_data_in, group_col, val_cols, 
                           ['mean'], min_length=5)
# Same for average position by srchDestID+PropID combined grouping
group_col = ['srch_destination_id', 'prop_id']
val_cols = ['position']
agg_vals_by_feature_to_csv(train_data_in, group_col, val_cols, 
                           ['mean', 'std', 'median'], min_length=5)
#%%
# These features we can use the full training set for training, and in final
# model we should use the full test set also...
train_data_in = pd.read_csv('training_set_VU_DM_2014.csv')
# Average of various features for each property id...
group_col = ['prop_id']
val_cols = ['prop_location_score2','price_usd','prop_starrating','prop_starrating']
# Drop unnecessary cols to save memory...
drop = [c for c in train_data_in.columns if c not in group_col and c not in val_cols and c not in ['srch_id']]
train_data_in=train_data_in.drop(drop, axis=1)
min_length = 10
#stats_out = stats_of_col_by_group(train_data_in, group_col, val_cols, 
#                              ['mean'], min_length)
agg_vals_by_feature_to_csv(train_data_in, group_col, val_cols,
                           ['mean', 'std', 'median'], min_length=10)
#%%
# Average/median/stdev price per search query
group_col = ['srch_id']
val_cols = ['price_usd']
agg_vals_by_feature_to_csv(train_data_in, group_col, val_cols, 
                           ['mean', 'std', 'median'], min_length=10)