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
class DataContainer:
    def __init__(self, train_data = None, test_data = None, 
                 null_cols_to_fill = []):
        self.train_data = train_data
        self.test_data = test_data
        self.preprocessed = False
        self.comp_cols = ['comp1_rate','comp1_inv','comp1_rate_percent_diff',
                          'comp2_rate','comp2_inv','comp2_rate_percent_diff',
                          'comp3_rate','comp3_inv','comp3_rate_percent_diff',
                          'comp4_rate','comp4_inv','comp4_rate_percent_diff',
                          'comp5_rate','comp5_inv','comp5_rate_percent_diff',
                          'comp6_rate','comp6_inv','comp6_rate_percent_diff',
                          'comp7_rate','comp7_inv','comp7_rate_percent_diff',
                          'comp8_rate','comp8_inv','comp8_rate_percent_diff']
        self.null_cols = ['visitor_hist_starrating','visitor_hist_adr_usd',
                          'prop_location_score2','srch_query_affinity_score',
                          'orig_destination_distance'] + self.comp_cols
        self.zero_cols = ['prop_starrating', 'prop_review_score', 
                          'prop_log_historical_price']
        self.null_cols_to_fill = null_cols_to_fill
    """ 
    Function to get matrices from dataframes stored within this class object
    Arguments:
        String: train_or_test = 'train' OR 'test' -> get training or test matr?
        String: Xyq = 'Xyq', 'Xy', 'y', 'yq', etc -> what matrices to return?
    Returns:
        Requested numpy matrice(s) containing train or test data
    """
    def get_Xyq(self, train_or_test='train', ret='Xyq'):
        if train_or_test == 'train':
            df = self.pp_data
        else:
            df = self.test_data
        if 'pred_rel' in df.columns:
            X = df.drop(self.drop_cols + ['pred_rel'], axis=1).values
        else:
            X = df.drop(self.drop_cols, axis=1).values
        y = np.maximum(df.loc[:,'click_bool'].values, 
                       df.loc[:,'booking_bool'].values * 5)
        q = df['srch_id'].values
        if ret == 'X':
            return X.astype(np.float32)
        if ret == 'y':
            return y.astype(np.float32)
        if ret == 'q':
            return q.astype(np.int32)
        if ret == 'Xy':
            return X.astype(np.float32), y.astype(np.float32)
        if ret == 'Xq':
            return X.astype(np.float32), q.astype(np.int32)
        if ret == 'yq':
            return y.astype(np.float32), q.astype(np.int32)
        if ret == 'Xyq':
            return X.astype(np.float32), y.astype(np.float32), \
                    q.astype(np.int32)
    def preprocess(self, data, option = 0):
        data.loc[:,'prop_review_score'] = \
            data.loc[:,'prop_review_score'].fillna(0)
        self.drop_cols = []
        self.drop_cols += ['srch_id','date_time','click_bool','position',
                           'booking_bool','gross_bookings_usd']
        for col in self.null_cols:
            data.loc[:,'_missing_idxs_' + col] = data.loc[:,col].isnull()
            self.drop_cols += ['_missing_idxs_' + col]
        for col in self.zero_cols:
            data.loc[:,'_missing_idxs_' + col] = (data.loc[:,col] == 0)
            self.drop_cols += ['_missing_idxs_' + col]
        if option == 0:
            for col in self.null_cols:
                data = self.fill_nulls(data, col)
            for col in self.zero_cols:
                data = self.fill_zeroes(data, col)
        if option == 1:
            for col in self.null_cols_to_fill:
                data = self.fill_nulls(data, col)
            self.drop_cols += ['srch_id','date_time','click_bool','position',
                               'booking_bool','gross_bookings_usd']
            self.drop_cols.append('visitor_location_country_id')
            self.drop_cols.append('visitor_hist_starrating')
            self.drop_cols.append('visitor_hist_adr_usd')
            self.drop_cols.append('prop_country_id')
            self.drop_cols.append('prop_id')
            data = self.make_isnull_column(data, 'prop_starrating', 'mean', 
                                           True)
            data = self.make_isnull_column(data, 'prop_review_score', 'mean', 
                                           True)
            data = self.fill_nulls(data, 'prop_location_score2')
            price_replace_idxs = data['prop_log_historical_price']==0
            data.loc[price_replace_idxs,'prop_log_historical_price']= \
                                       data.loc[price_replace_idxs,'price_usd']
            self.drop_cols.append('srch_destination_id')
            self.drop_cols.append('srch_query_affinity_score')
            data = self.make_isnull_column(data, 'orig_destination_distance',
                                           -99999999., False)
            for col in self.comp_cols:
                data = self.fill_nulls(data, col, 0)
            cols = [c for c in data.columns \
                    if '_rate' in c and '_diff' not in c]
            data.loc[:,'comp_rate'] = data.loc[:,cols].sum(axis=1)
            cols = [c for c in data.columns if '_diff' in c]
            data.loc[:,'comp_rate_percent_diff'] = data[cols].sum(axis=1)
            cols = [c for c in data.columns if '_inv' in c]
            data.loc[:,'comp_inv'] = data.loc[:,cols].sum(axis=1)
            self.drop_cols += self.comp_cols
        return data
    def get_downsampled_data(self, ratio, propn = 0.01):
        subset = pd.unique(self.train_data['srch_id'])
        subset = np.random.choice(subset, size=int(len(subset) * propn), 
                                  replace=False)
        subset = self.train_data['srch_id'].isin(subset)
        self.pp_data = self.train_data[subset]
        def downsample(d, ratio):
            n_clicks = np.sum(d['click_bool'])
            if n_clicks == 0:
                inc = np.random.choice(np.arange(len(d)) + min(d.index), ratio)
                return d.loc[inc,:]
            elif len(d) == n_clicks:
                return d
            elif n_clicks / (len(d) - n_clicks) < 1. / ratio:
                drop_rows = np.where(d['click_bool'].values==0)[0]
                drop_rows = np.random.choice(drop_rows, 
                                     size = len(drop_rows) - n_clicks * ratio, 
                                     replace = False)
                return d.drop(drop_rows + min(d.index), axis = 0)
            return d
        grouped = self.pp_data.groupby('srch_id')
        self.pp_data = grouped.apply(lambda x: downsample(x, ratio))
    def predict_missing(self, data, col_name, pred_cols = None, n_trees = 100):
        now = time.time()
        not_missing = ~data['_missing_idxs_' + col_name]
        model = RandomForestRegressor(n_estimators=n_trees)
        if pred_cols != None:
            X = data.loc[not_missing,pred_cols]
        else:
            X = data[not_missing].drop(self.drop_cols + [col_name], axis=1)
        X = X.values
        y = data.loc[not_missing,col_name].values
        model.fit(X, y)
        if pred_cols != None:
            X = data.loc[~not_missing,pred_cols]
        else:
            X = data[~not_missing].drop(self.drop_cols + [col_name], axis=1)
        X = X.values
        y = model.predict(X)
        data.loc[:,col_name + '_pred_missing'] = np.nan
        data.loc[~not_missing,col_name + '_pred_missing'] = y
        data.loc[not_missing, col_name + '_pred_missing'] = \
            data.loc[not_missing,col_name]
        print 'took ' + str(np.round((time.time()-now)/60,2)) + ' minutes ' + \
            'to fill in missing values for column: ' + col_name
        return data
        
    def predict_missingKNN(self, data, col_name, pred_cols = None, K = 1):
        now = time.time()
        not_missing = ~data['_missing_idxs_' + col_name]
        model =  KNeighborsClassifier(n_neighbors=k)
        if pred_cols != None:
            X = data.loc[not_missing,pred_cols]
        else:
            X = data[not_missing].drop(self.drop_cols + [col_name], axis=1)
        X = X.values
        y = data.loc[not_missing,col_name].values
        model.fit(X, y)
        if pred_cols != None:
            X = data.loc[~not_missing,pred_cols]
        else:
            X = data[~not_missing].drop(self.drop_cols + [col_name], axis=1)
        X = X.values
        y = model.predict(X)
        data.loc[:,col_name + '_pred_missing'] = np.nan
        data.loc[~not_missing,col_name + '_pred_missing'] = y
        data.loc[not_missing, col_name + '_pred_missing'] = \
            data.loc[not_missing,col_name]
        print 'took ' + str(np.round((time.time()-now)/60,2)) + ' minutes ' + \
            'to fill in missing values for column: ' + col_name
        return data
        
    def make_isnull_column(self, data, col_name, method = -1, 
                           is_zero_na = False):
        if is_zero_na:
            data.loc[:,col_name + '_isnull'] = data.loc[:,col_name].values == 0
            data = self.fill_zeroes(data, col_name, method)
        else:
            data.loc[:,col_name + '_isnull'] = np.isnan(data.loc[:,col_name].values)
            data = self.fill_nulls(data, col_name, method)
        return data
    def cat_to_prob(self, data, col_name):
        grouped = self.train_data.groupby(col_name)
        ctr = grouped.apply(lambda x: sum(x['click_bool'])*1./len(x))
        data = data.merge(pd.DataFrame(ctr, columns = [col_name + '_ctr']), 
                           right_index = True, left_on = col_name)
        btr = grouped.apply(lambda x: sum(x['booking_bool'])*1./len(x))
        data = data.merge(pd.DataFrame(btr, columns = [col_name + '_btr']), 
                           right_index = True, left_on = col_name)
        return data
    def fill_nulls(self, data, col_name, method = 'mean'):
        if method == 'mean':
            data.loc[:,col_name] = \
                data.loc[:,col_name].fillna(self.train_data[col_name].mean())
        elif method == 'median':
            data.loc[:,col_name] = \
                data.loc[:,col_name].fillna(self.train_data[col_name].median())
        elif method == 'mode':
            data.loc[:,col_name] = \
                data.loc[:,col_name].fillna(self.train_data[col_name].mode()[0])
        elif method == 'min':
            data.loc[:,col_name] = \
                data.loc[:,col_name].fillna(self.train_data[col_name].min())
        elif method == 'max':
            data.loc[:,col_name] = \
                data.loc[:,col_name].fillna(self.train_data[col_name].max())
        else:
            data.loc[:,col_name] = data.loc[:,col_name].fillna(method)
        return data
    def fill_zeroes(self, data, col_name, method = 'mean'):
        if method == 'mean':
            data.loc[:,col_name] = data.loc[:,col_name].replace(0, 
                self.train_data.loc[self.train_data[col_name]!=0,col_name].mean())
        elif method == 'median':
            data.loc[:,col_name] = data.loc[:,col_name].replace(0, 
                self.train_data[col_name][
                    self.train_data[col_name]!=0].median())
        elif method == 'mode':
            data.loc[:,col_name] = data.loc[:,col_name].replace(0, 
                self.train_data[col_name][\
                    self.train_data[col_name]!=0].mode()[0])
        elif method == 'min':
            data.loc[:,col_name] = data.loc[:,col_name].replace(0, 
                self.train_data[col_name][self.train_data[col_name]!=0].min())
        elif method == 'max':
            data.loc[:,col_name] = data.loc[:,col_name].replace(0, 
                self.train_data[col_name][self.train_data[col_name]!=0].max())
        else:
            data.loc[:,col_name] = data.loc[:,col_name].replace(0, method)
        return data
'visitor_hist_starrating'