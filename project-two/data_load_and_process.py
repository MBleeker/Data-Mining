import pandas as pd
import numpy as np
import time
from sklearn.ensemble import RandomForestRegressor
from query import *
from lambda_rank import *
from theano_NN_model import *

"""
This block of code reads in the data, and adds features like click-through and
book-through rate by hotel. These are seperate because they are calculated on
the entire training data (90% of it at least), and we don't want to calc this
every time... null_cols_to_fill is a list of column names that have their
missing values auto-filled by their overall mean.
"""
null_cols_to_fill = []
def merge_in_csv(data, fname):
    features = pd.read_csv(fname)
    # get names of columns containing at least one NaN...
    null_cols = features.columns[1:]
    data = data.merge(features, how='left')
    return data, null_cols
train_data_in = pd.read_csv('train_set_10pct_of_90pct.csv')
test_data_in = pd.read_csv('test_set_10pct.csv')
train_data_in, null_cols_to_fill = merge_in_csv(train_data_in, 
                                                'features_by_prop_id.csv')
test_data_in, null_cols_to_fill = merge_in_csv(test_data_in, 
                                               'features_by_prop_id.csv')
"""
Class to store training and test data, preprocess it, pull out matrices, etc.
"""
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
    ...Might add something later to auto fillna with mean any cols with NaNs...
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
    def predict_missing(self, data, colname, pred_cols = None, n_trees = 100):
        now = time.time()
        not_missing = ~data['_missing_idxs_' + colname]
        model = RandomForestRegressor(n_estimators=n_trees)
        if pred_cols != None:
            X = data.loc[not_missing,pred_cols]
        else:
            X = data[not_missing].drop(self.drop_cols + [colname], axis=1)
        X = X.values
        y = data.loc[not_missing,colname].values
        model.fit(X, y)
        if pred_cols != None:
            X = data.loc[~not_missing,pred_cols]
        else:
            X = data[~not_missing].drop(self.drop_cols + [colname], axis=1)
        X = X.values
        y = model.predict(X)
        data.loc[:,colname + '_pred_missing'] = np.nan
        data.loc[~not_missing,colname + '_pred_missing'] = y
        data.loc[not_missing, colname + '_pred_missing'] = \
            data.loc[not_missing,colname]
        print 'took ' + str(np.round((time.time()-now)/60,2)) + ' minutes ' + \
            'to fill in missing values for column: ' + colname
        return data
    def make_isnull_column(self, data, colname, method = -1, 
                           is_zero_na = False):
        if is_zero_na:
            data.loc[:,colname + '_isnull'] = data.loc[:,colname].values == 0
            data = self.fill_zeroes(data, colname, method)
        else:
            data.loc[:,colname + '_isnull'] = \
                np.isnan(data.loc[:,colname].values)
            data = self.fill_nulls(data, colname, method)
        return data
    def cat_to_prob(self, data, colname):
        grouped = self.train_data.groupby(colname)
        ctr = grouped.apply(lambda x: sum(x['click_bool'])*1./len(x))
        data = data.merge(pd.DataFrame(ctr, columns = [colname + '_ctr']), 
                           right_index = True, left_on = colname)
        btr = grouped.apply(lambda x: sum(x['booking_bool'])*1./len(x))
        data = data.merge(pd.DataFrame(btr, columns = [colname + '_btr']), 
                           right_index = True, left_on = colname)
        return data
    def fill_nulls(self, data, colname, method = 'mean'):
        if method == 'mean':
            data.loc[:,colname] = \
                data.loc[:,colname].fillna(self.train_data[colname].mean())
        elif method == 'median':
            data.loc[:,colname] = \
                data.loc[:,colname].fillna(self.train_data[colname].median())
        elif method == 'mode':
            data.loc[:,colname] = \
                data.loc[:,colname].fillna(self.train_data[colname].mode()[0])
        elif method == 'min':
            data.loc[:,colname] = \
                data.loc[:,colname].fillna(self.train_data[colname].min())
        elif method == 'max':
            data.loc[:,colname] = \
                data.loc[:,colname].fillna(self.train_data[colname].max())
        else:
            data.loc[:,colname] = data.loc[:,colname].fillna(method)
        return data
    def fill_zeroes(self, data, colname, method = 'mean'):
        if method == 'mean':
            data.loc[:,colname] = data.loc[:,colname].replace(0, 
                self.train_data.loc[self.train_data[colname]!=0,colname].mean())
        elif method == 'median':
            data.loc[:,colname] = data.loc[:,colname].replace(0, 
                self.train_data[colname][
                    self.train_data[colname]!=0].median())
        elif method == 'mode':
            data.loc[:,colname] = data.loc[:,colname].replace(0, 
                self.train_data[colname][\
                    self.train_data[colname]!=0].mode()[0])
        elif method == 'min':
            data.loc[:,colname] = data.loc[:,colname].replace(0, 
                self.train_data[colname][self.train_data[colname]!=0].min())
        elif method == 'max':
            data.loc[:,colname] = data.loc[:,colname].replace(0, 
                self.train_data[colname][self.train_data[colname]!=0].max())
        else:
            data.loc[:,colname] = data.loc[:,colname].replace(0, method)
        return data

d = DataContainer(train_data=train_data_in, test_data=test_data_in,
                  null_cols_to_fill=null_cols_to_fill)
d.get_downsampled_data(6, propn = 1.)
d.pp_data = d.preprocess(d.pp_data, option=1)
d.test_data = d.preprocess(d.test_data, option=1)
#%%
# Fit an RF model and predict test set
# Got 0.4443 on the complete data set
model_rf = RandomForestRegressor(n_estimators=100)
model_rf.fit(*d.get_Xyq('train','Xy'))
preds = model_rf.predict(d.get_Xyq('test','X'))
d.test_data['pred_rel'] = preds
result = ndcg_of_df(d.test_data, plus_random=True)
print 'RF model NDCG: ', result
#%%
# Fit a LambdaRank model
# Got .47 with 10 epochs and lr = 0.0001
lr = 0.0001
total_epochs = 10
num_features = d.get_Xyq('test','X').shape[1]
model_nn = LambdaRank(num_features, 'LambdaRank', lr, 
                   train_queries=Queries(*d.get_Xyq('train')),
                   width=[30,10], normalise=False, drop_hidden=0, drop_input=0)
model_nn.train_with_queries(num_epochs=total_epochs)
preds = model_nn.score_matrix(d.get_Xyq('test', 'X'))
d.test_data['pred_rel'] = preds
result = ndcg_of_df(d.test_data, plus_random=False)
print 'LambdaRank model NDCG: ', result