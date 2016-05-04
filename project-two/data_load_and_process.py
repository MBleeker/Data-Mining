import pandas as pd
import numpy as np
import time
from make_predictions import *
from sklearn.ensemble import RandomForestRegressor
import query
import lambda_rank

#==============================================================================
# # Code to create 10pct training subset
# train_data_in = pd.read_csv('training_set_VU_DM_2014.csv')
# train_data_in['prop_log_historical_price'] = \
#     np.exp(train_data_in['prop_log_historical_price']).replace(1, 0)
# np.random.seed(1)
# train_subset = pd.unique(train_data_in['srch_id'])
# train_subset = np.random.choice(train_subset, 
#                                 size = int(len(train_subset) * 0.1), 
#                                 replace = False)
# train_idxs = train_data_in['srch_id'].isin(train_subset)
# train_data_in[train_idxs].to_csv('training_set_VU_DM_2014_10pct.csv', 
#                                  index = False)
#==============================================================================
train_data_in = pd.read_csv('training_set_VU_DM_2014_10pct.csv')
#%%
"""
Making a class to store all the training and test data.
I guess the idea is to have methods to preprocess the raw .csv data,
and to pull out X and y training / test matrices.
"""
class DataContainer:
    def __init__(self, train_data = None, test_data = None, test_propn=0.98):
        np.random.seed(1)
        self.train_data_full = train_data
        train_subset = pd.unique(self.train_data_full.loc[:,'srch_id'])
        train_subset = np.random.choice(
                      train_subset, size = int(len(train_subset) * test_propn),
                      replace = False)
        train_idxs = self.train_data_full.loc[:,'srch_id'].isin(train_subset)
        test_idxs = ~self.train_data_full.loc[:,'srch_id'].isin(train_subset)
        self.train_data = self.train_data_full.loc[train_idxs,:]
        self.test_data = self.train_data_full.loc[test_idxs,:]
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
    def get_Xy(self):
        X = self.pp_data.drop(self.drop_cols, axis=1).values
        y = np.maximum(self.pp_data.loc[:,'click_bool'].values, 
                       self.pp_data.loc[:,'booking_bool'].values * 5)
        return X.astype(np.float32), y.astype(np.float32)
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
            self.drop_cols += ['srch_id','date_time','click_bool','position',
                               'booking_bool','gross_bookings_usd']
            data = self.cat_to_prob(data, 'visitor_location_country_id')
            self.drop_cols.append('visitor_location_country_id')
            self.drop_cols.append('visitor_hist_starrating')
            self.drop_cols.append('visitor_hist_adr_usd')
            data = self.cat_to_prob(data, 'prop_country_id')
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
            data = self.cat_to_prob(data, 'srch_destination_id')
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
        data.merge(pd.DataFrame(ctr, columns = [col_name + '_ctr']), 
                           right_index = True, left_on = col_name)
        btr = grouped.apply(lambda x: sum(x['booking_bool'])*1./len(x))
        data.merge(pd.DataFrame(btr, columns = [col_name + '_btr']), 
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

d = DataContainer(train_data = train_data_in, test_propn=0.9)
d.get_downsampled_data(6, propn = .1)
d.pp_data = d.preprocess(d.pp_data, option=1)
d.test_data = d.preprocess(d.test_data, option=1)

# Fit an RF model and predict test set
# Got 0.4445 on the complete data set
#model_rf = RandomForestRegressor(n_estimators=1000)
#model_rf, imps_rf = make_predictions(d, model_rf)

"""
# Predict missing value for 'prop_location_score2' and drop original column
# Can just add these lines into a particular preprocessing option I suppose,
# after doing all other preprocessing
# Got .4395 on the complete data set
d.pp_data = d.predict_missing(d.pp_data, 'prop_location_score2')
d.test_data = d.predict_missing(d.test_data, 'prop_location_score2')
d.drop_cols += ['prop_location_score2']
# Fit an RF model again and predict test set
model_rf = RandomForestRegressor(n_estimators=1000)
model_rf, imps_rf = make_predictions(d, model_rf)
# Predict missing 'prop_location_score2' using just 'prop_location_score1'
# Got .4392 on the complete data set
d.pp_data = d.predict_missing(d.pp_data, 'prop_location_score2',
                              pred_cols=['prop_location_score1'])
d.test_data = d.predict_missing(d.test_data, 'prop_location_score2',
                                pred_cols=['prop_location_score1'])
d.drop_cols += ['prop_location_score2']
# Fit an RF model again and predict test set
model_rf = RandomForestRegressor(n_estimators=1000)
model_rf, imps_rf = make_predictions(d, model_rf)
"""
#%%
def get_queries(df, drop_cols):
    if 'pred_rel' in df.columns:
        X = df.drop(drop_cols + ['pred_rel'], axis=1).astype('float32').values
    else:
        X = df.drop(drop_cols, axis=1).astype('float32').values
    print X.shape
    y = np.maximum(df['click_bool'].values, 
                   df['booking_bool'].values * 5)
    qry_ids = df['srch_id'].values
    return Queries(X, y, qry_ids)

queries_train = get_queries(d.pp_data, d.drop_cols)
queries_test = get_queries(d.test_data, d.drop_cols)

lr = 0.00001
total_epochs = 50
model = LambdaRank(24, 'LambdaRank', lr)
model.train_with_queries(queries_train, total_epochs)

# Function to get NDCG score of a model 'l_rank' trying to predict relevance 
# scores for a set of queries 'qry_set'
def get_predicted_relevance_nn(qry_set, l_rank):
    qids = qry_set.get_qids()
    init = False
    for q in range(len(qids)):
        test_query = qry_set.get_query(qids[q])
        #print sum(test_query.get_labels())
        scores = l_rank.score(test_query)
        scores = np.array([scores[i][0] for i in range(len(scores))])
        if not init:
            scores_out = scores
            init = True
        else:
            scores_out = np.hstack((scores_out, scores))
        #print "ndcg = " + str(ndcg[q])
    return scores_out

preds = get_predicted_relevance_nn(queries_test, model)
d.test_data['pred_rel'] = preds
grouped = d.test_data.groupby('srch_id')
ndcgs = grouped.apply(lambda x: ndcg_of_table_chunk(x))
print 'mean NDCG over predicted test set:'
print ndcgs.mean()
print 'mean NDCG using random order (for comparison):'
#%%print grouped.apply(lambda x: ndcg_of_table_chunk(x, random_order=True)).mean()