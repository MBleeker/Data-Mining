import pandas as pd
import numpy as np

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
        train_subset = pd.unique(self.train_data_full['srch_id'])
        train_subset = np.random.choice(train_subset, 
                                        size = int(len(train_subset) * 
                                                   test_propn),
                                        replace = False)
        train_idxs = self.train_data_full['srch_id'].isin(train_subset)
        test_idxs = ~self.train_data_full['srch_id'].isin(train_subset)
        self.train_data = self.train_data_full[train_idxs]
        self.test_data = self.train_data_full[test_idxs]
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
        y = np.maximum(self.pp_data['click_bool'].values, 
                       self.pp_data['booking_bool'].values * 5)
        return X.astype(np.float32), y.astype(np.float32)
    def preprocess(self, data, option = 0):
        data['prop_review_score'] = data['prop_review_score'].fillna(0)
        if option == 0:
            for col in self.null_cols:
                data = self.fill_nulls(data, col)
            for col in self.zero_cols:
                data = self.fill_zeroes(data, col)
            self.drop_cols = ['srch_id','date_time','click_bool','position',
                                    'booking_bool','gross_bookings_usd']
        if option == 1:
            self.drop_cols = ['srch_id','date_time','click_bool','position',
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
            data['prop_log_historical_price'][price_replace_idxs] = \
                                          data['price_usd'][price_replace_idxs]
            data = self.cat_to_prob(data, 'srch_destination_id')
            self.drop_cols.append('srch_destination_id')
            self.drop_cols.append('srch_query_affinity_score')
            data = self.make_isnull_column(data, 'orig_destination_distance',
                                           -99999999., False)
            for col in self.comp_cols:
                data = self.fill_nulls(data, col, 0)
            cols = [c for c in data.columns \
                    if '_rate' in c and '_diff' not in c]
            data['comp_rate'] = data[cols].sum(axis=1)
            cols = [c for c in data.columns if '_diff' in c]
            data['comp_rate_percent_diff'] = data[cols].sum(axis=1)
            cols = [c for c in data.columns if '_inv' in c]
            data['comp_inv'] = data[cols].sum(axis=1)
            self.drop_cols += self.comp_cols
        return data
    def get_downsampled_data(self, ratio, propn = 0.01):
        subset = pd.unique(self.train_data['srch_id'])
        subset = np.random.choice(subset, size = int(len(subset) * propn))
        self.pp_data = self.train_data[self.train_data['srch_id'].isin(subset)]
        def downsample(d, ratio):
            n_clicks = np.sum(d['click_bool'])
            if n_clicks == 0:
                return d[np.random.choice(
                                      np.arange(len(d)) + min(d.index), ratio)]
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
    def make_isnull_column(self, data, col_name, method = -1, 
                           is_zero_na = False):
        if is_zero_na:
            data[col_name + '_isnull'] = data[col_name].values == 0
            data = self.fill_zeroes(data, col_name, method)
        else:
            data[col_name + '_isnull'] = np.isnan(data[col_name].values)
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
            data[col_name] = \
                data[col_name].fillna(self.train_data[col_name].mean())
        elif method == 'median':
            data[col_name] = \
                data[col_name].fillna(self.train_data[col_name].median())
        elif method == 'mode':
            data[col_name] = \
                data[col_name].fillna(self.train_data[col_name].mode()[0])
        elif method == 'min':
            data[col_name] = \
                data[col_name].fillna(self.train_data[col_name].min())
        elif method == 'max':
            data[col_name] = \
                data[col_name].fillna(self.train_data[col_name].max())
        else:
            data[col_name] = data[col_name].fillna(method)
        return data
    def fill_zeroes(self, data, col_name, method = 'mean'):
        if method == 'mean':
            data[col_name] = data[col_name].replace(0, 
                self.train_data[col_name][self.train_data[col_name]!=0].mean())
        elif method == 'median':
            data[col_name] = data[col_name].replace(0, 
                self.train_data[col_name][\
                    self.train_data[col_name]!=0].median())
        elif method == 'mode':
            data[col_name] = data[col_name].replace(0, 
                self.train_data[col_name][\
                    self.train_data[col_name]!=0].mode()[0])
        elif method == 'min':
            data[col_name] = data[col_name].replace(0, 
                self.train_data[col_name][self.train_data[col_name]!=0].min())
        elif method == 'max':
            data[col_name] = data[col_name].replace(0, 
                self.train_data[col_name][self.train_data[col_name]!=0].max())
        else:
            data[col_name] = data[col_name].replace(0, method)
        return data

d = DataContainer(train_data = train_data_in, test_propn=0.9)
d.get_downsampled_data(4, propn = 0.5)
d.pp_data = d.preprocess(d.pp_data, option=1)
d.test_data = d.preprocess(d.test_data, option=1)