import pandas as pd
import numpy as np

train_data = pd.read_csv('training_set_VU_DM_2014.csv')
#%%
"""
Making a class to store all the training and test data.
I guess the idea is to have methods to preprocess the raw .csv data,
and to pull out X and y training / test matrices.
"""
class DataContainer:
    def __init__(self, train_data = None, test_data = None, train_size = 50, 
                 test_size = 50):
        self.train_data = train_data
        self.test_data = test_data
        self.train_drop_cols = ['srch_id','date_time','click_bool',
                                'booking_bool','gross_bookings_usd']
        self.preprocessed = False
        self.null_cols = ['visitor_hist_starrating','visitor_hist_adr_usd',
                          'prop_location_score2','srch_query_affinity_score',
                          'comp1_rate','comp1_inv','comp1_rate_percent_diff',
                          'comp2_rate','comp2_inv','comp2_rate_percent_diff',
                          'comp3_rate','comp3_inv','comp3_rate_percent_diff',
                          'comp4_rate','comp4_inv','comp4_rate_percent_diff',
                          'comp5_rate','comp5_inv','comp5_rate_percent_diff',
                          'comp6_rate','comp6_inv','comp6_rate_percent_diff',
                          'comp7_rate','comp7_inv','comp7_rate_percent_diff',
                          'comp8_rate','comp8_inv','comp8_rate_percent_diff']
        self.zero_cols = ['prop_starrating', 'prop_review_score', 
                          'prop_log_historical_price']
    def get_Xy(self, data):
        X = self.pp_data.drop(self.drop_cols, axis=1).values
        y = np.maximum(self.pp_data['click_bool'].values, 
                       self.pp_data['booking_bool'].values * 5)
        return X.astype(np.float32), y.astype(np.float32)
    def preprocess(self, data, option = 0):
        if option == 0:
            self.pp_data = self.get_downsampled_data()
            for col in self.null_cols:
                self.fill_nulls(col)
            for col in self.zero_cols:
                self.fill_zeroes(col)
            self.train_drop_cols = ['srch_id','date_time','click_bool',
                                    'booking_bool','gross_bookings_usd']
    def get_downsampled_data(self):
        pass
    def make_isnull_column(self, col_name, replace_val = -1, zeros = False):
        if zeros:
            self.pp_data[col_name + '_isnull'] = \
                self.pp_data[col_name].values == 0
            self.pp_data[col_name].replace(0, replace_val)
        else:
            self.pp_data[col_name + '_isnull'] = \
                np.isnan(self.pp_data[col_name].values)
            self.pp_data[col_name].fillna(replace_val)
    def cat_to_prob(self, col_name):
        grouped = self.train_data.groupby(col_name)
        ctr = grouped.apply(lambda x: sum(x['click_bool'])*1./len(x))
        self.pp_data.merge(pd.DataFrame(ctr, columns = [col_name + '_ctr']), 
                           right_index = True, left_on = col_name)
        btr = grouped.apply(lambda x: sum(x['booking_bool'])*1./len(x))
        self.pp_data.merge(pd.DataFrame(btr, columns = [col_name + '_btr']), 
                           right_index = True, left_on = col_name)
    def fill_nulls(self, col_name, method = 'mean'):
        if method == 'mean':
            self.pp_data[col_name].fillna(self.train_data[col_name].mean())
        elif method == 'median':
            self.pp_data[col_name].fillna(self.train_data[col_name].median())
        elif method == 'mode':
            self.pp_data[col_name].fillna(self.train_data[col_name].mode()[0])
        elif method == 'min':
            self.pp_data[col_name].fillna(self.train_data[col_name].min())
        elif method == 'max':
            self.pp_data[col_name].fillna(self.train_data[col_name].max())
        else:
            self.pp_data[col_name].fillna(method)
    def fill_zeroes(self, col_name, method = 'mean'):
        if method == 'mean':
            self.pp_data[col_name].replace(0, 
                self.train_data[col_name][train_data[col_name]==0].mean())
        elif method == 'median':
            self.pp_data[col_name].replace(0, 
                self.train_data[col_name][train_data[col_name]==0].median())
        elif method == 'mode':
            self.pp_data[col_name].replace(0, 
                self.train_data[col_name][train_data[col_name]==0].mode()[0])
        elif method == 'min':
            self.pp_data[col_name].replace(0, 
                self.train_data[col_name][train_data[col_name]==0].min())
        elif method == 'max':
            self.pp_data[col_name].replace(0, 
                self.train_data[col_name][train_data[col_name]==0].max())
        else:
            self.pp_data[col_name].replace(0, method)
        

data_cont = DataContainer(train_data = train_data)
X, y = data_cont.get_Xy()
#%%
small_data = train_data[:10000]
col_name = 'visitor_location_country_id'
grouped = small_data.groupby(col_name)
res = grouped.apply(lambda x: sum(x['booking_bool'])*1./len(x))
small_data.merge(pd.DataFrame(res, columns = [col_name + '_btr']), 
                 right_index = True, left_on = col_name)
#%%
v = 'prop_starrating'
#print (train_data[v])
print sum(train_data[v]==0)*1./len(train_data)
print sum(np.isnan(train_data[v].values))*1./len(train_data)
#%%
import matplotlib.pyplot as plt
cols = ['visitor_hist_starrating','visitor_hist_adr_usd','prop_starrating',
        'prop_review_score','prop_location_score2',
        'prop_log_historical_price','srch_query_affinity_score','comp1_rate',
        'comp1_inv','comp1_rate_percent_diff']
nans = np.isnan(train_data[cols])
nans = pd.DataFrame(nans)
corr = nans.corr()
size = corr.shape[0]
fig, ax = plt.subplots(figsize=(size, size))
cax = ax.matshow(corr)
fig.colorbar(cax, ticks=[-1, 0, 1])
plt.xticks(range(len(corr.columns)), corr.columns);
plt.yticks(range(len(corr.columns)), corr.columns);

plt.show()

#%%(27-train_data.drop(train_data.columns[27:], axis=1).count(axis=1)).unique()
#%%


#==============================================================================
#     def split_train_data(self):
#         self.train_data_train_subset = \
#             self.train_data[self.train_data['srch_id'] < self.train_size]
#         self.train_data_test_subset = \
#             self.train_data[np.bitwise_and(self.train_data['srch_id']>=self.train_size, 
#                                 self.train_data['srch_id']<(self.train_size+self.test_size))]
#==============================================================================
