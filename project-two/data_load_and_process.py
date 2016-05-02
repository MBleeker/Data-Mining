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
        
    def get_Xy(self, data):
        if not self.preprocessed:
            data = self.preprocess(data)
        X = data.drop(self.drop_cols, axis=1).values
        y = np.maximum(data['click_bool'].values, 
                       data['booking_bool'].values * 5)
        return X, y
    def preprocess(self, data):
        data = data.fillna(data.mean())
        self.
        self.preprocessed = True
        return data
        
data_cont = DataContainer(train_data = train_data)
X, y = data_cont.get_Xy()

#%%
v = 'comp1_rate_percent_diff'
print (train_data[v])
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
