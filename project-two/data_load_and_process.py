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
        self.train_data = test_data
        self.train_size = train_size
        self.test_size = test_size
        self.train_drop_cols = ['srch_id','date_time','click_bool',
                                'booking_bool','gross_bookings_usd']
    def split_train_data(self):
        self.train_data_train_subset = \
            self.train_data[self.train_data['srch_id'] < self.train_size]
        self.train_data_test_subset = \
            self.train_data[np.bitwise_and(self.train_data['srch_id']>=self.train_size, 
                                self.train_data['srch_id']<(self.train_size+self.test_size))]
    def get_train_Xy(self, data):
        data = self.preprocess_training_data(data)
        X = data.drop(self.drop_cols, axis=1).values
        y = np.maximum(data['click_bool'].values, 
                       data['booking_bool'].values * 5)
        return X, y
    def preprocess_training_data(self, data):
        data = data.fillna(data.mean())
        return data

data_cont = DataContainer(train_data = train_data)
data_cont.split_train_data()
X_train, y_train = data_cont.get_train_Xy(data_cont.train_data_train_subset)
X_test, y_test = data_cont.get_train_Xy(data_cont.train_data_test_subset)