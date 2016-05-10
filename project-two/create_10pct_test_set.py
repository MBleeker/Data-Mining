import pandas as pd
import numpy as np
"""
Idea is we extract a random 10% of queries and treat it as if it were the
Kaggle test set - to ensure this is the case I will remove the columns from it
that do not appear in the test set - click_bool, booking_bool, etc.
The other 90% will be saved as the 'new' training set.
"""
train_data_in = pd.read_csv('training_set_VU_DM_2014.csv')
train_data_in['prop_log_historical_price'] = \
    np.exp(train_data_in['prop_log_historical_price']).replace(1, 0)
np.random.seed(0)
train_subset = pd.unique(train_data_in['srch_id'])
train_subset = np.random.choice(train_subset, 
                                size = int(len(train_subset) * 0.1), 
                                replace = False)
test_idxs = train_data_in['srch_id'].isin(train_subset)
train_data_in[test_idxs].to_csv('test_set_10pct.csv', index = False)
train_data_in[~test_idxs].to_csv('train_set_90pct.csv', index = False)

# Now create a 10% subset of the 90% training data to train on, as 90% is too much data!
train_data_in = pd.read_csv('train_set_90pct.csv')
np.random.seed(0)
train_subset = pd.unique(train_data_in['srch_id'])
train_subset = np.random.choice(train_subset, 
                                size = int(len(train_subset) * 0.1), 
                                replace = False)
train_idxs = train_data_in['srch_id'].isin(train_subset)
train_data_in[train_idxs].to_csv('train_set_10pct_of_90pct.csv', index = False)