from sklearn.ensemble import RandomForestRegressor
import numpy as np
import time
from ndcg import *

def make_predictions(d, model):
    now = time.time()
    X_train = d.pp_data.drop(d.drop_cols, axis=1).astype('float32').values
    y_train = np.maximum(d.pp_data['click_bool'].values, 
                   d.pp_data['booking_bool'].values * 5)
    model.fit(X_train, y_train)
    print 'took ' + str(np.round((time.time()-now)/60,2)) + ' minutes ' + \
        'to fit model'
    now = time.time()
    if 'pred_rel' in d.test_data.columns:
        test_drops = d.drop_cols + ['pred_rel']
    else:
        test_drops = d.drop_cols
    X_test = d.test_data.drop(test_drops, axis=1).values
    d.test_data.loc[:,'pred_rel'] = model.predict(X_test)
    grouped = d.test_data.groupby('srch_id')
    ndcgs = grouped.apply(lambda x: ndcg_of_table_chunk(x))
    print 'took ' + str(np.round((time.time()-now)/60,2)) + ' minutes ' + \
        'to make predictions from model'
    print 'mean NDCG over predicted test set:'
    print ndcgs.mean()
    print 'mean NDCG using random order (for comparison):'
    print grouped.apply(lambda x: ndcg_of_table_chunk(x, random_order=True)).mean()
    imps = np.argsort(-model.feature_importances_)
    imps = d.pp_data.drop(d.drop_cols, axis=1).columns[imps]
    return model, imps