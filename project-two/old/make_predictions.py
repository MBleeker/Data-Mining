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
    if 'pred_rel' in d.test_data.columns:
        test_drops = d.drop_cols + ['pred_rel']
    else:
        test_drops = d.drop_cols
    X_test = d.test_data.drop(test_drops, axis=1).values
    d.test_data.loc[:,'pred_rel'] = model.predict(X_test)
    ndcg_of_df(d.test_data)
    model_has_importance = False
    try:
        model.feature_importances_
        model_has_importance = True
    except AttributeError:
        None
    imps = None
    if model_has_importance:
        imps = np.argsort(-model.feature_importances_)
        imps = d.pp_data.drop(d.drop_cols, axis=1).columns[imps]
    return model, imps