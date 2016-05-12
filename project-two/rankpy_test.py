# -*- coding: utf-8 -*-
"""
Created on Tue May 10 14:59:17 2016

@author: liam
"""

from rankpy.models import LambdaMART, LambdaRandomForest
from rankpy import queries

d.drop_cols += [u'srch_children_count',
       u'srch_saturday_night_bool', u'hour_of_day_dum20', u'hour_of_day_dum13',
       u'srch_room_count', u'day_of_week_dum0', u'hour_of_day_dum21',
       u'day_of_week_dum1', u'day_of_week_dum5', u'srch_adults_count',
       u'hour_of_day_dum19', u'day_of_week_dum4', u'day_of_week_dum2',
       u'hour_of_day_dum11', u'hour_of_day_dum1', u'hour_of_day_dum17',
       u'hour_of_day_dum8', u'hour_of_day_dum9', u'hour_of_day_dum16',
       u'hour_of_day_dum15', u'hour_of_day_dum18', u'hour_of_day_dum14',
       u'hour_of_day_dum4', u'day_of_week_dum3', u'hour_of_day_dum12',
       u'hour_of_day_dum10', u'hour_of_day_dum7', u'hour_of_day_dum6',
       u'hour_of_day_dum5', u'hour_of_day_dum3', u'hour_of_day_dum0',
       u'hour_of_day_dum22', u'hour_of_day_dum2',
       u'prop_starrating_mean_by_prop_id_x',
        u'prop_starrating_mean_by_prop_id_y',
        u'prop_starrating_median_by_prop_id_y',
        u'prop_starrating_median_by_prop_id_x',
        u'prop_starrating_std_by_prop_id_y',
        u'prop_starrating_std_by_prop_id_x']
d.drop_cols = list(set(d.drop_cols))

X,y,q = d.get_Xyq('train')
q_indptr = q[0:-1] - q[1:]
q_indptr = np.array([0] + [int(i + 1) for i in \
                            np.where(q_indptr!=0)[0]] + [X.shape[0]])
train_queries = queries.Queries(X, y, q_indptr)
model = LambdaMART(metric='nDCG@38', max_leaf_nodes=7, shrinkage=0.1,
                   estopping=50, n_jobs=-1, min_samples_leaf=50,
                   random_state=42)
#model = LambdaRandomForest(metric='nDCG@38', n_estimators=1000)
model.fit(train_queries)
X,y,q = d.get_Xyq('test')
q_indptr = q[0:-1] - q[1:]
q_indptr = np.array([0] + [int(i + 1) for i in \
                            np.where(q_indptr!=0)[0]] + [X.shape[0]])
test_queries = queries.Queries(X, y, q_indptr, has_sorted_relevances=True)
preds = model.predict(test_queries)
d.test_data['pred_rel'] = preds
result = ndcg_of_df(d.test_data, plus_random=False)
print result
imps = np.argsort(-model.feature_importances())
imps = d.pp_data.drop(d.drop_cols, axis=1).columns[imps]
print imps