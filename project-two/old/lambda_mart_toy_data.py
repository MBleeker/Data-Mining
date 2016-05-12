# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:25:33 2016

@author: liam
"""
from rankpy.models import LambdaMART, LambdaRandomForest
from rankpy import queries
import numpy as np
# testing lambda rank
def gen_qrys(n):
    qry_ids = np.array([])
    for qry_id in range(n):
        qry_ids_curr = np.repeat(qry_id, np.random.choice(range(30,50),size=1))
        qry_ids = np.hstack((qry_ids, qry_ids_curr))
    X, y = [], []
    def gen_features_from_rel(rel):
        #f1 = np.random.normal(rel, 2) + np.random.uniform(0,1)
        #f2 = np.random.normal(-rel + 5, 2) * np.random.uniform(rel - 1, rel + 1)
        #f3 = np.random.normal(rel, 2) / np.random.uniform(0,1)
        f1 = np.random.normal(rel, 0.01)
        f2 = np.random.normal(rel, 0.01)
        f3 = np.random.normal(rel, 0.1)
        return list((f1,f2,f3))
    for q in range(n):
        num = sum(qry_ids==q)
        for i in range(num):
            this_y = np.random.choice([0,0,0,0,0,0,1,1,1,5],size=1)[0]
            y.append(this_y)
            X.append(gen_features_from_rel(this_y))
    y = np.array(y)
    X = np.array(X)
    return X, y, qry_ids
f=10
X, y, q = gen_qrys(10*f)

q_indptr = q[0:-1] - q[1:]
q_indptr = np.array([0] + [int(i + 1) for i in np.where(q_indptr!=0)[0]] + [X.shape[0]])
train_queries = queries.Queries(X,y,q_indptr)
#model = LambdaMART(metric='nDCG@38', max_leaf_nodes=3, shrinkage=0.1,
#                   estopping=50, n_jobs=-1, min_samples_leaf=50,
#                   random_state=1)
model = LambdaRandomForest(metric='nDCG@10')
model.fit(train_queries)
#%%

#%%
X, y, q = gen_qrys(50*f)
#X = np.hstack((X,np.arange(X.shape[0]).reshape((X.shape[0],1))))
q_indptr = q[0:-1] - q[1:]
q_indptr = np.array([0] + [int(i + 1) for i in np.where(q_indptr!=0)[0]] + [X.shape[0]])
test_queries = queries.Queries(X,y,q_indptr,has_sorted_relevances=True)
#%%
order = (test_queries.feature_vectors[:,3]).astype(np.int)
#%%
preds = model.predict(test_queries)
df = pd.DataFrame(np.vstack((y, y, q, preds)).T)
df.columns = ['click_bool','booking_bool','srch_id','pred_rel']
ndcg_of_df(df)