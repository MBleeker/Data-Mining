# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:25:33 2016

@author: liam
"""
# testing lambda rank
def gen_qrys(n):
    qry_ids = np.array([])
    for qry_id in range(n):
        qry_ids_curr = np.repeat(qry_id, np.random.choice(range(30,50),size=1))
        qry_ids = np.hstack((qry_ids, qry_ids_curr))
    X, y = [], []
    def gen_features_from_rel(rel):
        f1 = np.random.normal(rel, 2) + np.random.uniform(0,1)
        f2 = np.random.normal(-rel + 5, 2) * np.random.uniform(rel - 1, rel + 1)
        f3 = np.random.normal(rel, 2) / np.random.uniform(0,1)
        #f1 = rel + 5
        #f2 = np.random.normal(rel, 0.01)
        #f3 = np.random.normal(rel, 0.1)
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
X, y, qry_ids = gen_qrys(10*f)
X_test, y_test, qry_ids_test = gen_qrys(50*f)
model_rf = RandomForestRegressor(n_estimators=100)
#from sklearn.linear_model import LinearRegression
#model_rf = LinearRegression()
model_rf = RandomForestRegressor(n_estimators = 100)
model_rf.fit(X,y)
preds = model_rf.predict(X_test)
np.mean(np.abs(preds - y_test))
df = pd.DataFrame(np.vstack((y_test, y_test, qry_ids_test, preds)).T)
df.columns = ['click_bool','booking_bool','srch_id','pred_rel']
ndcg_of_df(df)

queries_train = Queries(X, y, qry_ids)
queries_test = Queries(X_test, y_test, qry_ids_test)
#%%
lr = 0.001
total_epochs = 30
num_features = queries_train.get_feature_vectors()[0].shape[1]
model = LambdaRank(num_features, 'LambdaRank', lr, train_queries=queries_train)
model.train_with_queries(num_epochs=total_epochs)

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
df = pd.DataFrame(np.vstack((y_test, y_test, qry_ids_test, -preds)).T)
df.columns = ['click_bool','booking_bool','srch_id','pred_rel']
ndcg_of_df(df)
df = pd.DataFrame(np.vstack((y_test, y_test, qry_ids_test, preds)).T)
df.columns = ['click_bool','booking_bool','srch_id','pred_rel']
ndcg_of_df(df)
#%%print grouped.apply(lambda x: ndcg_of_table_chunk(x, random_order=True)).mean()