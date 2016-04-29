from sklearn.ensemble import RandomForestRegressor

models = {}
models['rf'] = RandomForestRegressor(n_estimators = 1000)
models['rf'].fit(X, y)
data_test['pred_rel'] = models['rf'].predict(X_test)
grouped = data_test.groupby('srch_id')
ndcgs = grouped.apply(lambda x: ndcg_of_table_chunk(x))
print ndcgs.mean()
print grouped.apply(lambda x: ndcg_of_table_chunk(x, random_order=True)).mean()