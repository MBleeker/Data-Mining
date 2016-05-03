from sklearn.ensemble import RandomForestRegressor

models = {}
models['rf'] = RandomForestRegressor(n_estimators = 500)

s = pd.unique(d.pp_data['srch_id'])
train = s[0:int(len(s)/2)]
test = s[int(len(s)/2):]
train_df = d.pp_data
X_train = train_df.drop(d.drop_cols, axis=1).astype('float32').values
y_train = np.maximum(train_df['click_bool'].values, 
               train_df['booking_bool'].values * 5)
models['rf'].fit(X_train, y_train)
test_df = d.test_data
if 'pred_rel' in test_df.columns:
    test_drops = d.drop_cols + ['pred_rel']
else:
    test_drops = d.drop_cols
X_test = test_df.drop(test_drops, axis=1).values
test_df['pred_rel'] = models['rf'].predict(X_test)
grouped = test_df.groupby('srch_id')
ndcgs = grouped.apply(lambda x: ndcg_of_table_chunk(x))
print ndcgs.mean()
print grouped.apply(lambda x: ndcg_of_table_chunk(x, random_order=True)).mean()
#%%
imps = np.argsort(-models['rf'].feature_importances_)
train_df.drop(d.drop_cols, axis=1).columns[imps]
#%%
d.pp_data[train_df.drop(d.train_drop_cols, axis=1).columns[np.where(np.sum(np.isnan(X_train), axis = 0))]]