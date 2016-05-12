from query import *
from sklearn.ensemble import RandomForestRegressor
from lambda_rank import *
from theano_NN_model import *

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
       u'hour_of_day_dum22', u'hour_of_day_dum2']
d.drop_cols=list(set(d.drop_cols))
# Fit an RF model and predict test set
# Got 0.4443 on the complete data set
model_rf = RandomForestRegressor(n_estimators=100)
model_rf.fit(*d.get_Xyq('train','Xy'))
preds = model_rf.predict(d.get_Xyq('test','X'))
d.test_data['pred_rel'] = preds
result = ndcg_of_df(d.test_data, plus_random=True)
print 'RF model NDCG: ', result
imps = np.argsort(-model_rf.feature_importances_)
imps = d.pp_data.drop(d.drop_cols, axis=1).columns[imps]
print imps
#%%
# Fit a LambdaRank model
# Got .47 with 10 epochs and lr = 0.0001
lr = 0.0001
total_epochs = 2
num_features = d.get_Xyq('test','X').shape[1]
for w in ([[30,10],[30,20,10],[50,30,10],[50,10],[50,30,5]]):
    print w
    model_nn = LambdaRank(num_features, 'LambdaRank', lr, 
                       train_queries=Queries(*d.get_Xyq('train')),
                       width=w, normalise=False, drop_hidden=0, drop_input=0)
    for i in range(15):
        model_nn.train_with_queries(num_epochs=total_epochs)
        preds = model_nn.score_matrix(d.get_Xyq('test', 'X'))
        d.test_data['pred_rel'] = preds
        result = ndcg_of_df(d.test_data, plus_random=False)
        print 'LambdaRank model NDCG: ', result