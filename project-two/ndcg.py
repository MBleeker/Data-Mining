import numpy as np
import time

def dcg_at_k(relevances, k = 38):
    if np.sum(relevances) == 0:
        return 0.
    else:
        k = min(k, len(relevances))
        dcg = 0
        for r in range(k):
            dcg += (2 ** relevances[r] - 1) / np.log2(2 + r)
        return dcg
        
def ndcg_at_k(relevances, k = 38):
    if np.sum(relevances) == 0:
        return 0.
    else:
        dcg = dcg_at_k(relevances, k)
        idcg = dcg_at_k(sorted(relevances, reverse=True), k)
        return dcg / idcg

def ndcg_of_table_chunk(data, k = 38, random_order = False):
    if random_order:    
        data['pred_rel'] = np.random.rand(len(data))
    order = np.argsort(-data['pred_rel'])
    rels = np.maximum(data['click_bool'].values, data['booking_bool'].values * 5)
    return ndcg_at_k(rels[order])

def ndcg_of_df(df, plus_random = True):
    now = time.time()
    grouped = df.groupby('srch_id')
    ndcg_mean = grouped.apply(lambda x: ndcg_of_table_chunk(x)).mean()
    #print 'took ' + str(np.round((time.time()-now)/60,2)) + ' minutes ' + \
    #    'to make predictions from model'
    #print 'mean NDCG over predicted test set:'
    #print ndcg_mean
    if plus_random:
        print 'mean NDCG using random order (for comparison):'
        print grouped.apply(lambda x: ndcg_of_table_chunk(x, random_order=True)).mean()
    return ndcg_mean