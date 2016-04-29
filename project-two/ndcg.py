import numpy as np

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
    data['pred_position'] = np.argsort(-data['pred_rel'])
    data['relevance_score'] = np.maximum(data['click_bool'].values, 
                                     data['booking_bool'].values * 5)
    return ndcg_at_k(data['relevance_score'].\
                     values[data['pred_position'].values])