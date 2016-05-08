# -*- coding: utf-8 -*-
"""
READ IN A NEURAL NETWORK AND THEN OPTIMISE FOR NDCG VIA PARTICLE SWARM

Created on Wed Apr 20 21:54:35 2016

@author: liam
"""

from pyswarm import pso
import pandas as pd
import numpy as np

np.random.seed(0)
# Create small subset for training efficiency
def get_random_df_subset(df, p):
    subset = pd.unique(d.test_data['srch_id'])
    subset = np.random.choice(subset, size = int(len(subset) * p), 
                              replace = False)
    idxs = d.test_data['srch_id'].isin(subset)
    return df[idxs]

def get_X_test(df):
    if 'pred_rel' in test_df_small.columns:
        X_test = df.drop(d.drop_cols + ['pred_rel'], axis=1).values
    else:
        X_test = df.drop(d.drop_cols, axis=1).values
    return X_test

subsample_propn = 0.3
test_df_small = get_random_df_subset(d.test_data, subsample_propn)
X_test = get_X_test(test_df_small)
# Initialise a NN model
#nn_pso_model = NNModel(batchsize=768, num_epochs=1, width=[30,10], 
#                       drop_input=0, learnrate=0.01, drop_hidden=0,
#                       normalise=True)
nn_pso_model = LambdaRankPred(X_test.shape[1], width=[30,10])

def update_nn_params(old_params, updates):
    output_params = [np.copy(a) for a in old_params]
    i = 0
    for p in range(len(output_params)):
        if type(output_params[p]) == np.ndarray:
            for pp in range(len(output_params[p])):
                if type(output_params[p][pp]) == np.ndarray:
                    for ppp in range(len(output_params[p][pp])):
                        output_params[p][pp][ppp] += updates[i]
                        i += 1
                else:
                    output_params[p][pp] += updates[i]
                    i += 1
        else:
            output_params[p] += updates[i]
            i += 1
    return output_params

def get_lb_and_ub(old_params):
    i = 0
    for p in new_params:
        if type(p) == np.ndarray:
            for pp in p:
                if type(pp) == np.ndarray:
                    for ppp in pp:
                        i += 1
                else:
                    i += 1
        else:
            i += 1
        lo = np.min([np.min(a) for a in new_params]) * 1.
        hi = np.max([np.max(a) for a in new_params]) * 1.
        lo = -0.001
        hi = 0.001
        lb, ub = np.ones(i) * lo, np.ones(i) * hi
    return lb, ub
#%%
X_test_big = d.get_Xyq('test', 'X')
nn_params_loaded = cPickle.load(open('nn_30_10_params.dump', 'rb'))
new_params = [np.float32(a) for a in nn_params_loaded[0]]
lasagne.layers.set_all_param_values(nn_pso_model.output_layer, new_params)
nn_pso_model.set_norm_constants(*nn_params_loaded[1])
#%%
lb, ub = get_lb_and_ub(new_params)
print 'starting -------------------------'

def nn_err(updates, orig_params, X_test, test_df_small, nn_pso_model):
    params = update_nn_params(orig_params, updates)
    lasagne.layers.set_all_param_values(nn_pso_model.output_layer, params)
    preds = nn_pso_model.predict(X_test)
    test_df_small['pred_rel'] = preds
    result = ndcg_of_df(test_df_small, plus_random=False)
    return 1. - result

def get_result_on_all_data(old_params, xopt):
    lasagne.layers.set_all_param_values(nn_pso_model.output_layer,
                                        update_nn_params(old_params, xopt))
    preds = nn_pso_model.predict(X_test_big)
    lasagne.layers.set_all_param_values(nn_pso_model.output_layer, old_params)
    d.test_data['pred_rel'] = preds
    return ndcg_of_df(d.test_data, plus_random=False)
    
all_data_result = get_result_on_all_data(new_params, np.zeros(len(lb)))
all_data_result_orig = all_data_result
print 'starting point: ', all_data_result_orig
#%%
for i in range(100):
    now = time.time()
    xopt, fopt = pso(nn_err, lb, ub, 
                     kwargs = {'orig_params' : new_params,
                               'X_test' : X_test,
                               'test_df_small' : test_df_small,
                               'nn_pso_model' : nn_pso_model}, 
                     swarmsize = 6, maxiter = 10)
    print 'new result on all data:'
    all_data_result_new = get_result_on_all_data(new_params, xopt)
    print all_data_result_new
    if all_data_result_new > all_data_result:
        new_params = update_nn_params(new_params, xopt)
        print all_data_result
        print '------------------ improvement this iter, old ^^^:'
        all_data_result = all_data_result_new
    else:
        print '---------------------------------- no improvement!'
    test_df_small = get_random_df_subset(d.test_data, subsample_propn)
    X_test = get_X_test(test_df_small)
    print now - time.time()
    cPickle.dump((new_params, nn_params_loaded[1]),
                  open('best_pso_params.dump', 'wb'))
print 'total improvement: old to new:'
print all_data_result_orig
print all_data_result
#%%
#==============================================================================
# nn_params_loaded = cPickle.load(open('best_pso_params.dump', 'rb'))
# lasagne.layers.set_all_param_values(nn_pso_model.network, nn_params_loaded[0])
# lasagne.layers.set_all_param_values(nn_pso_model.network, new_params)
# preds = nn_pso_model.predict(X_test_big)
# d.test_data['pred_rel'] = preds
# ndcg_of_df(d.test_data, plus_random=False)
# #cPickle.dump((new_params, nn_params_loaded[1]), open('best_pso_params.dump', 'wb'))
#==============================================================================
