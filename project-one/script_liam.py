# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 18:13:39 2016

@author: liam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.decomposition import PCA

# Read in data to big pandas df and setup time variable
os.chdir('/home/liam/cloud/uni/dm/assign1/')
big_df = pd.read_csv('dataset_mood_smartphone.csv')
big_df['time'] = pd.to_datetime(big_df['time'])

# Function to extract a particular feature for a particular individual
def get_feature(feature_name, indiv_id, big_df = big_df):
    feature = big_df[big_df['variable']==feature_name]
    feature = feature[feature['id']==indiv_id][['time','value']]
    return feature.set_index(['time'])
    
'''
In the data, we often have a couple of readings, then a gap of several days,
then the 'core' section of the data. This function returns the start and end
date of this core. This 'core' is defined as the series of values with at most 
one NaN between any two values, surrounded by runs of NaNs which have at least 
two NaNs.
'''
def find_start_and_end_dates(series, num_nans = 4, max_crop = 100):
    start = 0
    # default end is last non-NaN value
    end = np.max(np.where(np.isnan(series)==False))
    found_nan_run, look_for_end = False, False
    for i in range(len(series) - 1):
        if np.all(np.isnan(series[i:(i+num_nans)])):
            if look_for_end:
                end = min(i - 1, end)
                break
            elif i < max_crop:
                start = i + num_nans
                if not found_nan_run:
                    found_nan_run = True
        elif found_nan_run:
            look_for_end = True
    return start, end

def get_mood_and_create_time_index(current_indiv, period):
    ### Get average mood per day for a particular individual ###
    mood_indiv = get_feature('mood', current_indiv)
    # Get average per day
    #mood_indiv = mood_indiv.resample('d', how='mean')
    # Get average per 3 hour period
    mood_indiv = mood_indiv.resample(period, how='mean')
    if 'T' in period:
        start, end = find_start_and_end_dates(mood_indiv['value'], 10)
        mood_indiv = mood_indiv.reindex(
            pd.date_range(start = mood_indiv.index[start].replace(hour = 9), 
                          end = mood_indiv.index[end].replace(hour = 21),
                          freq=period))
        # No mood readings at midnight, 3, or 6am, so remove these periods
        mood_indiv = mood_indiv[mood_indiv.index.hour>8]
    else:
        # Make sure we have every period covered over the 'core' date range
        start, end = find_start_and_end_dates(mood_indiv['value'], 5)
        mood_indiv = mood_indiv.reindex(
            pd.date_range(start = mood_indiv.index[start], 
                          end = mood_indiv.index[end],
                          freq=period))
    mood_indiv.index = mood_indiv.index.rename('time')
    return mood_indiv
def get_features_for_individual(current_indiv, feature_names, period = '180T'):
    # Features that we sum versus average over each time period
    avg_features = ['mood', 'circumplex.valence', 'circumplex.arousal']
    sum_features = [s for s in feature_names if s not in avg_features]
    all_features = get_mood_and_create_time_index(current_indiv, period)
    # Create big data frame containing average daily values for all features
    # for this individual. Do this by joining in each individual feature
    for feature_name in feature_names[1:]:
        feature_indiv = get_feature(feature_name, current_indiv)
        if feature_name in sum_features:
            feature_indiv = feature_indiv.resample(period, how='sum')
        else:
            feature_indiv = feature_indiv.resample(period, how='mean')
        if 'T' in period:
            feature_indiv = feature_indiv[feature_indiv.index.hour>8]
        # Merge in current feature to big features list, on matching index (time)
        # Left join so that we keep all mood readings and fill missing with NaN
        all_features = pd.merge(all_features, feature_indiv, 
                                  left_index = True, right_index = True, 
                                  how = 'left')
    all_features.columns = feature_names
    # Sum features: NaN -> 0, Avg features: NaN -> interpolate
    all_features[sum_features] = all_features[sum_features].fillna(0)
    all_features[avg_features] = all_features[avg_features].interpolate()
    return all_features
# Make list of feature names, ensure mood is the first feature in the list
feature_names = ['mood']+[v for v in big_df['variable'].unique() if v !='mood']
# Make a big dictionary, each entry is df with all features for an individual
features_all_indivs = {}
indiv_ids = big_df['id'].unique()
period = '180T'
for current_indiv in indiv_ids:
    features_all_indivs[current_indiv] = \
        get_features_for_individual(current_indiv, feature_names, period)
        
# Fit a PCA to all the 'appCat' data (for all individuals)
pca_vars = [f for f in feature_names if 'appCat' in f]
start = True
for current_indiv in indiv_ids:
    if start:
        bigdf = features_all_indivs[current_indiv][pca_vars]
        start = False
    else:
        bigdf = pd.concat([bigdf, 
                           features_all_indivs[current_indiv][pca_vars]])
pca = PCA()
pca = pca.fit(bigdf)
num_pca_components = 5

# Define some models
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor,\
                             ExtraTreesRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor
#import xgboost as xgb
from sklearn.svm import SVR
models = {}
models['sgdlm'] = SGDRegressor(l1_ratio = .9)
models['lm'] = LinearRegression()
models['rf'] = RandomForestRegressor(n_estimators = 1000)
models['gbr'] = GradientBoostingRegressor(n_estimators=10000,
                                          learning_rate=0.01, 
                                          max_depth=1, random_state=0, 
                                          loss='ls')
models['svr'] = SVR()
models['extra_trees'] = ExtraTreesRegressor(n_estimators = 1000)

days_prior = 7
train_subset_propn = .7
def get_benchmark_predictions(y, train_subset):
    y_bench = {}
    y_bench['mean'] = np.repeat(y[train_subset].mean(), len(y))
    y_bench['previous_day'] = \
        np.concatenate((y[train_subset].mean().reshape([1,]), y[0:-1]))
    return y_bench

'''
This function gets us a big X matrix.
Each row contains these values for each day:
Days prior = 1,2,..,N
Variable = mood, etc
Hour = 9, 12, 15, 18, 21
So that is, each row contains N * NumberOfVariables * 5 values
We can use these to predict the next day's average mood, which is returned as y
'''
def get_Xy(current_indiv, days_prior, features_all_indivs=features_all_indivs,
           period=period):  
    df = features_all_indivs[current_indiv]
    if num_pca_components:
        pcomps = pca.transform(df[pca_vars])
        for i in range(num_pca_components):
            df['pca_var_' + str(i)] = pcomps[:,i]
        df = df.drop(pca_vars, axis = 1)
    if 'T' in period:
        feature_names = [i + '_h' + str(h) + '_lag' + str(j) 
             for j in range(1, days_prior + 1)
             for i in df.columns
             for h in pd.unique(features_all_indivs[current_indiv].index.hour)]
        df['hour']=df.index.hour
        df['date']=df.index.date
        df = df.pivot(index='date', columns = 'hour')
    else:
        feature_names = [i + '_lag' + str(j) for j in range(1,days_prior+1)
                         for i in df.columns]
    df = df.dropna(axis = 'index')
    #print 'Lost ' + str(df_size - len(df)) + ' row/s containing NaNs'
    X = df.values[(days_prior-1):(len(df)-1),:]
    for i in range(2,days_prior+1):
        X = np.hstack((X, df.values[(days_prior-i):(len(df)-i),:]))
    # Mean mood of each day
    if 'T' in period:
        y = df['mood'].mean(axis=1).values[days_prior:]
    else:
        y = df['mood'].values[days_prior:]
    return X, y, feature_names
# Get RMSE for vector of predictions and vector of targets
def rmse(y, pred):
    return np.sqrt(np.mean(np.power(y - pred, 2)))
scores = {}
predictions = {}
test_subset = {}
for current_indiv in indiv_ids:
    print 'predicting with individual model for ' + current_indiv
    X, y, X_colnames = get_Xy(current_indiv, days_prior)
    train_size = np.int(np.round(train_subset_propn * len(y), 0))
    train_subset = np.arange(train_size)
    test_subset[current_indiv] = np.arange(train_size,len(y))
    tst_idxs = test_subset[current_indiv]
    # Export X and y data for use in R -> arima
    if days_prior == 1 and False:
        np.savetxt('Xys/' + current_indiv + '_X.csv', X, delimiter = ',', 
                   header = ','.join(X_colnames))
        np.savetxt('Xys/' + current_indiv + '_y.csv', y, delimiter = ',',
                   header = 'mood')
        np.savetxt('Xys/' + current_indiv + '_testSubset.csv', 
                   test_subset[current_indiv], delimiter = ',')
    # Benchmark models of y
    y_bench = get_benchmark_predictions(y, train_subset)
    scores_indiv = {}
    predictions[current_indiv] = {}
    for bench in y_bench:
        scores_indiv[bench] = rmse(y[tst_idxs], y_bench[bench][tst_idxs])
    for model in models:
        models[model] = models[model].fit(X[train_subset,:], y[train_subset])
        predictions[current_indiv][model] = models[model].predict(X[tst_idxs,:])
        scores_indiv[model] = rmse(y[tst_idxs], 
                                   predictions[current_indiv][model])
    scores_indiv['avg_gbr_extress'] = \
        rmse(y[tst_idxs],
             (predictions[current_indiv]['extra_trees'] + \
              predictions[current_indiv]['gbr'] ) / 2 )
    scores_indiv['avg_rf_extress'] = \
        rmse(y[tst_idxs],
             (predictions[current_indiv]['extra_trees'] + \
              predictions[current_indiv]['rf'] ) / 2 )
    scores_indiv['avg_3tress'] = \
        rmse(y[tst_idxs],
             (predictions[current_indiv]['extra_trees'] + \
              predictions[current_indiv]['rf'] + \
              predictions[current_indiv]['gbr'] ) / 3 )
    scores[current_indiv] = scores_indiv
    
# Get all individuals data from the training set into one big X, y pair
start = True
for current_indiv in indiv_ids:
    X, y, _ = get_Xy(current_indiv, days_prior)
    train_size = np.int(np.round(train_subset_propn * len(y), 0))
    train_subset = np.arange(train_size)
    if start:
        start = False
        bigX, bigy = X, y
        big_train_subset = train_subset
    else:
        big_train_subset = np.concatenate((big_train_subset, 
                                           train_subset + len(bigy)))
        bigX = np.vstack((bigX, X))
        bigy = np.concatenate((bigy, y))
                                
# Fit each model to these bigXy's
big_models = models
for model in big_models:
    big_models[model] = big_models[model].fit(bigX[big_train_subset,:],
                                              bigy[big_train_subset])
y_bench_big = get_benchmark_predictions(bigy, big_train_subset)
# Now we have fitted the big models, predict each individuals mood for the test
# data using these models
scores_big = {}
predictions_big = {}
for current_indiv in indiv_ids:
    print 'predicting with big model for ' + current_indiv
    X, y, _ = get_Xy(current_indiv, days_prior)
    scores_indiv = {}
    predictions_big[current_indiv] = {}
    tst_idxs = test_subset[current_indiv]
    for bench in y_bench_big:
        scores_indiv[bench] = rmse(y[tst_idxs], 
                                   y_bench_big[bench][tst_idxs])
    for model in big_models:
        predictions_big[current_indiv][model] = big_models[model].predict(X[tst_idxs,:])
        scores_indiv[model] = rmse(y[tst_idxs], predictions_big[current_indiv][model])
    scores_big[current_indiv] = scores_indiv
    
#%% Variable importance
models_with_importance = []
for model in models.keys():
    try:
        big_models[model].feature_importances_
        models_with_importance.append(model)
    except AttributeError:
        None
imp = np.vstack([[X_colnames[n] for n in 
    np.argsort(-big_models[model].feature_importances_)]
    for model in models_with_importance])
pd.DataFrame(imp.T, columns = models_with_importance)[0:50]
#%%
def load_arima_preds(
            path = '/home/liam/cloud/uni/dm/assign1/arima_preds/arima2_preds_'):
    arima_preds = np.genfromtxt(path + current_indiv + '.csv', 
                                delimiter = ',')[1:,1]
    return(arima_preds[(len(arima_preds) - \
                        len(predictions_big[current_indiv]['extra_trees'])):])
# combined predictions
scores_mixed = {}
for current_indiv in indiv_ids:
    scores_mixed[current_indiv] = {}
    X, y, _ = get_Xy(current_indiv, days_prior)
    tst_idxs = test_subset[current_indiv]
    # svr individual + extra trees big
    scores_mixed[current_indiv]['svrInd_ext'] = rmse(y[tst_idxs],
                            (predictions_big[current_indiv]['extra_trees'] + \
                             predictions[current_indiv]['svr'])/2)
    # arima individual + extra trees big
    predictions[current_indiv]['arima'] = load_arima_preds()
    scores_mixed[current_indiv]['arima_ext'] = rmse(y[tst_idxs],
                            (predictions_big[current_indiv]['extra_trees'] + \
                             predictions[current_indiv]['arima'])/2)
    scores_mixed[current_indiv]['arima'] = rmse(y[tst_idxs],
                                          predictions[current_indiv]['arima'])
    scores_mixed[current_indiv]['avg_gbr_extress'] = \
        rmse(y[tst_idxs],
             (predictions_big[current_indiv]['extra_trees'] + \
              predictions_big[current_indiv]['gbr'] ) / 2 )
    scores_mixed[current_indiv]['avg_extrees_bigsmall'] = \
        rmse(y[tst_idxs],
             (predictions_big[current_indiv]['extra_trees'] + \
              predictions[current_indiv]['extra_trees'] ) / 2 )
    scores_mixed[current_indiv]['avg_rf_extress'] = \
        rmse(y[tst_idxs],
             (predictions_big[current_indiv]['extra_trees'] + \
              predictions_big[current_indiv]['rf'] ) / 2 )
    scores_mixed[current_indiv]['avg_3tress'] = \
        rmse(y[tst_idxs],
             (predictions_big[current_indiv]['extra_trees'] + \
              predictions_big[current_indiv]['rf'] + \
              predictions_big[current_indiv]['gbr'] ) / 3 )
    scores_mixed[current_indiv]['avg_2tress_arima'] = \
        rmse(y[tst_idxs],
             (predictions_big[current_indiv]['extra_trees'] + \
              predictions_big[current_indiv]['rf'] + \
              predictions[current_indiv]['arima']) / 3 )
              
# Plot some results
#%pylab qt
scores_df = pd.DataFrame.from_dict(scores, orient = 'index')
scores_df_big = pd.DataFrame.from_dict(scores_big, orient = 'index')
scores_df_mixed = pd.DataFrame.from_dict(scores_mixed, orient = 'index')
scores_df['kind'] = 'individual'
scores_df_big['kind'] = 'big'
scores_df_mixed['kind'] = 'mixed'
scores_df = pd.concat([scores_df, scores_df_big, scores_df_mixed])
cols = {'individual': 'o', 'big': 'b', 'mixed': 'g'}
grouped = scores_df.groupby('kind').mean()
grouped.transpose().plot(kind = 'bar', ylim = (0.5, 0.6))
print grouped