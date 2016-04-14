# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:00:55 2016

@author: liam
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from statsmodels.tsa.arima_model import ARIMA

# Read in data to big pandas df and setup time variable
os.chdir('/Users/Maurits/Documents/GitHub/School Projects/Data-Mining/project-one/data')
big_df = pd.read_csv('dataset_mood_smartphone.csv')
big_df['time'] = pd.to_datetime(big_df['time'])
#%%
# Get RMSE for vector of predictions and vector of targets
def rmse(y, pred):
    return np.sqrt(np.mean(np.power(y - pred, 2)))
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
def get_mood_and_create_time_index(current_indiv):
    ### Get average mood per day for a particular individual ###
    mood_indiv = get_feature('mood', current_indiv)
    # Get average per day
    #mood_indiv = mood_indiv.resample('d', how='mean')
    # Get average per 3 hour period
    mood_indiv = mood_indiv.resample('180T', how='mean')
    # Make sure we have every period covered over the 'core' date range
    start, end = find_start_and_end_dates(mood_indiv['value'], 10)
    mood_indiv = mood_indiv.reindex(
        pd.date_range(start = mood_indiv.index[start].replace(hour = 9), 
                      end = mood_indiv.index[end].replace(hour = 21),
                      freq='180T'))
    # No mood readings at midnight, 3, or 6am, so remove these periods
    mood_indiv = mood_indiv[mood_indiv.index.hour>8]
    mood_indiv.index = mood_indiv.index.rename('time')
    return mood_indiv
def get_features_for_individual(current_indiv, feature_names):
    # Features that we sum versus average over each time period
    avg_features = ['mood', 'circumplex.valence', 'circumplex.arousal']
    sum_features = [s for s in feature_names if s not in avg_features]
    all_features = get_mood_and_create_time_index(current_indiv)
    # Create big data frame containing average daily values for all features
    # for this individual. Do this by joining in each individual feature
    for feature_name in feature_names[1:len(feature_names)]:
        feature_indiv = get_feature(feature_name, current_indiv)
        if feature_name in sum_features:
            feature_indiv = feature_indiv.resample('180T', how='sum')
        else:
            feature_indiv = feature_indiv.resample('180T', how='mean')
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
for current_indiv in indiv_ids:
    features_all_indivs[current_indiv] = \
        get_features_for_individual(current_indiv, feature_names)
#%%
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
# xgb sucks... models['xgb'] = xgb.XGBRegressor(n_estimators=100)
#%%
'''
This function gets us a big X matrix.
Each row contains these values for each day:
Days prior = 1,2,..,N
Variable = mood, etc
Hour = 9, 12, 15, 18, 21
So that is, each row contains N * NumberOfVariables * 5 values
We can use these to predict the next day's average mood, which is returned as y
'''
days_prior = 3
train_subset_propn = .7
def get_arima_predictions(y, train_subset, order = [1,0,0], X = None):
    if X == None:
        arima = ARIMA(y[train_subset], order = order).fit()
        predictions = arima.predict()
    else:
        arima = ARIMA(y[train_subset], order = order, 
                      exog = X[train_subset,:]).fit()
        predictions = arima.predict(exog = X[train_subset,:])
    for i in range(max(train_subset)+1,len(y)):
        if X == None:
            arima = ARIMA(y[0:i], order = order).fit()
            predictions = np.append(predictions, 
                                    arima.predict(0, len(y) + i)[-1])
        else:
            arima = ARIMA(y[0:i], order = order, exog = X[0:i,:]).fit()
            predictions = np.append(predictions, 
                                    arima.predict(0, len(y) + i, 
                                                  exog = X[0:i+1,:])[-1])
    return predictions
def get_benchmark_predictions(y, train_subset):
    y_bench = {}
    y_bench['arima'] = get_arima_predictions(y, train_subset)
    y_bench['mean'] = np.repeat(y[train_subset].mean(), len(y))
    y_bench['previous_day'] = \
        np.concatenate((y[train_subset].mean().reshape([1,]), y[0:-1]))
    return y_bench
def get_lagged_feature_matrix_and_target_mood(current_indiv, days_prior,
                                   features_all_indivs = features_all_indivs):
    df = features_all_indivs[current_indiv]
    df['hour']=df.index.hour
    df['date']=df.index.date
    df = df.pivot(index='date', columns = 'hour')
    df = df.dropna(axis = 'index')
    #print 'Lost ' + str(df_size - len(df)) + ' row/s containing NaNs'
    X = df.values[(days_prior-1):(len(df)-1),:]
    for i in range(2,days_prior+1):
        X = np.hstack((X, df.values[(days_prior-i):(len(df)-i),:]))
    # Mean mood of each day
    y = df['mood'].mean(axis=1).values[days_prior:]
    return X, y
scores = {}
for current_indiv in indiv_ids:
    print 'predicting with individual model for ' + current_indiv
    X, y = get_lagged_feature_matrix_and_target_mood(current_indiv, days_prior)    train_size = np.int(train_subset_propn * len(y))
    train_subset = np.arange(train_size)
    test_subset = np.arange(train_size,len(y))
    # Benchmark models of y
    y_bench = get_benchmark_predictions(y, train_subset)
    scores_indiv = {}
    for bench in y_bench:
        scores_indiv[bench] = rmse(y[test_subset], y_bench[bench][test_subset])
    for model in models:
        models[model] = models[model].fit(X[train_subset,:], y[train_subset])
        scores_indiv[model] = rmse(y[test_subset], 
                              models[model].predict(X[test_subset,:]))
    arima_preds = get_arima_predictions(y, train_subset, X = X)
    scores_indiv['arima_exog'] = rmse(arima_preds[test_subset], y[test_subset])
    scores[current_indiv] = scores_indiv
# Get all individuals data from the training set into one big X, y pair
start = True
for current_indiv in indiv_ids:
    X, y = get_lagged_feature_matrix_and_target_mood(current_indiv, days_prior)
    train_size = np.int(train_subset_propn * len(y))
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
big_test_subset = np.array([i for i in range(len(bigy)) \
                            if i not in big_train_subset])
# Fit each model to these bigXy's
big_models = models
for model in big_models:
    big_models[model] = big_models[model].fit(bigX[big_train_subset,:],
                                              bigy[big_train_subset])
y_bench_big = get_benchmark_predictions(bigy, big_train_subset)
# Now we have fitted the big models, predict each individuals mood for the test
# data using these models
scores_big = {}
for current_indiv in indiv_ids:
    print 'predicting with big model for ' + current_indiv
    X, y = get_lagged_feature_matrix_and_target_mood(current_indiv, days_prior)
    train_size = np.int(train_subset_propn * len(y))
    train_subset = np.arange(train_size)
    test_subset = np.arange(train_size,len(y))
    scores_indiv = {}
    for bench in y_bench_big:
        scores_indiv[bench] = rmse(y[test_subset], 
                                   y_bench_big[bench][test_subset])
    for model in big_models:
        scores_indiv[model] = rmse(y[test_subset], 
                                   big_models[model].predict(X[test_subset,:]))
    scores_big[current_indiv] = scores_indiv
# Plot some results
scores_df = pd.DataFrame.from_dict(scores, orient = 'index')
scores_df_big = pd.DataFrame.from_dict(scores_big, orient = 'index')
scores_df['kind'] = 'normal'
scores_df_big['kind'] = 'big'
scores_df = pd.concat([scores_df, scores_df_big])
cols = {'normal': 'r', 'big': 'b'}
grouped = scores_df.groupby('kind').mean()
#%%
grouped.transpose().plot(kind = 'bar', ylim = (0,0.8))
#%%
# Variable importance for rf and gbr in our big models...
def get_variable_names_for_big_x(days_prior):
    current_indiv = indiv_ids[0]
    df = features_all_indivs[current_indiv]
    df['hour']=df.index.hour
    df['date']=df.index.date
    df = df.pivot(index='date', columns = 'hour')
    df = df.dropna(axis = 'index')
    names = ["_".join([str(j) for j in i]) for i in df.columns]
    names = ['lag' + tp + '_' + i for tp in 
        [str(n) for n in range(1,days_prior+1)] for i in names]
    return names
varnames_bigX = get_variable_names_for_big_x(days_prior)
models_with_importance = []
for model in models.keys():
    try:
        big_models[model].feature_importances_
        models_with_importance.append(model)
    except AttributeError:
        None
imp = np.vstack([[varnames_bigX[n] for n in 
    np.argsort(-big_models[model].feature_importances_)] 
    for model in models_with_importance])
pd.DataFrame(imp.T, columns = models_with_importance)[0:50]
#%%
nn = NNModel(num_epochs = 5000, learnrate = .005, batchsize = 10, 
             width = [500, 300, 100, 50], normalise = True)
nn.fit(bigX[big_train_subset,:],bigy[big_train_subset])
nnpreds = nn.predict(bigX[big_test_subset,:])
print rmse(bigy[big_test_subset], nnpreds)
print rmse(bigy[big_test_subset], 
           big_models['extra_trees'].predict(bigX[big_test_subset,:]))
#%%
# plot histogram of feature
def plot_histogram(individual, feature):
    print 'Plotting histogram of ' + feature
    feature_indiv = get_feature(feature, individual)
    if feature == 'mood':
        ax = (feature_indiv['value']).hist(range = [0,10])
    else:
        ax = (feature_indiv['value']).hist()
    fig = ax.get_figure()
    plt.show(block=False)
    plt.close(fig)
# plot time series of feature
def plot_series(individual, feature):
    print 'Plotting ' + feature
    feature_indiv = get_feature(feature, individual)
    ax = feature_indiv.plot(y='value',use_index=True)
    if feature == 'mood':
        ax.set_ylim((0,10))
    ax.set_xlim((min(feature_indiv.index),max(feature_indiv.index)))
    fig = ax.get_figure()
    plt.show(block=False)
    plt.close(fig)
for individual in indiv_ids:
    print individual
    #plot_histogram(individual, 'mood')
    #plot_series(individual, 'mood')
#%%
y = get_feature_by_day(feature, current_indiv).values
train_subset = range(30)
arima = ARIMA(y[train_subset], [1,0,0]).fit()
predictions = arima.predict()
for i in range(max(train_subset)+1,len(y)):
    arima = ARIMA(y[0:i], [1,0,0]).fit()
    predictions = np.append(predictions,arima.predict(0, len(y) + i)[-1])
y = get_feature_by_day(feature, current_indiv)
y['preds'] = predictions
y.plot()
rmse(y['preds'].values,y['value'].values)
#%%
arima.predict(start = min(y.index), end = 50)
#%%
from statsmodels.tsa.stattools import acf, pacf
def get_feature_by_day(feature, current_indiv):
    y = get_feature(feature, current_indiv)
    avg_features = ['mood', 'circumplex.valence', 'circumplex.arousal']
    sum_features = [s for s in feature_names if s not in avg_features]
    if feature in avg_features:
        mn = 'mean'
    else:
        mn = 'sum'
    y = y.resample('d', how=mn)
    start, end = find_start_and_end_dates(y['value'], 10)
    y = y.reindex(pd.date_range(start = y.index[start], end = y.index[end]))
    y.index = y.index.rename('time')
    if feature in avg_features:
        y = y.interpolate()
    else:
        y = y.fillna(0)
    return y

def plot_acf_and_pacf(y):
    lag_acf = acf(y, nlags=20)
    lag_pacf = pacf(y, nlags=20, method='ols')
    
    plt.subplot(121) 
    plt.plot(lag_acf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(y)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(y)),linestyle='--',color='gray')
    plt.title('Autocorrelation Function')
    
    #Plot PACF:
    plt.subplot(122)
    plt.plot(lag_pacf)
    plt.axhline(y=0,linestyle='--',color='gray')
    plt.axhline(y=-1.96/np.sqrt(len(y)),linestyle='--',color='gray')
    plt.axhline(y=1.96/np.sqrt(len(y)),linestyle='--',color='gray')
    plt.title('Partial Autocorrelation Function')
    plt.tight_layout()
    plt.show()
    plt.close()
#%%
for current_indiv in indiv_ids:
    print current_indiv
    y = get_feature_by_day('mood', current_indiv)
    plot_acf_and_pacf(y)