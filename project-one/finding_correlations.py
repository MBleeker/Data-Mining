# -*- coding: utf-8 -*-
"""
Created on Thu Apr 07 15:04:33 2016

@author: Jaimy
"""

import script

df = features_all_indivs[indiv_ids[0]][feature_names]
df.reindex(pd.date_range(start = min(big_df['time']).replace(hour = 9, minute = 0, second = 0), 
                      end = max(big_df['time']).replace(hour = 21),
                      freq='180T'))
df['counts'] = pd.Series(np.zeros(len(df.index)), index=df.index)
#%%
for user in indiv_ids[1:]:
    temp = features_all_indivs[user][feature_names].dropna()
    temp['counts'] = 1
    df = df.add(temp, fill_value=0)
df = df.div(df['counts'], axis = 'index')
del df['counts']
#%%
lag = np.hstack((df.values[1:len(df),1].reshape((len(df)-1,1)), df.values[0:(len(df)-1),1:]))
lagdf = pd.DataFrame(lag, columns = df.columns)
#%%
%pylab qt
def dataframe_correlation_plot(df):
    size = len(feature_names)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    cax = ax.matshow(corr)
    fig.colorbar(cax, ticks=[-1, 0, 1])
    plt.xticks(range(len(corr.columns)), corr.columns);
    plt.yticks(range(len(corr.columns)), corr.columns);
    
    plt.show()
#%%
ax = df['mood'].plot(y='value',use_index=True)
ax.set_ylim((0,10))
ax.set_xlim((min(df.index),max(df.index)))
fig = ax.get_figure()
plt.show(block=False)
plt.close(fig)