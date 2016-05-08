import itertools
import numpy as np

from sklearn.ensemble import RandomForestRegressor

def transform_pairwise(X, y):
    """Transforms data into pairs with balanced labels for ranking
    Transforms a n-class ranking problem into a two-class classification
    problem. Subclasses implementing particular strategies for choosing
    pairs should override this method.
    In this method, all pairs are choosen, except for those that have the
    same target value. The output is an array of balanced classes, i.e.
    there are the same number of -1 as +1
    Parameters
    ----------
    X : array, shape (n_samples, n_features)
        The data
    y : array, shape (n_samples,) or (n_samples, 2)
        Target labels. If it's a 2D array, the second column represents
        the grouping of samples, i.e., samples with different groups will
        not be considered.
    Returns
    -------
    X_trans : array, shape (k, n_feaures)
        Data as pairs
    y_trans : array, shape (k,)
        Output class labels, where classes have values {-1, +1}
    """
    X_new = []
    y_new = []
    y = np.asarray(y)
    if y.ndim == 1:
        y = np.c_[y, np.ones(y.shape[0])]
    comb = itertools.combinations(range(X.shape[0]), 2)
    for k, (i, j) in enumerate(comb):
        if y[i, 0] == y[j, 0] or y[i, 1] != y[j, 1]:
            # skip if same target or different group
            continue
        X_new.append(X[i] - X[j])
        y_new.append(np.sign(y[i, 0] - y[j, 0]))
        # output balanced classes
        if y_new[-1] != (-1) ** k:
            y_new[-1] = - y_new[-1]
            X_new[-1] = - X_new[-1]
    print 'Pairwise transformation complete!'
    return np.asarray(X_new), np.asarray(y_new).ravel()

# Get matrices from training data frame, transform to pairwise comparisons,
# then fit a pairwise model based on this.
X, y, qry_ids = get_X_y_qry_ids(d.pp_data, d.drop_cols)
X_t, y_t = transform_pairwise(X, np.vstack((y,qry_ids)).T)
model_rf = RandomForestRegressor(n_estimators=1000)
model_rf.fit(X_t, y_t)
"""
Define a new comparer to be passed to the .sort method, that compares
two items by predicting their order according to the fitted model.
"""
def compare_from_model(a, b, model=model_rf):
    cmp = model.predict((a[1]-b[1]).reshape(1,-1))
    if cmp > 0.:
        return -1
    elif cmp < 0.:
        return 1
    else:
        return 0
""" Sorts a table chunk according to the pairwise model"""
def sort_table_chunk(x):
    features = get_X_from_df(x, d.drop_cols)
    sorted_idxs = zip(x.index, features)
    sorted_idxs.sort(compare_from_model)
    pred_order = [i[0] for i in sorted_idxs]
    x['pred_rel'] = pred_order - min(pred_order)
    return x
# Fit the model to a subset of the test data
ss = 10000
t = d.test_data[0:ss]
grouped = t.groupby('srch_id')
print grouped.apply(lambda x: ndcg_of_table_chunk(sort_table_chunk(x))).mean()