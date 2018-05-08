import numpy as np
import pickle as pkl
import time
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

import lightgbm as lgb

DATA_PATH='data'

# =============================================================================
# Loading Data
# =============================================================================
load_18k = True

if load_18k:
    #load half
    start = time.time()
    print("\n loading half ...")
    data = np.load(DATA_PATH + '/' + 'Xtrain_challenge_owkin_half_18k.npy')
    lbls = np.load(DATA_PATH + '/' + 'Ytrain_challenge_owkin_half.npy')
    end = time.time()
    print(f"data loaded in {round(end-start,3)} seconds")
else:
    # load all
    import rcsv
    start = time.time()
    print("loading all ...")
    data = rcsv.read(DATA_PATH + '/' + 'Xtrain_challenge_owkin.csv')
    data = data[1:, 1:]
    lbls = np.load(DATA_PATH + '/' + 'Ytrain_challenge_owkin_half.npy')
    end = time.time()
    print(f"data loaded in {round(end-start,3)} seconds")

# general parameters
ntrees = 5000
print(f"General parameters :\n ntrees : {ntrees}")

results = []

# =============================================================================
# PARAMETERS TO TEST WITH HYPEROPT
# =============================================================================

from hyperopt import hp, tpe
from hyperopt.fmin import fmin


def objective(params,X = data, Y = lbls, results = results,ntrees = ntrees):
    # dtr = lgb.Dataset(X, label = Y)

    params = {
        'num_leaves': int(params['num_leaves']),
        'min_data_in_leaf': int(params['min_data_in_leaf']),
        'objective'   : ['binary'],
        'metric'     : ['auc'],
    }

    clf = lgb.LGBMClassifier(
        n_estimators=ntrees,
        **params
    )

    score = cross_val_score(clf, data, lbls, scoring='roc_auc', cv=StratifiedKFold()).mean()
    print("\n\n**** AUC {:.3f} params {} ****\n\n".format(score, params))
    # save score
    results.append({})
    results[len(results)-1]['params'] = params
    results[len(results)-1]['auc'] = score
    return score

space = {
        'num_leaves' : hp.quniform('num_leaves', 10, 90, 2),
        'min_data_in_leaf' : hp.quniform('min_data_in_leaf', 1, 50, 10),
        }

start = time.time()
best = fmin(fn=objective,
            space=space,
            algo=tpe.suggest,
            max_evals=10)
end = time.time()

print("\n\nHyperopt estimated optimum {}".format(best))
print(f"Cross Validation done in {round(end-start, 3)} seconds")
# =============================================================================
# SAVE RESULTS
# =============================================================================

from time import gmtime, strftime
date = strftime("%Y-%m-%d %H:%M:%S", gmtime()) # date for file saving
date = date.replace(" ", "_")
file_name = 'lgbm_cv_'+str(data.shape[1])+'k_'+str(ntrees)+'t_hypop'+date+'.pkl'
with open(file_name, 'wb') as f:
    pkl.dump(results, f)
    print(f"\nlgbm log saved as {file_name}")
