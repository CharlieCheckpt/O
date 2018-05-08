import numpy as np
import pickle as pkl
import time

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

import lightgbm as lgb

DATA_PATH='data'


class Lgbm:

    def __init__(self, Xtr, Ytr, Xval, Yval, base_params = {}, hist_file='lgbm.log'):
        self.dtr  = lgb.Dataset(Xtr, label=Ytr)
        self.dval = lgb.Dataset(Xval, label=Yval)

    def test_params(self, param, nrounds, early_stop, Xval, Yval):

        booster = lgb.train(param, self.dtr, nrounds,
                            valid_sets=[self.dval],
                            early_stopping_rounds=early_stop)

        best_booster_auc = roc_auc_score(Yval, booster.predict(Xval,num_iteration=booster.best_iteration)) # CS

        return booster,best_booster_auc

# =============================================================================
# Loading Data
# =============================================================================
load_18k = True

if load_18k:
#load half
    start = time.time()
    print("\n loading half ...")
    data = np.load(DATA_PATH + '/' + 'Xtrain_challenge_owkin_half.npy')
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
nfolds = 3
ntrees = 5000
nearly_stop = 200
print(f"General parameters :\n nfolds : {nfolds}, ntrees : {ntrees}, nearly_stop : {nearly_stop}")

results = []

skf = StratifiedKFold(nfolds,random_state=777)


print(f" Cross validation with {nfolds} folds using StratifiedKfolds" )

# =============================================================================
# PARAMETERS TO TEST
# =============================================================================

param_grid = {
        'num_leaves' : [31], # 31 seems best
        'objective'   : ['binary'],
        'metric'     : ['auc'],
        'min_data_in_leaf' :[1], # [1,5, 20, 50,100]
        'learning_rate':[0.1], # 0.1 seems best
    # 'boosting' : ['gbdt'], # ['dart', 'gbdt'] gbdt seems best,
        # 'min_sum_hessian_in_leaf' : [1e-4,1e-3,1e-2], #1e-3 is default
        # 'min_gain_to_split' : [0,0.5],
#        'feature_fraction' : [0.8,1],
#        'bagging_fraction' : [0.8,1],
        }

# get number of combinations we are going to test in cv
import itertools as it
allNames = sorted(param_grid)
ncombi = len(list(it.product(*(param_grid[Name] for Name in allNames))))
print(f"*** Let's test {ncombi} combinations ***")

# =============================================================================
# STARTING THE "BIG LOOP"
# =============================================================================
compt = 0
compt_cv = 0
for param in ParameterGrid(param_grid):

    results.append(param.copy())
    results[compt]['auc'] = [] # here we store auc results for each cross validiation

    for train_index, val_index in skf.split(data, lbls):
        print(f'\n \n*** starting cross validation n°{compt_cv+1}/{ncombi*nfolds}... ***')
        print(f'*** starting cross validation n°{compt_cv+1}/{ncombi*nfolds}... ***')
        print(f'*** starting cross validation n°{compt_cv+1}/{ncombi*nfolds}... *** \n \n')

        print("TRAIN:", train_index, "VAL:", val_index)
        Xtr, Xval = data[train_index], data[val_index]
        Ytr, Yval = lbls[train_index], lbls[val_index]

        print(f' \n Xtr : {Xtr.shape}')
        print(f' Xval : {Xval.shape}')
        print(f' Ytr : {len(Ytr)}')
        print(f' Yval : {len(Yval)}')

        lgb_model = Lgbm(Xtr, Ytr, Xval, Yval)

        booster, best_booster_auc = lgb_model.test_params(param, ntrees, nearly_stop, Xval, Yval) # CHANGE IT  !!!!
        compt_cv += 1

        del Xtr,Xval,Ytr,Yval
        del lgb_model

        results[compt]['auc'].append(best_booster_auc)
        print(results[compt]['auc'])


    mean_val_auc = np.mean(results[compt]['auc'])
    print(f" \n \n**** CV n°{compt_cv-1} -- mean val accuracy : {mean_val_auc}")
    print(f"\n *** CV n°{compt_cv-1} -- val accuracies : {results[compt]['auc']}")
    compt = compt + 1


# =============================================================================
# SAVE RESULTS
# =============================================================================

from time import gmtime, strftime
date = strftime("%Y-%m-%d %H:%M:%S", gmtime()) # date for file saving
date = date.replace(" ", "_")
file_name = 'lgbm_cv_'+str(data.shape[1])+'k_'+str(ntrees)+'t_'+str(nfolds)+'f_gs__'+date+'.pkl'
with open(file_name, 'wb') as f:
    pkl.dump(results, f)
    print(f"\nlgbm log saved as {file_name}")
