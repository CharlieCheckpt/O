import os
import time
import numpy as np
import pickle as pkl

#from sys import stdout, stdin, argv

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid,StratifiedKFold

import xgboost as xgb

DATA_PATH='data'



class XGBoost:

    def __init__(self, Xtr, Ytr, Xval, Yval, base_params = {}, hist_file='xgboost.log'):
        self.dtr  = xgb.DMatrix(Xtr, label=Ytr)
        self.dval = xgb.DMatrix(Xval, label=Yval)
        self.base_params = base_params
        self.hist_file = hist_file

    def test_params(self, grid, nrounds, early_stop):

        best_params = {}

        if not os.path.exists(self.hist_file):
            params_history = {}
        else:
            with open(self.hist_file, 'rb') as fp:
                params_history = pkl.load(fp)

        date_str = time.strftime("%m/%d/%Y", time.gmtime())
        if date_str not in params_history:
            params_history[date_str] = []
        params_set = params_history[date_str]

        best_booster_acc = 0

        for update_params in ParameterGrid(grid):

            params = dict(self.base_params)
            params.update(update_params)

            booster_eval={}
            booster = xgb.train(params, self.dtr, nrounds,
                                [(self.dtr, 'train'), (self.dval, 'valid')],
                                early_stopping_rounds=early_stop,
                                evals_result=booster_eval)

            booster_acc = np.max(booster_eval['valid']['auc'])

            if (booster_acc > best_booster_acc):
                best_booster_acc = booster_acc
                best_booster = booster
                best_params = params

            params_set.append((params, booster_acc))

            with open(self.hist_file, 'wb') as fp:
                pkl.dump(params_history, fp)

        return best_params, best_booster_acc, best_booster




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
    print(f"Data {data.shape} loaded in {round(end-start,3)} seconds")
else:
    # load all
    print("loading all ...")
    start = time.time()
    data = rcsv.read(DATA_PATH + '/' + 'Xtrain_challenge_owkin.csv')
    data = data[1:, 1:]
    lbls = np.load(DATA_PATH + '/' + 'Ytrain_challenge_owkin_half.npy')
    end = time.time()
    print(f"Data {data.shape} loaded in {round(end-start,3)} seconds")

nfolds = 5
compt = 0
store_val_acc = {}

skf = StratifiedKFold(nfolds,random_state=777)
skf.get_n_splits(data,lbls)

print(f" Cross validation on : {nfolds} using StratifiedKfolds" )

for train_index, val_index in skf.split(data, lbls):
    print(f'\n \n *** starting cross validation n°{compt+1}... *** \n \n')

    print("TRAIN:", train_index, "VAL:", val_index)
    Xtr, Xval = data[train_index], data[val_index]
    Ytr, Yval = lbls[train_index], lbls[val_index]

    print(f' \n Xtr : {Xtr.shape}')
    print(f' Xval : {Xval.shape}')
    print(f' Ytr : {len(Ytr)}')
    print(f' Yval : {len(Yval)}')

    xgb_model = XGBoost(Xtr, Ytr, Xval, Yval)

    best_params,best_booster_acc,best_booster = xgb_model.test_params({
        'max_depth'       : [5], # tried [5, 6, 7, 8]
        'objective'       : ['binary:logistic'],
        'eval_metric'     : ['auc'],
    }, 5000, 300)

    store_val_acc[compt] = best_booster_acc
    compt +=1

print(" \n \n  **** Mean val accuracy *** \n " + str(np.mean([store_val_acc[i] for i in range(len(store_val_acc))])))

print(f"\n *** Val accuracies : {store_val_acc}")


import pickle
with open('xgb_cv_acc_18000_5000t_5d.pkl', 'wb') as f:
    pickle.dump(store_val_acc, f)
    print("xgb log saved")

#xgb_model.base_params = best_params
#best_params = xgb_model.test_params({
#    'subsample' : [1], #[0.5, 0.8, 1]
#    'colsample_bytree' : [1] # [0.5, 0.8, 1]
#}, , 2)


#best_params = {
#    'seed'            : [777],
#    'max_depth'       : [6], # tried [5, 6, 7, 8]
#    'objective'       : ['binary:logistic'],
#    'eval_metric'     : ['auc'],
#    'min_child_weight': [1], # [0.5, 1, 5, 10]
#    'booster' : ['gbtree'],
#    'nthread' : [8],
#    'early_stopping_rounds' : [300]
#}
