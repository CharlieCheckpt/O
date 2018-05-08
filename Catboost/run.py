import os
import time
import numpy as np
import pickle as pkl

from sys import stdout, stdin, argv

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold

import xgboost as xgb
from catbooster import *

DATA_PATH='data'

# =============================================================================
# Loading Data
# =============================================================================
load_18k = True

if load_18k:
    print("\n loading half ...")
    start = time.time()
    data = np.load(DATA_PATH + '/' + 'Xtrain_challenge_owkin_half_18k.npy')
    lbls = np.load(DATA_PATH + '/' + 'Ytrain_challenge_owkin_half.npy')
    Xte = np.load(DATA_PATH + '/' + 'Xtest_challenge_owkin_half_18k.npy')
    end = time.time()
    print(f"data {data.shape} loaded in {round(end-start,3)} seconds")
else:
    print("loading subset ...")
    start = time.time()
    data = np.load(DATA_PATH + '/' + 'Xtrain_challenge_owkin_half_subset.npy')
    lbls = np.load(DATA_PATH + '/' + 'Ytrain_challenge_owkin_half.npy')
    all_features = np.loadtxt('features_importance_it3.txt')
    end = time.time()
    ids = np.nonzero(all_features)[0]
    data = data[:, ids]
    Xte = Xte[:, ids]
    print(f"data {data.shape} loaded in {round(end-start,3)} seconds")


print(f' \n Xtr : {data.shape}')
print(f' Ytr : {len(lbls)}')
print(f' \n Xte : {Xte.shape}')


model = CatBoostClassifier(
                eval_metric='AUC',
                iterations=8000,
                random_seed=42,
                logging_level='Silent')


print("fitting...")
model.fit( data, lbls,cat_features= None,logging_level='Verbose' )
print("predicting...")
preds = model.predict_proba(Xte)
np.savetxt('pred_cb18k_8000t.csv', preds, fmt='%.5f', delimiter=',')
print("predictions saved")


#nfolds = 5

#print(f"train percentage : {len(data)/nfolds}")
#compt = 0
#store_val_acc = {}

#skf = StratifiedKFold(nfolds,random_state=777)
#skf.get_n_splits(data,lbls)


#print(f" Cross validation with {nfolds} folds using StratifiedKfolds" )

#for train_index, val_index in skf.split(data, lbls):
#    print(f'\n \n *** starting cross validation n°{compt+1}... *** \n \n')
#
#    print("TRAIN:", train_index, "VAL:", val_index)
#    Xtr, Xval = data[train_index], data[val_index]
#    Ytr, Yval = lbls[train_index], lbls[val_index]


#    Xtr = data[1:10]
#    Xval = data[10:20]
#    Ytr = lbls[1:10]
#    Yval = lbls[10:20]
#    print(f' \n Xtr : {Xtr.shape}')
#    print(f' Xval : {Xval.shape}')
#    print(f' Ytr : {len(Ytr)}')
#    print(f' Yval : {len(Yval)}')


#    model = CatBoost(Xtr, Ytr, Xval, Yval)

#    best_params, best_booster_auc = model.test_params({},3000,200)

#    del Xtr,Xval,Ytr,Yval
#    store_val_acc[compt] = best_booster_auc
#    compt +=1

#print(" \n \n  **** Mean val accuracy *** \n " + str(np.mean([store_val_acc[i] for i in range(len(store_val_acc))])))

#print(f"\n *** Val accuracies : {store_val_acc}")


import pickle
with open('catb_cv_acc_sub_cat_3000t_def.pkl', 'wb') as f:
    pickle.dump(store_val_acc, f)
    print("catboost log saved")
