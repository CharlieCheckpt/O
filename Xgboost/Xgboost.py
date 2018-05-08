import os
import time
import numpy as np
import pickle as pkl

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid

import xgboost as xgb

DATA_PATH='data'



class XGBoost:

    def __init__(self, Xtr, Ytr, Xval = [], Yval= [],Xte = [], base_params = {}, hist_file='xgboost.log'):
        self.dtr  = xgb.DMatrix(Xtr, label=Ytr)
#        self.dval = xgb.DMatrix(Xval, label=Yval)
        self.dte = xgb.DMatrix(Xte)
        self.base_params = base_params
        self.hist_file = hist_file
        

    def predict(self, params, nrounds):
        booster = xgb.train(params, self.dtr,nrounds)
        preds_te  = booster.predict(self.dte)
        return preds_te
   

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

#Loading half
print("loading half ...")
data = np.load(DATA_PATH + '/' + 'Xtrain_challenge_owkin_half.npy')
lbls = np.load(DATA_PATH + '/' + 'Ytrain_challenge_owkin_half.npy')

train_percent = 1
split = int(train_percent * len(data))

Xtr  = data[:split]
#Xval = data[split:]
Xval = []

Ytr  = lbls[:split]
#Yval = lbls[split:]
Yval = []

Xte = np.load(DATA_PATH + '/' + 'Xtest_challenge_owkin_half.npy')
#dte = xgb.DMatrix(Xte)


print(f' \n Xtr : {Xtr.shape}')
#print(f' Xval : {Xval.shape}')
print(f' Xte : {Xte.shape}')
print(f' Ytr : {len(Ytr)}')
#print(f' Yval : {len(Yval)}')


xgb_model = XGBoost(Xtr,Ytr,Xval,Yval,Xte)


#best_params,best_booster_acc,best_booster = xgb_model.test_params({
#    'seed'            : [777],
#    'max_depth'       : [6], # tried [5, 6, 7, 8]
#    'objective'       : ['binary:logistic'],
#    'eval_metric'     : ['auc'],
#    'min_child_weight': [1], # [0.5, 1, 5, 10]
#    'booster' : ['gbtree'],
#    'nthread' : [8]
#}, 5000, 350)

#print(best_booster_acc)
#print(best_params)


params = {'max_depth' : 6,
          'objective'  : 'binary:logistic',
          'eval_metric' : 'auc',
          }

print("predicting with ...")
print(f"parameters : {params}")
ntrees = 5000
print("number of trees : " +str(ntrees))
preds = xgb_model.predict(params,ntrees)


#import pickle
#with open('preds_xgb18000_5000t_6d.pkl','wb') as f :
#	pickle.dump(preds,f)
#	print("Xgboost predictions saved")

#preds = best_booster.predict(dte,ntree_limit=best_booster.best_ntree_limit)
np.savetxt('preds_xgb18000_5000t_6d.csv', preds, fmt='%.5f', delimiter=',')
print("predictions saved")


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


