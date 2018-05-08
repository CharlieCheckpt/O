import os
import time
import numpy as np
import pickle as pkl
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid

import lightgbm as lgb
#import rcsv

DATA_PATH='data'



class Lgbm:

    def __init__(self, Xtr, Ytr,Xte, base_params = {}, hist_file='lgbm.log'):
        self.dtr  = lgb.Dataset(Xtr, label=Ytr)
       # self.dval = lgb.Dataset(Xval, label=Yval)
        self.Xte = Xte
        self.base_params = base_params
        self.hist_file = hist_file

    def predict(self, params, nrounds):
        booster = lgb.train(params, self.dtr,nrounds)
        preds_te  = booster.predict(self.Xte)
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
            booster = lgb.train(params, self.dtr, nrounds,
                                valid_sets=[self.dval],
                                early_stopping_rounds=early_stop,
                                evals_result=booster_eval)

            booster_acc = roc_auc_score(self.Yval, booster.predict(self.Xval,num_iteration=booster.best_iteration)) # CS
            print("booster_acc : " + str(booster_acc))

            if (booster_acc > best_booster_acc):
                best_booster_acc = booster_acc
                best_booster = booster
                best_params = params

            params_set.append((params, booster_acc))

            with open(self.hist_file, 'wb') as fp:
                pkl.dump(params_history, fp)

        return best_params,best_booster_acc


# =============================================================================
# Loading Data
# =============================================================================

#load half
print("\n loading half ...")
data = np.load(DATA_PATH + '/' + 'Xtrain_challenge_owkin_half.npy')
lbls = np.load(DATA_PATH + '/' + 'Ytrain_challenge_owkin_half.npy')
Xte = np.load(DATA_PATH + '/' + 'Xtest_challenge_owkin_half.npy')


print(f"\n Xtr : {data.shape} ")
print(f"\n Ytr : {lbls.shape} ")
print(f"\n Xte : {Xte.shape} ")

# load all
#print("loading all ...")
#data = rcsv.read(DATA_PATH + '/' + 'Xtrain_challenge_owkin.csv')
#data = data[1:, 1:]
#lbls = np.load(DATA_PATH + '/' + 'Ytrain_challenge_owkin_half.npy')


lgb_model = Lgbm(data, lbls,Xte)

params = {'num_leaves' : [31],
         'objective' : ['binary'],
         'metric' : ['auc'],
         'min_data_in_leaf':[1],
         'learning_rate' : [0.1],
        }

print(f"parameters : \n {params}")

preds = lgb_model.predict(params, 5000)
id_num = np.arange(0,13250)
id_list = ["ID"+str(id) for id in id_num]
d = {"Ids": id_list, 'TARGET':preds}
preds_2save = pd.DataFrame(data=d)

preds_2save.to_csv('pred_lgbm18000_5000t_31l_1mdl_0.1lr.csv', index = False)
print("predictions saved")

#best_params, best_booster_acc = lgb_model.test_params({
#    'seed' : [777],
#    'num_leaves' : [31], # tested [70,90,120,150]
#    'objective'   : ['binary'],
#    'metric'     : ['auc']
#
#}, 4000,400)
#print(" *-**-**-* best_booster_acc : "+str(best_booster_acc) + " *-**-**-* ")

#best_params = {
#    'seed' : [777],
#    'num_leaves' : [90], # tested [70,90,120,150]
#    'objective'   : ['binary'],
#    'metric'     : ['auc'],
#    'min_data_in_leaf' :[20], # [20, 50,100]
#    'feature_fraction' : [1],
#    'bagging_fraction' : [1],
#
#}

# preds
#testPreds = lgb_model.predict(best_params,2000)
#np.savetxt('pred_lgbm.csv', testPreds, fmt='%.5f', delimiter=',')

#print(" **** 1st Grid Search done **** ")
#print(" **** Starting Grid Search 2 *** ")
#lgb_model.base_params = best_params
#best_params, best_booster_acc = lgb_model.test_params({
#
#}, 500, 150)
#print(" **** 2nd Grid Search done **** ")
#best_params['val_acc'] = best_booster_acc
#import json
#
#with open('bestParams_lgbm.txt', 'w') as file:
#    file.write(json.dumps(best_params)) # use `json.loads` to do the reverse
