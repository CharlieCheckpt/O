# -*- coding: utf-8 -*-
"""

Title   : Challenge Owkin
Author  : Charlie Saillard
mail    : csaillar@ens-paris-saclay.fr
"""
#%% Import libraries
import os
os.chdir('/home/charlie/Documents/Challenge') # working directory
import numpy as np
#import sklearn
# from sklearn.neural_network import MLPClassifier
import time
from sklearn.metrics import roc_auc_score
# from sklearn.preprocessing import StandardScaler
import pandas as pd


#%% Load TRAINING and VALIDATION data
start = time.time()

tr = pd.read_csv('CSV/ALL_ie_18000/tr_18_all.csv')
tr = tr.drop(tr.columns[0], 1)
tr_out = pd.read_csv('CSV/ALL_ie_18000/tr_out_18.csv')

val = pd.read_csv('CSV/ALL_ie_18000/val_18.csv')
val = val.drop(val.columns[0], 1)
val_out = pd.read_csv('CSV/ALL_ie_18000/val_out_18.csv')


end= time.time()
print("Training and Validation data loaded in : "+str(end-start))
print(tr.shape)
print(val.shape)

#%% ************ XgBoost ************

from xgboost import XGBClassifier
clf = XGBClassifier(n_estimators=3000,objective='binary:logistic',max_depth=3,
                    min_child_weight=1, learning_rate=0.1)

eval_set  = [(tr,tr_out.values.ravel()), (val,val_out.values.ravel())]

clf.fit(tr, tr_out.values.ravel(), eval_set=eval_set,eval_metric="auc", 
            early_stopping_rounds=200)

## save model
import pickle
#with open('xgboost_9122.pkl', 'wb') as f:
#    pickle.dump(clf, f)
#    print("XgBoost model saved")

## and later you can load it
with open('Models/xgboost_18000.pkl', 'rb') as f:
    clf = pickle.load(f)
    
#%%  ************ Adaboost ************
    
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

clf2 = AdaBoostClassifier(DecisionTreeClassifier(max_depth=3),n_estimators=1000)

#eval_set  = [(tr,tr_out.values.ravel()), (val,val_out.values.ravel())]

clf2.fit(tr, tr_out.values.ravel())

#%% ************ Gradient Boosting Classifier ************
from sklearn.ensemble import GradientBoostingClassifier
clf3 = GradientBoostingClassifier(n_estimators=1000,max_depth=3,verbose=1)

clf3.fit(tr,tr_out.values.ravel())
# save model
#import pickle
#with open('gradient_boost_9122.pkl', 'wb') as f:
#    pickle.dump(clf3, f)
#    print("Gradient Boosting model saved")
import pickle
with open('gradient_boost_9122.pkl', 'wb') as f:
    pickle.dump(clf3, f)
    print("Gradient Boosting model saved")


#%% ************ Random Forest Classifier ************
from sklearn.ensemble import RandomForestClassifier
clf4 = RandomForestClassifier(n_estimators=1000,max_depth=3,verbose=3,
                              oob_score=True)

clf4.fit(tr,tr_out.values.ravel())
# save model
import pickle
with open('rf_9122.pkl', 'wb') as f:
    pickle.dump(clf4, f)
    print("Random forest model saved")
    
#%% *********** LightGBM ************
import lightgbm as lgb

train_data = lgb.Dataset(tr, label=tr_out.values.ravel())
valid_data = lgb.Dataset(val, label=val_out.values.ravel())
    
del tr
del val
param = {'num_trees':4000, 'objective':'binary','num_leaves':32}
param['metric'] = 'auc'
num_round = 10
bst = lgb.train(param, train_data, num_round, valid_sets=valid_data, early_stopping_rounds=500)

# save model
bst.save_model('light_gbm_18000.txt')
#load model
#bst = lgb.Booster(model_file='light_gbm_10864.txt')
#%% Results on TRAIN and VALIDATION sets
pred_tr = clf.predict_proba(tr)[:,1]
pred_val = clf.predict_proba(val)[:,1]
#pred_val =  bst.predict(val,num_iteration=bst.best_iteration)
print("AUC on train : " + str(roc_auc_score(tr_out.values.ravel(),pred_tr)))
print("AUC on val :" + str(roc_auc_score(val_out.values.ravel(),pred_val)))


#%% Load TEST data
test = pd.read_csv('CSV/ALL_ie_18000/test_18.csv') 
# test = pd.read_csv('CSV/test_18.csv') 
# test = test[tr.columns[0:len(tr.columns)]]
print(test.shape)
print("test set loaded")
#%% Predictions on test set
pred_te = clf.predict_proba(test)[:,1]
#pred_te = bst.predict(test,num_iteration=bst.best_iteration)

#save predictions
np.savetxt('./CSV/PREDS/my_pred_xgboost_18000.csv', pred_te, fmt='%.5f', delimiter=',')
np.savetxt('CSV/Stack/my_pred_val_lgbm_10864.csv', pred_val, fmt='%.5f', delimiter=',')
