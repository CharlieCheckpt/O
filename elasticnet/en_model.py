"""Logistic regression with elastic net penalty.
"""

import os
import yaml
import numpy as np
import time
import pickle

from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


class ElasticReg:
    def __init__(self, X, y, config: str, name_data=""):
        self.X = X
        self.y = y
        self.predictions = []  # predictions on validation set
        self.labels = []  # labels on validation set
        self.models = []  # trained models
        self.config = config  # name of config used
        self.name_data = name_data # name of data
        self.dict_res = {}  # dictionary of cross validation results

    def train(self, Xtr, ytr, l1_ratio: float):
        """Trains a cross-validated (on train set) elastic net model.
        
        Args:
            Xtr (np.array): 
            ytr (np.array): 
            l1_ratio (float): ratio between l1 and l2 penalty.
        
        Returns:
            model: cross validated elastic net model.
        """

        regr = ElasticNetCV(l1_ratio=l1_ratio, cv=3, n_jobs=-1, verbose=1)
        regr.fit(Xtr, ytr)
        return regr

    def cross_validation(self, l1_ratio: float, nfolds: int):
        """Cross-validation to assess performance of model.
        
        Args:
            l1_ratio (float): ratio between l1 and l2 penalty.
            nfolds (int): number of Stratified folds for cross-validation.
        """

        self.l1_ratio = l1_ratio
        dict_res = {}
        dict_res["auc_train"] = []
        dict_res["auc_val"] = []
        dict_res["nnz_coef"] = []  #  number of non zero coefs
        dict_res["alpha"] = []  #  value of alpha chosen by CV

        skf = StratifiedKFold(nfolds, random_state=777)
        start = time.time()
        for train_index, val_index in skf.split(self.X, self.y):
            Xtr, ytr = self.X[train_index], self.y[train_index]
            Xval, yval = self.X[val_index], self.y[val_index]

            regr = self.train(Xtr, ytr, l1_ratio)
            preds_tr, preds_val = regr.predict(Xtr), regr.predict(Xval)
            auc_tr, auc_val = roc_auc_score(
                ytr, preds_tr), roc_auc_score(yval, preds_val)
            auc_tr, auc_val = round(auc_tr, 3), round(auc_val, 3)
            auc_tr, auc_val = float(auc_tr), float(auc_val)  # make it yaml readable
            nnz_coef = int(np.sum(regr.coef_ != 0))  # make it yaml readable
            chosen_alpha = float(regr.alpha_)  # make it yaml readable

            dict_res["auc_train"].append(auc_tr)
            dict_res["auc_val"].append(auc_val)
            dict_res['nnz_coef'].append(nnz_coef)
            dict_res["alpha"].append(chosen_alpha)

            avg_auc_train, avg_auc_val = round(
                np.mean(dict_res["auc_train"]), 3), round(np.mean(dict_res["auc_val"]), 3)
            print(
                f"Auc on train : {auc_tr}, validation: {auc_val}, nnz coef : {nnz_coef}")
            print(
                f"Average Auc on train : {avg_auc_train}, validation : {avg_auc_val}")
            
            self.dict_res = dict_res
            self.predictions.append(preds_val)
            self.labels.append(yval)
            self.models.append(regr)

        end = time.time()
        # For easier read later, let's write the mean auc on val in the dictionary
        self.dict_res["mean_auc_val"] = float(
            round(np.mean(dict_res["auc_val"]), 4))
        self.dict_res["run_time"] = float(round(end-start, 3))


    def print_results(self):
        """Print results (auc, number of non zero coefficients) per fold.
        """

        avg_auc_train, avg_auc_val = round(np.mean(self.dict_res["auc_train"]), 3), round(
            np.mean(self.dict_res["auc_val"]), 3)
        print("*********************************************************************")
        print(f"Average Auc on train : {avg_auc_train}, val: {avg_auc_val}")
        for auc_val, nnz_coef in zip(self.dict_res["auc_val"], self.dict_res["nnz_coef"]):
            print(f"- {auc_val} (nnz coef:{nnz_coef}) -")


    def save_results(self):
        """Save results (auc, number of non zero coef) and parameters 
        in a file "./experiments/results.yaml".
        """
        # create name of directory where to save
        directory = os.path.join("./experiments", self.name_data, self.config)
        os.makedirs(directory, exist_ok=True)  # overwrite
        # dump parameters and results to same file "results.yaml"
        with open(os.path.join(directory, "results.yaml"), "w") as outfile:
            yaml.dump({"l1_ratio":self.l1_ratio}, outfile, default_flow_style=False)
            yaml.dump(self.dict_res, outfile, default_flow_style=False)

        print(f"results saved in {directory}")


    def save_preds(self):
        """Save validation predictions and labels on folder "./experiments/"
        """
        directory = os.path.join(
            "./experiments", self.name_data, self.config, "preds")
        os.makedirs(directory, exist_ok=True)
        for i, (preds, labels) in enumerate(zip(self.predictions, self.labels)):
            fn_preds = "preds_val" + str(i) + ".npy"
            fn_preds = os.path.join(directory, fn_preds)
            fn_labels = "labels_val" + str(i) + ".npy"
            fn_labels = os.path.join(directory, fn_labels)
            np.save(fn_preds, preds)
            np.save(fn_labels, labels)
        print(f"predictions and labels saved in {directory}")

    def save_models(self):
        directory = os.path.join(
            "./experiments", self.name_data, self.config, "models")
        os.makedirs(directory, exist_ok=True)
        for i, regr in enumerate(self.models):
            filename = os.path.join(directory, "model" + str(i) + ".pkl")
            pickle.dump(regr, open(filename,"wb"))
        print(f"models saved : {directory}")
