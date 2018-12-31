"""Xgboost model for binary classification.
"""
import os
import csv
import numpy as np
import time
import yaml

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

import xgboost as xgb


class Xgboost:
    def __init__(self, X, y, config: str, params: dict, name_data=""):
        self.X = X
        self.y = y
        self.models = []  # trained models
        self.predictions = []
        self.labels = []  
        self.config = config  # name of config
        self.name_data = name_data  # name of data
        self.params = params  # parameters
        self.dict_res = {}  # dictionary of results

    def train(self, dtr, ddev, nrounds: int, early_stop_rounds: int):
        """Trains booster with early stopping on a dev set.

        Args:
            dtr (xgb.DMatrix): 
            ddev (xgb.DMatrix): 
            nrounds (int): number of maximum rounds.
            early_stop_rounds (int): number of non-improving rounds before early stopping.

        Returns:
            [model, history]: trained model and history of training on dev set.
        """
        history_eval = {}
        bst = xgb.train(self.params, dtr, nrounds, evals=[(dtr, "train"), (ddev, '"dev')],
                        early_stopping_rounds=early_stop_rounds, evals_result=history_eval)

        return bst, history_eval

    def cross_validation(self, nrounds: int, nfolds: int, early_stop_rounds: int):
        """Stratified cross-validation.

        Args:
            nrounds (int): number of maximum rounds.
            nfolds (int): number of folds.
            early_stop_rounds (int): number of non-improving rounds before early stopping.
        """
        # useful for saving later
        self.params["nrounds"] = nrounds

        dict_res = {}  #  dictionary of results
        dict_res["nepochs"] = []  # number of epochs before early stop
        dict_res["auc_train"] = []  #  auc on train set
        dict_res["auc_dev"] = []  #  auc on train set
        dict_res["auc_val"] = []  #  auc on validation set

        skf = StratifiedKFold(nfolds, random_state=777)
        start = time.time()
        for train_index, val_index in skf.split(self.X, self.y):
            Xtr, ytr = self.X[train_index], self.y[train_index]
            # creation of dev set for early stopping
            Xtr, Xdev, ytr, ydev = train_test_split(
                Xtr, ytr, test_size=0.2, stratify=ytr, random_state=777)
            Xval, yval = self.X[val_index], self.y[val_index]

            dtr = xgb.DMatrix(Xtr, ytr)
            ddev = xgb.DMatrix(Xdev, ydev)
            dval = xgb.DMatrix(Xval, yval)

            booster, history_eval = self.train(
                dtr, ddev, nrounds, early_stop_rounds)
            preds_tr, preds_dev, preds_val = booster.predict(
                dtr), booster.predict(ddev), booster.predict(dval)

            auc_tr, auc_dev, auc_val = roc_auc_score(ytr, preds_tr), roc_auc_score(
                ydev, preds_dev), roc_auc_score(yval, preds_val)
            auc_tr, auc_dev, auc_val = round(auc_tr, 3), round(
                auc_dev, 3), round(auc_val, 3)
            # convert from np.float to float to be yaml readable
            auc_tr, auc_val, auc_dev = float(auc_tr), float(auc_val), float(auc_dev)  

            dict_res["auc_train"].append(auc_tr)
            dict_res["auc_val"].append(auc_val)
            dict_res["auc_dev"].append(auc_dev)
            dict_res["nepochs"].append(len(history_eval["train"]["auc"]))

            avg_auc_train, avg_auc_val = round(
                np.mean(dict_res["auc_train"]), 3), round(np.mean(dict_res["auc_val"]), 3)
            print(
                f"Auc on train : {auc_tr}, dev : {auc_dev}, validation: {auc_val}")
            print(
                f"Average Auc on train : {avg_auc_train}, validation : {avg_auc_val}")

            self.dict_res = dict_res
            self.predictions.append(preds_val)
            self.labels.append(yval)
            self.models.append(booster)

        end = time.time()
        # For easier read later, let's write the mean auc on val in the dictionary
        self.dict_res["mean_auc_val"] = float(
            round(np.mean(dict_res["auc_val"]), 4))
        self.dict_res["run_time"] = float(round(end-start, 3))

    def print_results(self):
        """Print results (auc) per fold.
        """
        avg_auc_train, avg_auc_val = round(np.mean(self.dict_res["auc_train"]), 3), round(
            np.mean(self.dict_res["auc_val"]), 3)
        print("*********************************************************************")
        print(
            f"Average Auc on train : {avg_auc_train}, val: {avg_auc_val}")
        details_val = [(auc, nepochs) for auc, nepochs in zip(
            self.dict_res["auc_val"], self.dict_res["nepochs"])]
        details_val = [
            str(d[0])+' (' + str(d[1])+' ep)' for d in details_val]
        for auc_val, auc_dev, nep in zip(self.dict_res["auc_val"], self.dict_res["auc_dev"], self.dict_res["nepochs"]):
            print(f"- {auc_val} (dev: {auc_dev}, nep:{nep}) -")

    def save_results(self):
        """Save results (auc, number of non zero coef) and parameters 
        in a file "./experiments/results.yaml".
        """
        # create name of directory where to save
        directory = os.path.join("./experiments", self.name_data, self.config)
        os.makedirs(directory, exist_ok=True)  # overwrite
        # dump parameters and results to same file "results.yaml"
        with open(os.path.join(directory, "results.yaml"), "w") as outfile:
            yaml.dump(self.params, outfile, default_flow_style=False)
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
        """Save trained models in "./experiments/models".
        """
        directory = os.path.join(
            "./experiments", self.name_data, self.config, "models")
        os.makedirs(directory, exist_ok=True)
        for i, booster in enumerate(self.models):
            filename = os.path.join(directory, "model" + str(i) + ".txt")
            booster.save_model(filename)
        print(f"models saved : {directory}")
