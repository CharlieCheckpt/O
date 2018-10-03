"""Predict from data given by user with models trained with config.
"""
import sys
import argparse
import os
import glob
import yaml

import numpy as np
import pandas as pd
import xgboost as xgb

sys.path.insert(0, "../")
from utils import load_data


def load_xgb_models(config: str, name_data_train:str):
    """Load models trained with config.
    
    Args:
        config (str): name of config.
        name_data_train(str): name of data.
    
    Returns:
        list: list of trained models.
    """
    models = []
    path2models = os.path.join("./experiments", name_data_train, config, "models")
    for model in glob.glob(os.path.join(path2models, "*.txt")):
        print(f"model {model} loaded")
        booster = xgb.Booster()
        booster.load_model(model)
        models.append(booster)
    print("models loaded")
    return models


def get_xgb_predictions(X, models):
    """Computes predictions (average of predictions from all trained models).
    
    Args:
        X (np.array): matrix to predict from.
        models (list): list of trained models.
    
    Returns:
        np.array: predictions.
    """
    preds = []
    for model in models:
        preds.append(model.predict(xgb.DMatrix(X)))
    # convert list to matrix
    preds = np.array(preds)
    preds = np.mean(preds, axis=0)
    preds = np.expand_dims(preds, 1)
    assert preds.shape == (len(X), 1), preds.shape
    print("average of predictions computed")
    return preds


def save_predictions(predictions, config: str, name_data: str):
    """Save predictions as predictions_<data_name>.csv in path2preds.
    
    Args:
        predictions (np.array):
        config (str): name of config.
        name_data (str): name of data.
    """
    path2preds = os.path.join("./experiments", name_data, config)
    # remove extension name
    fn_preds = "preds.csv"
    # Create folder
    os.makedirs(path2preds, exist_ok=True)  #  overwrite
    # save in thois folder
    np.savetxt(os.path.join(path2preds, fn_preds), predictions, delimiter=",")
    print(f"predictions saved in {path2preds} as {fn_preds}")


def main():
    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="test",
                        type=str, help="config for predicting.")
    parser.add_argument("--fn_X", default="X_local.csv",
                        type=str, help="data to use for predicting.")
    args = parser.parse_args()
    config = args.config  # name of config
    name_data = args.fn_X.split(".")[0]  # name of data (without extension)
    print(f"\n ----> You chose to predict with config : {config} <---- \n")
    print(f"\n ----> You chose to predict data : {args.fn_X} <---- \n")

    # load data
    X, _ = load_data(filename_X=args.fn_X, filename_y=None)
    # Load trained models
    # models are trained on name_data_train
    name_data_train = name_data.replace("test", "train")
    models = load_xgb_models(config, name_data_train)
    # get predictions
    preds = get_xgb_predictions(X, models)
    # save predictions
    save_predictions(preds, config, name_data)


if __name__ == '__main__':
    main()
