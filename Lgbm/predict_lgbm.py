"""Predict from data given by user with models trained with config.
"""
import sys
import argparse
import os
import glob
import yaml

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, "../")
from utils import load_data


def load_lgb_models(config:str, name_data:str):
    """Load models trained with config.
    
    Args:
        config (str): name of config.
        name_data (str): name of data.
    
    Returns:
        list: list of trained models.
    """
    models = []
    path2models = os.path.join("./experiments", config, name_data, "models")
    for model in glob.glob(os.path.join(path2models, "*.txt")):
        print(f"model {model} loaded")
        booster = lgb.Booster(model_file=model)
        models.append(booster)
    print("models loaded")
    return models


def get_lgb_predictions(X, models):
    """Computes predictions (average of predictions from all trained models).
    
    Args:
        X (np.array): matrix to predict from.
        models (list): list of trained models.
    
    Returns:
        np.array: predictions.
    """
    preds = np.zeros((len(models), len(X)))
    for i, model in enumerate(models):
        preds[i] = model.predict(X)
    preds = np.mean(preds, axis=0)
    preds = np.expand_dims(preds, 1)
    assert preds.shape == (len(X), 1), preds.shape
    print("average of predictions computed")
    return preds


def save_predictions(predictions, config:str, data_name:str):
    """Save predictions as predictions_<data_name>.csv in path2preds.
    
    Args:
        predictions (np.array):
        config (str): name of config.
        data_name (str): name of data.
    """
    path2preds = os.path.join("./experiments", config, "final_preds")
    fn_preds = "preds_" + data_name.split(".")[0] + ".csv" # remove extension name
    # Create folder
    os.makedirs(path2preds, exist_ok=True)  # overwrite
    # save in thois folder
    np.savetxt(os.path.join(path2preds, fn_preds), predictions, delimiter=",")
    print(f"predictions saved in {path2preds} as {fn_preds}")


def main():
    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="test",
                        type=str, help="config for predicting.")
    parser.add_argument("--fn_data", default="X_local.csv",
                        type=str, help="data to use for predicting.")
    args = parser.parse_args()
    config = args.config  # name of config
    name_data = args.fn_data.split(".")[0]  #  name of data (without extension)
    print(f"\n ----> You chose to predict with config : {config} <---- \n")
    print(f"\n ----> You chose to predict data : {args.fn_data} <---- \n")

    # load data
    X, _ = load_data(filename_X=args.fn_data, filename_y=None)
    # Load trained models
    models = load_lgb_models(config, args.type_data)
    # get predictions
    preds = get_lgb_predictions(X, models)
    # save predictions
    save_predictions(preds, config, name_data)


if __name__ == '__main__':
    main()
