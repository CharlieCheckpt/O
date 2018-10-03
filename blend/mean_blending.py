"""Computes and save blended (average) predictions from multiple models.
"""
import os
import sys
import glob
import argparse
import numpy as np
import pandas as pd
# Import functions corresponding to each type of models
sys.path.insert(0, "../Xgboost/")
from predict_xgb import load_xgb_models, get_xgb_predictions
sys.path.insert(0, "../Lgbm/")
from predict_lgbm import load_lgb_models, get_lgb_predictions
sys.path.insert(0, "../Catboost/")
from predict_cb import load_cb_models, get_cb_predictions
sys.path.insert(0, "../Elasticnet/")
from predict_elasticnet import load_en_models, get_en_predictions
sys.path.insert(0, "../")
from utils import load_data

def get_model_predictions(X, type_model: str, config: str, name_data:str):
    
    if type_model == "Xgboost":
        models = load_xgb_models(config, name_data)
        preds = get_xgb_predictions(X, models)
        print(f"Xgboost predictions computed (config:{config}, data type:{name_data})")
    elif type_model == "Lgbm":
        models = load_lgb_models(config, name_data)
        preds = get_lgb_predictions(X, models)
        print(
            f"Lgbm predictions computed (config:{config}, data type:{name_data}")
    elif type_model == "Catboost":
        models = load_cb_models(config, name_data)
        preds = get_cb_predictions(X, models)
        print(
            f"Catboost predictions computed (config:{config}, data type:{name_data}")
    elif type_model == "Elasticnet":
        models = load_en_models(config, name_data)
        preds = get_en_predictions(X, models)
        print(
            f"Elasticnet predictions computed (config:{config}, data type:{name_data}")

    return preds



def get_blend_predictions(X, type_models:list, configs:list, name_data:str):
    preds = []
    for type_model, config in zip(type_models, configs):
        preds.append(get_model_predictions(X, type_model, config, name_data))
    # convert list to matrix
    preds = np.array(preds)
    preds = np.mean(preds, 0)
    print("average of predictions computed")
    return preds



def save_blend_preds(preds, models: list, configs: list, name_data: str):
    print("Blended predictions saved in ")
    raise NotImplementedError


def main():
    # parse config
    parser = argparse.ArgumentParser()
    # must be in ["Xgboost", "Lgbm", "ElasticNet", "Catboost"]
    parser.add_argument("--type_models", type=list, help="list of models.")
    parser.add_argument("--configs", type=list, help="list of configs.")
    parser.add_argument("--filename", type=str, help="name of data to predict")
    
    args = parser.parse_args()
    name_data = args.filename.split(".")[0]
    X, _ = load_data(args.filename)
    # get average predictions from all models/configs
    blend_preds = get_blend_predictions(X, args.type_models, args.configs, name_data)
    # save blended predictions
    save_blend_preds(blend_preds, args.type_models, args.configs, name_data)


if __name__ == '__main__':
    main()
