"""Computes and save blended (average, median or extreme) predictions from multiple models.
"""
import csv
import glob
import os
import sys
import uuid

import numpy as np
import pandas as pd

# Import functions corresponding to each type of models
from O.catboost.predict_cb import get_cb_predictions, load_cb_models
from O.elasticnet.predict_en import get_en_predictions, load_en_models
from O.lgbm.predict_lgb import get_lgb_predictions, load_lgb_models
from O.xgboost.predict_xgb import get_xgb_predictions, load_xgb_models
from utils import load_data



# know where the script is from inside the script
PATH_SCRIPT = os.path.dirname(os.path.realpath(__file__))

def get_model_predictions(X, type_model: str, config: str, name_data_train:str):
    """Get predictions per model type.
    
    Args:
        X (np.array): feature matrix.
        type_model (str): type of model.
        config (str): config used for training.
        name_data_train (str): name of data used for training.
    
    Returns:
        np.array: predictions.
    """

    if type_model == "Xgboost":
        models = load_xgb_models(config, name_data_train)
        preds = get_xgb_predictions(X, models)
        print(f"Xgboost predictions computed (config:{config}, data type:{name_data_train})")
    elif type_model == "Lgbm":
        models = load_lgb_models(config, name_data_train)
        preds = get_lgb_predictions(X, models)
        print(
            f"Lgbm predictions computed (config:{config}, data type:{name_data_train})")
    elif type_model == "Catboost":
        models = load_cb_models(config, name_data_train)
        preds = get_cb_predictions(X, models)
        print(
            f"Catboost predictions computed (config:{config}, data type:{name_data_train})")
    elif type_model == "ElasticNet":
        models = load_en_models(config, name_data_train)
        preds = get_en_predictions(X, models)
        print(
            f"Elasticnet predictions computed (config:{config}, data type:{name_data_train})")

    return preds



def get_blend_predictions(X, type_models:list, configs:list, name_data:str, type_blend:str):
    """Computes blended predictions.
    
    Args:
        X (np.array): feature matrix.
        type_models (list): list of type of models.
        configs (list): list of type of configs.
        name_data (str): name of data to predict from.
        type_blend (str): type of blending?
    
    Returns:
        np.array: predictions.
    """
    name_data_train= name_data.replace("test", "train")
    preds = []
    for type_model, config in zip(type_models, configs):
        preds.append(get_model_predictions(X, type_model, config, name_data_train))
    # convert list to matrix
    preds = np.array(preds)
    if type_blend == "mean":
        preds = np.mean(preds, 0)
        print("average of predictions computed")
    elif type_blend == "median":
        preds = np.median(preds, 0)
        print("Median of predictions computed")
    elif type_blend == "extreme":
        preds_dist_from_half = np.abs(preds - 0.5)
        ind_max_dist_from_half = np.argmax(preds_dist_from_half, 0)
        ind_max_dist_from_half = ind_max_dist_from_half.squeeze()
        extreme_preds = np.zeros_like(preds)
        for i, ind in enumerate(ind_max_dist_from_half):
            extreme_preds[ind, i] = preds[ind, i]
        preds = extreme_preds
        print("extreme preds computed")
    return preds



def save_blend_preds(preds, type_models: list, configs: list, name_data: str, type_blend:str):
    """Save blended predictions.
    
    Args:
        preds (np.array): blended predictions.
        type_models (list): list of type of models.
        configs (list): list of configs.
        name_data (str): name of data to predict.
        type_blend (str): type of blending.
    """

    preds_dir = os.path.join(PATH_SCRIPT, "preds", name_data, type_blend)
    os.makedirs(preds_dir, exist_ok=True)

    # get unique filename
    unique_id = str(uuid.uuid4())
    fn_preds = "preds_" + unique_id + ".csv"
    fn_info_preds = "info_preds_" + unique_id + ".csv"
    fn_preds = os.path.join(preds_dir, fn_preds)
    fn_info_preds = os.path.join(preds_dir, fn_info_preds)

    np.savetxt(fn_preds, preds)
    with open(fn_info_preds, 'w') as csv_file:
        writer = csv.writer(csv_file)
        # write info
        writer.writerow(["name_data", name_data])
        writer.writerow(["type_models", type_models])
        writer.writerow(["configs", configs])

    print(f"Blended predictions saved in {fn_preds}")
    print(f"Blended predictions info saved in {fn_info_preds}")


def main(type_blend:str, type_models:list, configs: list, filename:str):
    name_data = filename.split(".")[0]
    X, _ = load_data(filename, None)
    # get average predictions from all models/configs
    blend_preds = get_blend_predictions(X, type_models, configs, name_data, type_blend)
    # save blended predictions
    save_blend_preds(blend_preds, type_models, configs, name_data, type_blend)


if __name__ == '__main__':
    import argparse
    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument("--type_blend", type=str, default="mean",
                        choices=["mean", "median", "extreme"], help="how to blend predictions")
    # must be in ["Xgboost", "Lgbm", "ElasticNet", "Catboost"]
    parser.add_argument("--type_models", type=str,
                        default="Xgboost Catboost Lgbm ElasticNet", help="models separated by a space.")
    parser.add_argument("--configs", type=str, default="test test test test",
                        help="configs separated by a space, e.g. <31leaves 70leaves 31leaves lr3>")
    parser.add_argument("--filename", type=str,
                        default="X_local.csv", help="name of data to predict")
    args = parser.parse_args()
    # convert string to list of strings
    args.type_models = args.type_models.split(' ')
    args.configs = args.configs.split(' ')
    main(args.type_blend, args.type_models, args.configs, args.filename)
