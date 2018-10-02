import sys
import argparse
import os
import glob
import yaml

import numpy as np
import pandas as pd
import xgboost as xgb


def load_data_2_predict(data_name: str):
    """Load data to predict.
    """
    filename = os.path.join("../data/", data_name)
    if ".csv" in data_name:
        X = pd.read_csv(filename, index_col=0).values
    elif ".npy" in data_name:
        X = np.load(filename)
    else:
        print(f"data file {data_name} not recognized")
        sys.exit(1)
    X = xgb.DMatrix(X)  # avoid error when predicting later
    print(f"data loaded")
    return X


def load_models(config: str):
    """Load trained models corresponding to config.
    """
    models = []
    path2models = os.path.join("./experiments", config, "models")
    for model in glob.glob(os.path.join(path2models, "*.txt")):
        print(f"model {model} loaded")
        booster = xgb.Booster()
        booster.load_model(model)
        models.append(booster)
    print("models loaded")
    return models


def get_predictions(X, models):
    '''Get predictions (average of predictions from all models loaded).
    '''
    preds = np.zeros((len(models), X.num_row()))
    for i, model in enumerate(models):
        preds[i] = model.predict(X)
    preds = np.mean(preds, axis=0)
    preds = np.expand_dims(preds, 1)
    assert preds.shape == (X.num_row(), 1), preds.shape
    print("average of predictions computed")
    return preds


def save_predictions(predictions, config: str, data_name: str):
    '''Save predictions as .csv .
    '''
    path2preds = os.path.join("./experiments", config, "final_preds")
    # remove extension name
    fn_preds = "preds_" + data_name.split(".")[0] + ".csv"
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
    parser.add_argument("--data", default="X_local.csv",
                        type=str, help="data to use for predicting.")
    args = parser.parse_args()
    config = args.config  # name of config
    data_name = args.data
    print(f"\n ----> You chose to predict with config : {config} <---- \n")
    print(f"\n ----> You chose to predict data : {data_name} <---- \n")

    # load data
    X = load_data_2_predict(data_name=data_name)
    # Load trained models
    models = load_models(config)
    # get predictions
    preds = get_predictions(X, models)
    # save predictions
    save_predictions(preds, config, data_name)


if __name__ == '__main__':
    main()
