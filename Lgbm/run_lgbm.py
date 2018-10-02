import os
import argparse
import yaml

import sys
sys.path.insert(0, "../")
from utils import load_data
from lgbm_model import Lgbm


def main():
    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="test",
                        type=str, help="config for running Unet.")
    args = parser.parse_args()
    config = args.config  # name of config
    print(f"\n ----> You chose config : {config} <---- \n")
    # load config
    configs = yaml.load(open("configs.yaml"))
    opts = configs[config]  # dict of all options
    params = opts["params"]  # dict of parameters for training booster
    # load data
    X, y = load_data(opts["type_data"])
    # cross validation 
    booster = Lgbm(X, y, config, params)
    booster.cross_validation(
        nrounds=opts["nrounds"], nfolds=opts["nfolds"], early_stop_rounds=opts["early_stop_rounds"])
    booster.print_results()
    booster.save_results()
    booster.save_preds()
    booster.save_models()


if __name__ == '__main__':
    main()
