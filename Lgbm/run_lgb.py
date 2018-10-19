import os
import argparse
import yaml

import sys
sys.path.insert(0, "../")
from utils import load_data
from lgb_model import Lgbm


def main():
    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="test",
                        type=str, help="config for running cross validation.")
    args = parser.parse_args()
    config = args.config  # name of config
    print(f"\n ----> You chose config : {config} <---- \n")
    # load config
    configs = yaml.load(open("configs.yaml"))
    opts = configs[config]  # dict of all options
    params = opts["params"]  # dict of parameters for training booster
    name_data = opts["fn_X"].split(".")[0]  #  name of data (without extension)
    # load data
    X, y = load_data(opts["fn_X"], opts["fn_y"])
    # cross validation
    booster = Lgbm(X, y, config, params, name_data)
    booster.cross_validation(
        nrounds=opts["nrounds"], nfolds=opts["nfolds"], early_stop_rounds=opts["early_stop_rounds"])
    booster.print_results()
    booster.save_results()
    booster.save_preds()
    booster.save_models()


if __name__ == '__main__':
    main()
