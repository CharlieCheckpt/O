import os
import argparse
import yaml

import sys
sys.path.insert(0, "../")
from utils import load_data
from en_model import ElasticReg


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
    opts = configs[config]  #  dict of all options
    name_data = opts["fn_X"].split(".")[0]  #  name of data (without extension)
    # load data
    X, y = load_data(opts["fn_X"], opts["fn_y"])
    regressor = ElasticReg(X, y, config, name_data)
    regressor.cross_validation(opts["l1_ratio"], nfolds=opts["nfolds"])
    regressor.print_results()
    regressor.save_results()
    regressor.save_preds()
    regressor.save_models()


if __name__ == '__main__':
    main()
