import numpy as np
import pickle as pkl
import os
import time

from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
import lightgbm as lgb

DATA_PATH='data'


class Lgbm:
    def __init__(self, X, y, config:str):
        self.X = X
        self.y = y
        self.models = []
        self.config = config

    def train(self, Xtr, ytr, Xdev, ydev, params:dict):
        """Trains booster.
        Returns:
            model
            dict eval result
        """
        dtr = lgb.Dataset(Xval, label=yval)
        ddev = lgb.Dataset(Xdev, label=ydev)
        history_eval = {}
        bst = lgb.train(params, train_set=dtr, num_boost_round=nrounds, 
                        valid_sets=[ddev], early_stopping_rounds=100, evals_result=history_eval) # todo: see what contains history_eval
        return bst, history_eval


    def cross_validation(self, params:dict, nfolds:int):
        """Stratified cross-validation.
        """
        dict_res = {}  # dictionary of results
        dict_res["nepochs_early_stop"] = []  # number of epochs before early stop
        dict_res["auc_train"] = []  # auc on train set
        dict_res["auc_val"] = []  # auc on validation set

        skf = StratifiedKFold(nfolds, random_state=777)
        for train_index, val_index in skf.split(X, y):
            Xtr, ytr = X[train_index], y[train_index]
            # creation of dev set for early stopping
            Xtr, Xdev, ytr, ydev = train_test_split(Xtr, ytr, test_size = 0.2, random_state=777)
            Xval, yval = X[val_index], y[val_index]

            booster, history_eval = self.train(Xtr, ytr, Xdev, ydev,params)
            preds_tr, preds_dev, preds_val = booster.predict(Xtr), booster.predict(Xdev), booster.predict(Xval)
            auc_tr, auc_dev, auc_val = roc_auc_score(ytr, preds_tr), roc_auc_score(ydev, preds_dev), roc_auc_score(yval, preds_val)
            auc_tr, auc_dev, auc_val = round(auc_tr, 3), round(auc_dev, 3), round(auc_val, 3)
            
            dict_res["train"].append(auc_tr)
            dict_res["val"].append(auc_val)
            dict_res["dev"].append(auc_dev)
            dict_res["nepochs_early_stop"].append(len(history_eval))

            print(f"Auc on train : {auc_tr}, validation: {auc_val}")
            print(f"Average Auc on train : {np.mean(dict_res["train"])}, validation : {np.mean(dict_res["val"])}")
        
            self.dict_res = dict_res
            self.models.append(booster)

    def print_results(self):
        avg_auc_train, avg_auc_val = round(np.mean(self.dict_res["auc_train"]),3), round(np.mean(self.dict_res["auc_val"]),3)
        print(f"*** Average Auc on train : {avg_auc_train}, val: {avg_auc_val}")
        print(f"(", self.dict_res["auc_val"], ") ***")

    def save_results(self):
        # create name of directory where to save
        directory = os.path.join("./experiments", self.config)
        os.makedirs(directory)
        with open(os.path.join(directory, 'results.csv'), 'wb') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in self.dict_res.items():
                writer.writerow([key, value])
        print(f"results saved in {directory}")
    
    def save_models(self):
        for i, booster in enumerate(self.models):
            filename = os.path.join("./experiments", self.config, "model" + str(i) + ".txt")
            booster.save_model(filename)
            print(f"model saved : {filename}")



def load_data(type_data:str):
    start = time.time()
    if type_data == "18k":
        print("loading half data ...")
        X = np.load(os.path.join(DATA_PATH, 'Xtrain_challenge_owkin_half.npy'))
    else:
        print("loading full data ...")
        X = np.load(os.path.join(DATA_PATH, 'Xtrain_challenge_owkin.csv')) # todo: create full matrix as .npy
    y = np.load(os.path.join(DATA_PATH, 'Ytrain_challenge_owkin_half.npy'))
    end = time.time()
    print(f"data loaded in {round(end-start, 3)} seconds")
    return X, y

def main():
    # parse config
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default",
                        type=str, help="config for running Unet.")
    args = parser.parse_args()
    config = args.config
    print(f"\n ----> You chose config : {config} <---- \n")
    # load config
    configs = yaml.load(open("configs.yaml"))
    opts = configs[config]
    # load data
    X, y = load_data(opts["type_data"])
    # cross validation 
    booster = Lgbm(X, y, config)
    booster.cross_validation(opts["params"], 5)
    booster.print_results()
    booster.save_results()
    booster.save_models()


if __name__ == '__main__':
    
    main()


# =============================================================================
# Loading Data
# =============================================================================
load_18k = True

if load_18k:
#load half
    start = time.time()
    print("\n loading half ...")
    data = np.load(DATA_PATH + '/' + 'Xtrain_challenge_owkin_half.npy')
    lbls = np.load(DATA_PATH + '/' + 'Ytrain_challenge_owkin_half.npy')
    end = time.time()
    print(f"data loaded in {round(end-start,3)} seconds")
else:
    # load all
    import rcsv
    start = time.time()
    print("loading all ...")
    data = rcsv.read(DATA_PATH + '/' + 'Xtrain_challenge_owkin.csv')
    data = data[1:, 1:]
    lbls = np.load(DATA_PATH + '/' + 'Ytrain_challenge_owkin_half.npy')
    end = time.time()
    print(f"data loaded in {round(end-start,3)} seconds")

# general parameters
nfolds = 3
ntrees = 5000
nearly_stop = 200
print(f"General parameters :\n nfolds : {nfolds}, ntrees : {ntrees}, nearly_stop : {nearly_stop}")

results = []

skf = StratifiedKFold(nfolds,random_state=777)


print(f" Cross validation with {nfolds} folds using StratifiedKfolds" )

# =============================================================================
# PARAMETERS TO TEST
# =============================================================================

param_grid = {
        'num_leaves' : [31], # 31 seems best
        'objective'   : ['binary'],
        'metric'     : ['auc'],
        'min_data_in_leaf' :[1], # [1,5, 20, 50,100]
        'learning_rate':[0.1], # 0.1 seems best
    # 'boosting' : ['gbdt'], # ['dart', 'gbdt'] gbdt seems best,
        # 'min_sum_hessian_in_leaf' : [1e-4,1e-3,1e-2], #1e-3 is default
        # 'min_gain_to_split' : [0,0.5],
#        'feature_fraction' : [0.8,1],
#        'bagging_fraction' : [0.8,1],
        }

# get number of combinations we are going to test in cv
import itertools as it
allNames = sorted(param_grid)
ncombi = len(list(it.product(*(param_grid[Name] for Name in allNames))))
print(f"*** Let's test {ncombi} combinations ***")

# =============================================================================
# STARTING THE "BIG LOOP"
# =============================================================================
compt = 0
compt_cv = 0
for param in ParameterGrid(param_grid):

    results.append(param.copy())
    results[compt]['auc'] = [] # here we store auc results for each cross validiation

    for train_index, val_index in skf.split(data, lbls):
        print(f'\n \n*** starting cross validation n°{compt_cv+1}/{ncombi*nfolds}... ***')
        print(f'*** starting cross validation n°{compt_cv+1}/{ncombi*nfolds}... ***')
        print(f'*** starting cross validation n°{compt_cv+1}/{ncombi*nfolds}... *** \n \n')

        print("TRAIN:", train_index, "VAL:", val_index)
        Xtr, Xval = data[train_index], data[val_index]
        Ytr, Yval = lbls[train_index], lbls[val_index]

        print(f' \n Xtr : {Xtr.shape}')
        print(f' Xval : {Xval.shape}')
        print(f' Ytr : {len(Ytr)}')
        print(f' Yval : {len(Yval)}')

        lgb_model = Lgbm(Xtr, Ytr, Xval, Yval)

        booster, best_booster_auc = lgb_model.test_params(param, ntrees, nearly_stop, Xval, Yval) # CHANGE IT  !!!!
        compt_cv += 1

        del Xtr,Xval,Ytr,Yval
        del lgb_model

        results[compt]['auc'].append(best_booster_auc)
        print(results[compt]['auc'])


    mean_val_auc = np.mean(results[compt]['auc'])
    print(f" \n \n**** CV n°{compt_cv-1} -- mean val accuracy : {mean_val_auc}")
    print(f"\n *** CV n°{compt_cv-1} -- val accuracies : {results[compt]['auc']}")
    compt = compt + 1


# =============================================================================
# SAVE RESULTS
# =============================================================================

from time import gmtime, strftime
date = strftime("%Y-%m-%d %H:%M:%S", gmtime()) # date for file saving
date = date.replace(" ", "_")
file_name = 'lgbm_cv_'+str(data.shape[1])+'k_'+str(ntrees)+'t_'+str(nfolds)+'f_gs__'+date+'.pkl'
with open(file_name, 'wb') as f:
    pkl.dump(results, f)
    print(f"\nlgbm log saved as {file_name}")
