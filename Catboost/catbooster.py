import time
import numpy as np
import pickle as pkl
from catboost import Pool, CatBoostClassifier, cv, CatboostIpythonWidget
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid



class CatBoost:

    def __init__(self, Xtr, Ytr, Xval, Yval, base_params = {}):
        self.Xtr = Xtr
        self.Ytr = Ytr
        self.Xval = Xval
        self.Yval = Yval
        self.base_params = base_params

    def test_params(self, grid, nrounds, early_stop):

        date_str = time.strftime("%d_%m_%Y__%h_%m", time.gmtime())
        hist_file = 'catboost' + '__' + date_str + '.log'

        params_set = []
        best_params = {}

        best_booster_acc = 0
        for update_params in ParameterGrid(grid):

            params = dict(self.base_params)
            params.update(update_params)

            model = CatBoostClassifier(
                eval_metric='AUC',
                iterations=nrounds,
                od_type='Iter',
                od_wait=early_stop,
                random_seed=42,
                logging_level='Silent'
            )

            Nfeatures = self.Xtr.shape[1]

            model.fit(self.Xtr, self.Ytr,
                    cat_features=np.arange(Nfeatures),
                    # cat_features= None, # None is default
                    eval_set=(self.Xval, self.Yval),
                    logging_level='Verbose',  # you can uncomment this for text output
		            use_best_model=True,
                    )


            pred_val = model.get_test_eval()
            booster_acc = roc_auc_score(self.Yval,pred_val)

            if (booster_acc > best_booster_acc):
                best_booster_acc = booster_acc
                best_params = params

            params_set.append((params, booster_acc))

            with open(hist_file, 'wb') as fp:
                pkl.dump(params_set, fp)

        return best_params, best_booster_acc
