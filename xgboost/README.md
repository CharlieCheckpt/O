**Get best models by cross-validation**
---------------------------------------
In order to identify which parameters are the best, one needs to use cross-validation. You can specify the parameters in `configs.yaml`.

You can run the cross-validation like this : 
```sh
python run_xgb.py --config <name config>
```

Results and parameters used are saved as `O/Xgboost/experiments/<name_data>/<name config>/results.yaml`.
Models and predictions (on validation sets) are saved under the directory `O/Xgboost/experiments/<name config>/models` and
`O/Xgboost/experiments/<name config>/preds`.

**Predict from trained models**
---------------------------------------
Once you have identify best models with cross-validation, you can use the best models in order to predict on new (test) data. The script `predict_xgb.py` average the predictions from models from each split and saves the predictions.  

You can use the script like this : 
```sh
python predict_xgb.py --config <name config> --data <name data>
```

Dataset `<name data>` must be under the directory `O/data/`.
Predictions will be saved in `./experiments/<name config>/final_preds/preds_<name data>`