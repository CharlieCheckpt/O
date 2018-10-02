**Get best models by cross-validation**
---------------------------------------
Run the cross-validation with 
`python run_cb.py --config <name config>`.

Models and predictions (on validation sets) will be saved under the directory `O/Catboost/experiments/<name config>/models` and
`O/Catboost/experiments/<name config>/preds`.

**Predict from trained models**
---------------------------------------
Predict on a dataset with
`python predict_cb.py --config <name config> --data <name data>`.

Dataset must be under the directory `O/data/`.
Predictions are saved under `O/Catboost/experiments/<name config>/final_preds`.
