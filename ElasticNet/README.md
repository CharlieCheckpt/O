**Get best models by cross-validation**
---------------------------------------
Run the cross-validation with 
`python run_elasticnet.py --config <name config>`.

Models and predictions (on validation sets) will be saved under the directory `O/Elastic/experiments/<name config>`.

**Predict from trained models**
---------------------------------------
Predict on a dataset with
`python predict_en.py --config <name config> --data <name data>`.

Dataset `<name data>` must be under the directory `O/data/`.
Predictions will be saved in `./experiments/<name config>/final_preds/preds_<name data>`

