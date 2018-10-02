Run the cross-validation with 
`python run_cb.py --config <name config>`.

Models and predictions (on validation sets) will be saved under the directory `O/Catboost/experiments/<name config>`.

Predict on a dataset with
`python predict_cb.py --config <name config> --data <name data>`.

Dataset must be under the directory `O/data/`.

