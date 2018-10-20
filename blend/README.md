# Blending
Once we have several type of models trained, one may want to blend predictions of these different models in order to improve prediction power. The most obvious type of blending is to compute the average of the model predictions.


You can run the blending script like this : 
`python blending.py --type_blend <type blend> --type_models <type models> --configs <configs> --filename <filename>`

* `<type blend>` : type of blending. e.g. *"mean"*.
* `<type models>` : type of models. e.g. *"Xgboost Xgboost Catboost ElasticNet"*
* `<configs>` : 
* `<filename>` : filename of data to predict, e.g. *Xtest_challenge_owkin.csv*.