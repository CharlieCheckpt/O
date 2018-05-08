import rcsv
import numpy as np
import pandas as pd

use_rcsv = False

DATA_PATH='data'

def collapse_snip_pairs(X):
    N, M = X.shape

    half_X = np.zeros((N, int(M/2)))
    for i in range(0, int(M/2)):
        half_X[:, i] = X.iloc[:, 2*i] + X.iloc[:, 2*i+1] # X is a dataframe

    return half_X

if use_rcsv:
    Ytrain = rcsv.read(DATA_PATH + '/' + 'challenge_output_data_training_file_disease_prediction_from_dna_data.csv')
    Ytrain = Ytrain[1:, 1]
else:
    Ytrain = pd.read_csv(DATA_PATH + '/' + 'challenge_output_data_training_file_disease_prediction_from_dna_data.csv')
    Ytrain = Ytrain.iloc[:,1]

np.save(DATA_PATH + '/' + 'Ytrain_challenge_owkin_half.npy', Ytrain)
print(f"Ytr saved : {Ytrain.shape}")

if use_rcsv:
    Xtrain = rcsv.read(DATA_PATH + '/' + 'Xtrain_challenge_owkin.csv')
    Xtrain = Xtrain[1:, 1:]
else:
    Xtrain = pd.read_csv(DATA_PATH + '/' + 'Xtrain_challenge_owkin.csv')
    Xtrain = Xtrain.iloc[:,1:]

half_Xtrain = collapse_snip_pairs(Xtrain)
print(f"saving half Xtr : {half_Xtrain.shape}")
np.save(DATA_PATH + '/' + 'Xtrain_challenge_owkin_half.npy', half_Xtrain)
print("half Xtr saved")

if use_rcsv:
    Xtest = rcsv.read(DATA_PATH + '/' + 'Xtest_challenge_owkin.csv')
    Xtest = Xtest[1:, 1:]
else:
    Xtest = pd.read_csv(DATA_PATH + '/' + 'Xtest_challenge_owkin.csv')
    Xtest = Xtest.iloc[:,1:]

half_Xtest = collapse_snip_pairs(Xtest)
print(f"saving half Xte : {half_Xtest.shape}")
np.save(DATA_PATH + '/' + 'Xtest_challenge_owkin_half.npy', half_Xtest)
print("half_Xtest saved")
