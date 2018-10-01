import sys
import time
import os

import numpy as np
import pandas as pd

use_rcsv = False
sys.path.insert(0, "../")
from utils import DATA_PATH

def load_data(fn_X:str, fn_y=None):
    start = time.time()
    # load X
    X = pd.read_csv(fn_X)
    X = X.iloc[:, 1:]
    # load Y
    if fn_y is not None:
        y = pd.read_csv(fn_y)
        y = y.iloc[:, 1]
    end = time.time()   

    load_time = round(end-start, 3)
    print(f"data loaded in {load_time} seconds")
    return X, y 


def collapse_snip_pairs(X):
    N, M = X.shape

    half_X = np.zeros((N, int(M/2)))
    for i in range(0, int(M/2)):
        half_X[:, i] = X.iloc[:, 2*i] + X.iloc[:, 2*i+1] # X is a dataframe

    return half_X


def save_data(X, y, filename_X: str, filename_y:str):
    
    np.save(filename_X, X)
    print(f"data saved : {filename_X}")
    if y is not None:
        np.save(filename_y, y)
        print(f"y saved : {filename_y}")


def main():
    # load data
    filename_Xtr = os.path.join(DATA_PATH, 'Xtrain_challenge_owkin.csv')
    filename_ytr = os.path.join(DATA_PATH, 'Ytrain_challenge_owkin.csv')
    filename_Xte = os.path.join(DATA_PATH, 'Xtest_challenge_owkin.csv')

    Xtr, ytr = load_data(filename_Xtr, filename_ytr)
    Xte = load_data(filename_Xte)

    # create collapsed data
    Xtr_half = collapse_snip_pairs(Xtr)
    Xte_half = collapse_snip_pairs(Xte)

    # save_data
    filename_Xtr = filename_Xtr.replace(".csv", ".npy")
    filename_Xtr_half = filename_Xtr.replace("Xtrain", "Xtrain_half")

    filename_ytr = filename_ytr.replace(".csv", ".npy")
    filename_Xte = filename_Xtr.replace(".csv", ".npy")
    filename_Xte_half = filename_Xte.replace("Xtest", "Xtest_half")

    save_data(Xtr, ytr, filename_Xtr, filename_ytr)
    save_data(Xtr_half, None, filename_Xtr_half, None)

    save_data(Xte, None, filename_Xte, None)
    save_data(Xte_half, None, filename_Xte_half, None)


if __name__ == '__main__':
    main()
