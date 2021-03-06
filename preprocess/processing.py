"""Preprocess data.
Loads .csv original train and test files (it's slow) and saves it as .npy.
Also collapses SNP pairs for Xtrain and Xtest and save the new dataset as .npy,
which has 18k columns instead of 36k).
"""
import time
from pathlib import Path
import numpy as np
import pandas as pd
from O.utils import DATA_PATH

def load_data(fn_X:Path, fn_y=None):
    """load data (X and potentially y)
    
    Args:
        fn_X (str): X filename.
        fn_y (str, optional): Defaults to None. y filename.
    
    Returns:
        X, potentially y:
    """

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
    if fn_y is not None:
        return X, y 
    else:
         return X


def collapse_snip_pairs(X):
    """Sum SNP pairwise to reduce size of the matrix from 36k columns to 18k.
    
    Args:
        X (pd.DataFrame):
    
    Returns:
        half_X (np.array):
    """

    
    N, M = X.shape

    half_X = np.zeros((N, int(M/2)))
    for i in range(0, int(M/2)):
        half_X[:, i] = X.iloc[:, 2*i] + X.iloc[:, 2*i+1] # X is a dataframe

    return half_X


def save_data(X, y, filename_X:Path, filename_y:Path):
    """save X and y.
    
    Args:
        X (np.array): 
        y (np.array): 
        filename_X (str): filename of X.
        filename_y (str): filename of y.
    """

    np.save(filename_X, X)
    print(f"data saved : {filename_X}")
    if y is not None:
        np.save(filename_y, y)
        print(f"y saved : {filename_y}")


def main():
    # load data
    filename_Xtr = DATA_PATH / 'Xtrain_challenge_owkin.csv'
    filename_ytr = DATA_PATH / 'Ytrain_challenge_owkin.csv'
    filename_Xte = DATA_PATH / 'Xtest_challenge_owkin.csv'

    Xtr, ytr = load_data(filename_Xtr, filename_ytr)
    Xte = load_data(filename_Xte)

    # create collapsed data
    Xtr_half = collapse_snip_pairs(Xtr)
    Xte_half = collapse_snip_pairs(Xte)

    # save_data
    new_filename_Xtr = Path(str(filename_Xtr).replace(".csv", ".npy"))
    filename_Xtr_half = Path(str(new_filename_Xtr).replace("Xtrain", "Xtrain_half"))
    new_filename_ytr = Path(str(filename_ytr).replace(".csv", ".npy"))
    new_filename_Xte = Path(str(filename_Xte).replace(".csv", ".npy"))
    filename_Xte_half = Path(str(new_filename_Xte).replace("Xtest", "Xtest_half"))

    save_data(Xtr, ytr, new_filename_Xtr, new_filename_ytr)
    save_data(Xtr_half, None, filename_Xtr_half, None)

    save_data(Xte, None, new_filename_Xte, None)
    save_data(Xte_half, None, filename_Xte_half, None)


if __name__ == '__main__':
    main()
