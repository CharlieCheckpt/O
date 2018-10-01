import time
import numpy as np
import pandas as pd
import os
import pickle
DATA_PATH = '../data'


def load_data(type_data: str):
    start = time.time()
    if type_data == "local":
        print("loading local data ...")
        X = np.load(os.path.join(DATA_PATH, "X_local.npy"))
        y = np.load(os.path.join(DATA_PATH, "y_local.npy"))

    if type_data == "18k":
        print("loading half data ...")
        X = np.load(os.path.join(DATA_PATH, 'Xtrain_challenge_owkin_half.npy'))
        y = np.load(os.path.join(DATA_PATH, 'Ytrain_challenge_owkin_half.npy'))

    elif type == "36k":
        print("loading full data ...")
        #  todo: create full matrix as .npy
        X = np.load(os.path.join(DATA_PATH, 'Xtrain_challenge_owkin.csv'))
        y = np.load(os.path.join(DATA_PATH, 'Ytrain_challenge_owkin_half.npy'))
    end = time.time()
    print(f"data loaded in {round(end-start, 3)} seconds")

    return X, y


def convert_csv_2_hdf(filename:str):
    """Read csv file line by line (too big to fit in memory) and write it to pickle file.
    """
    new_filename = filename.replace("csv", "pkl")

    i = 0
    with open(filename, "rb") as csvfile:
        for line in csvfile:
            print(f"line {i}")
            # save
            if i == 0:  # 1st time we create the file
                with open(new_filename, "wb") as pklfile:
                    pickle.dump(line, pklfile)
            else:  # we just append to the file
                with open(new_filename, 'ab') as pklfile:
                    pickle.dump(line, pklfile)
            
            i = i + 1

    print(f"data saved in {new_filename}")
