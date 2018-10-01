import time
import numpy as np
import os
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
