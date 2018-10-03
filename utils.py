import time
import numpy as np
import pandas as pd
import os
import pickle
DATA_PATH = '/home/ubuntu/workspace/O/data'


def load_data(filename_X:str, filename_y:str):
    """Load X and potentially y. Handles .csv and .npy.
    
    Args:
        filename_X (str): file name of X.
        filename_y (str): file name of y.
    
    Returns:
        X, y: np.arrays. y is None if filename_y is None.
    """

    start = time.time()
    if ".npy" in filename_X:
        X = np.load(os.path.join(DATA_PATH, filename_X))
        if filename_y is not None:
            y = np.load(os.path.join(DATA_PATH, filename_y))
        else:
            y = None
    elif ".csv" in filename_X:
        X = pd.read_csv(os.path.join(DATA_PATH, filename_X)).values
        if filename_y is not None:
            y = pd.read_csv(os.path.join(DATA_PATH, filename_y)).values
        else:
            y = None
    end = time.time()

    print(f"data loaded in {round(end-start, 3)} seconds")
    return X, y


def convert_csv_2_hdf_linebyline(filename:str):
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
