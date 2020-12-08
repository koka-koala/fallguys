""" Split and match the data with labels
"""

import numpy as np
import tensorflow as tf
import pandas as pd
from pathlib import Path
import os


def make_npz(to_path):
    '''Create npz file'''
    file_list = ['climb-stairs', 'stay', 'down-stairs', 'walk', 'cycle', 'fall', 'run']
    for file in file_list:
        if file == ".DS_Store":
            continue
        path = os.path.join(to_path,file)
        npy_file = os.listdir(path)
        category = file
        print(f"start {file}")
        for npy in npy_file:
            if npy == ".DS_Store":
                continue
            file_path = os.path.join(path,npy)
            value = np.load(file_path, allow_pickle=True)[0][:,[2,3,4,]]
            list_np.append(value)
            category_int = file_list.index(category)
            list_c.append(category_int)
        print(f"end {file}")
    np.savez("test.npz", value = np.array(list_np), cat = np.array(list_c))
    print("Finsh creating npz file")


def combine_npz(to_path_train, to_path_test):
    '''combine 2 npz file, train and test'''
    test_path = to_path_train
    train_path = to_path_test
    npzfile_train = np.load(train_path,allow_pickle=True)
    npzfile_test = np.load(test_path,allow_pickle=True)
    np.savez("fallguys.npz", train_value = npzfile_train["value"], train_cat = npzfile_train["cat"],
        test_value = npzfile_test["value"], test_cat = npzfile_test["cat"])




