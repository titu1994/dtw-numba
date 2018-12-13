import numpy as np
import pandas as pd

import os


def load_dataset(dataset_name, normalize_timeseries=False, verbose=True):
    train_path = 'data/' + dataset_name + '_TRAIN'
    test_path = 'data/' + dataset_name + '_TEST'

    if verbose: print("Loading train / test dataset : ", train_path, test_path)

    if os.path.exists(train_path):
        df = pd.read_csv(train_path, header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (train_path))

    # remove all columns which are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    # fill all missing columns with 0
    df.fillna(0, inplace=True)

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    y_train = df[[0]].values
    nb_classes = len(np.unique(y_train))
    y_train = (y_train - y_train.min()) / (y_train.max() - y_train.min()) * (nb_classes - 1)

    # drop labels column from train set X
    df.drop(df.columns[0], axis=1, inplace=True)

    X_train = df.values

    # scale the values
    if normalize_timeseries:
        X_train_mean = X_train.mean(axis=-1, keepdims=True)
        X_train_std = X_train.std(axis=-1, keepdims=True)
        X_train = (X_train - X_train_mean) / (X_train_std + 1e-8)

    if verbose: print("Finished loading train dataset..")

    if os.path.exists(test_path):
        df = pd.read_csv(test_path, header=None, encoding='latin-1')
    else:
        raise FileNotFoundError('File %s not found!' % (test_path))

    # remove all columns which are completely empty
    df.dropna(axis=1, how='all', inplace=True)

    # fill all missing columns with 0
    df.fillna(0, inplace=True)

    # extract labels Y and normalize to [0 - (MAX - 1)] range
    y_test = df[[0]].values
    nb_classes = len(np.unique(y_test))
    y_test = (y_test - y_test.min()) / (y_test.max() - y_test.min()) * (nb_classes - 1)

    # drop labels column from train set X
    df.drop(df.columns[0], axis=1, inplace=True)

    X_test = df.values

    # scale the values
    if normalize_timeseries:
        X_test_mean = X_test.mean(axis=-1, keepdims=True)
        X_test_std = X_test.std(axis=-1, keepdims=True)
        X_test = (X_test - X_test_mean) / (X_test_std + 1e-8)

    if verbose:
        print("Finished loading test dataset..")
        print()
        print("Number of train samples : ", X_train.shape[0], "Number of test samples : ", X_test.shape[0])
        print("Number of classes : ", nb_classes)
        print("Sequence length : ", X_train.shape[-1])


    return X_train, y_train, X_test, y_test


