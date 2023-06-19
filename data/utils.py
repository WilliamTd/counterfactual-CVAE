import pandas as pd
import numpy as np
import pickle

def load_cured_data(path, ctype):
    X = np.load(f'{path}/compressed_X.npz')['a']
    Y = pd.read_csv(f'{path}/Y.csv')
    target_y = np.load(f'{path}/y_{ctype}.npy')
    with open(f'{path}/mlb_{ctype}.pkl', 'rb') as tokenizer:
        mlb = pickle.load(tokenizer)
    return X, Y, target_y, mlb


def split_ptb(X, Y, target_y, test_fold):
    # Train
    X_train = X[np.where(Y.strat_fold != test_fold)]
    Y_train = target_y[np.where(Y.strat_fold != test_fold)]

    # Test
    X_test = X[np.where(Y.strat_fold == test_fold)]
    Y_test = target_y[np.where(Y.strat_fold == test_fold)]

    assert X_train.shape[0] == Y_train.shape[0], "X_train and Y_train should be the same size"
    assert X_test.shape[0] == Y_test.shape[0], "X_test and Y_test should be the same size"

    return X_train, Y_train, X_test, Y_test
