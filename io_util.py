import numpy as np
import logging

def _load_categorical_dataset_no_concate(paths):
    X = []
    T = []

    # Load datasets class by class
    num_classes = len(paths)
    I = np.identity(num_classes)
    for i in xrange(num_classes):
        x = np.load(paths[i]).astype(dtype=np.float32)
        t = np.tile(I[i], [len(x), 1]).astype(dtype=np.float32)
        X.append(x)
        T.append(t)
        logging.info('Load %d data for class %d from %s' % (len(x), i, paths[i]))

    return X, T

def load_categorical_dataset(paths):
    X, T = _load_categorical_dataset_no_concate(paths)
    X = np.asarray(np.concatenate(X))
    T = np.asarray(np.concatenate(T))

    return X, T

def load_categorical_dataset_for_training(paths, fracs):
    X, T = _load_categorical_dataset_no_concate(paths)
    
    K = len(X)

    assert (len(X) == len(T) and len(fracs) == K)

    X_Train = []
    T_Train = []
    X_Test = []
    T_Test = []

    for k in xrange(K):
        n = len(X[k])
        n_train = int(fracs[k] * n)
        inds = range(n)
        np.random.shuffle(inds)
        
        X_Train.append(X[k][inds[:n_train]])
        T_Train.append(T[k][inds[:n_train]])
        X_Test.append(X[k][inds[n_train:]])
        T_Test.append(T[k][inds[n_train:]])
    X = np.asarray(np.concatenate(X))
    T = np.asarray(np.concatenate(T))
    X_Train = np.asarray(np.concatenate(X_Train))
    T_Train = np.asarray(np.concatenate(T_Train))
    X_Test = np.asarray(np.concatenate(X_Test))
    T_Test = np.asarray(np.concatenate(T_Test))

    return X, T, X_Train, T_Train, X_Test, T_Test
