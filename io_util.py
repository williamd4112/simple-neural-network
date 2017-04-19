import numpy as np
import logging

def load_categorical_dataset(paths):
    X = []
    Y = []

    # Load datasets class by class
    num_classes = len(paths)
    I = np.identity(num_classes)
    for i in xrange(num_classes):
        x = np.load(paths[i]).astype(dtype=np.float32)
        y = np.tile(I[i], [len(x), 1]).astype(dtype=np.float32)
        X.append(x)
        Y.append(y)
        logging.info('Load %d data for class %d from %s' % (len(x), i, paths[i]))
    X = np.asarray(np.concatenate(X))
    Y = np.asarray(np.concatenate(Y))

    return X, Y
 
