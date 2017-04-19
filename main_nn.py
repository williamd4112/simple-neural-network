import argparse
import sys
import logging
import numpy as np

from tqdm import * 

from io_util import *
from preprocess import Preprocessor
from model_np import MultiLayerPerceptron

def preprocess(X, T, d):
    X_normal, std = Preprocessor().normalize(X)
    X_phi, basis = Preprocessor().pca(X_normal, d)
    #X_phi = np.hstack((np.ones([len(X), 1]), X_phi))
    return X_phi

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='train/test task', 
            choices=['train', 'test', 'validate', 'eval', 'plot'], type=str, default='validate')
    parser.add_argument('--X', help='data (ordered)', type=str)
    parser.add_argument('--d', help='pca dimension', type=int, default=2)
    args = parser.parse_args()
    
    X, T = load_categorical_dataset(args.X.split(','))
    X_phi = preprocess(X, T, args.d)    

    sess = None
    model = MultiLayerPerceptron()
    model.fit(sess, X_phi, T)

