import argparse
import sys
import logging
import numpy as np

from tqdm import * 

from io_util import *
from preprocess import Preprocessor
from model_np import MultiLayerPerceptron

def get_basis(pre, d, X, T):
    X_normal, std = Preprocessor().normalize(X)
    _, basis = Preprocessor().pca(X_normal, d)
    
    return _, basis, std

def apply_basis(X, basis, std):
    X_Phi = (X / std).dot(basis.T)
    return X_Phi

if __name__ == '__main__':
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='train/test task', 
            choices=['train', 'test', 'validate', 'eval', 'plot'], type=str, default='validate')
    parser.add_argument('--X', help='data (ordered)', type=str)
    parser.add_argument('--d', help='basis dimension', type=int, default=2)
    parser.add_argument('--frac', help='fraction of training set', type=str, default='0.8,0.8,0.8')
    parser.add_argument('--pre', help='preprocess method', 
            choices=['pca', 'hist', 'lda'], type=str, default='pca')

    args = parser.parse_args()
    
    X, T, X_Train, T_Train, X_Test, T_Test = load_categorical_dataset_for_training(args.X.split(','), [float(f) for f in args.frac.split(',')])
    X_Phi, basis, std = get_basis(args.pre, args.d, X, T)    
    
    X_Phi_Train = apply_basis(X_Train, basis, std)
    X_Phi_Test = apply_basis(X_Test, basis, std)

    sess = None
    model = MultiLayerPerceptron()
    model.fit(sess, X_Phi, T)

