import argparse
import sys
import logging
import numpy as np

from tqdm import * 

from util import one_hot
from plot import plot_decision_boundary
from preprocess import Preprocessor
from model_np import ProbabilisticGenerativeModel, ProbabilisticDiscriminativeModel, MultiLayerPerceptron

import cPickle as pickle

def get_model(args):
    if args.model == 'gen':
        return ProbabilisticGenerativeModel()
    elif args.model == 'dis':
        return ProbabilisticDiscriminativeModel(lr=args.lr, 
                                                epochs=args.epoch, 
                                                batch_size=args.batch_size,
                                                tolerance=args.tolerance)
    elif args.model == 'nn':
        n_hidden_units = [int(s) for s in args.h_d.split(',')]
        logging.info('Create Multi-Layer Perceptron with (n_hidden_layer = %d, hidden_dims = %s, activations = %s)' % (args.h, args.h_d, args.activations))
        return MultiLayerPerceptron(lr=args.lr, 
                                    epochs=args.epoch, 
                                    batch_size=args.batch_size,
                                    tolerance=args.tolerance,
                                    n_hidden_layers=args.h,
                                    n_hidden_units=n_hidden_units,
                                    hidden_activations=args.activations.split(','))

def get_model_test(args):
    assert args.load != None
    if args.model == 'gen':
        model = ProbabilisticGenerativeModel()
    elif args.model == 'dis':
        model = ProbabilisticDiscriminativeModel(lr=None, 
                                                epochs=None, 
                                                batch_size=None,
                                                tolerance=None)
    elif args.model == 'nn':
        n_hidden_units = [int(s) for s in args.h_d.split(',')]
        logging.info('Create Multi-Layer Perceptron with (n_hidden_layer = %d, hidden_dims = %s, activations = %s)' % (args.h, args.h_d, args.activations))
        model = MultiLayerPerceptron(lr=args.lr, 
                                    epochs=args.epoch, 
                                    batch_size=args.batch_size,
                                    tolerance=args.tolerance,
                                    n_hidden_layers=args.h,
                                    n_hidden_units=n_hidden_units,
                                    hidden_activations=args.activations.split(','))
    model.load(args.load)

    return model

def load(args):
    logging.info('Loading stddev from %s ...' % args.std)
    std = np.load(args.std)

    logging.info('Loading basis from %s ...' % args.basis)
    phi = np.load(args.basis)

    model = get_model_test(args)
    return model, std, phi

def evaluate(args, model, x_, t_):
    sess = None
    accuracy = model.eval(sess, x_, t_)
    logging.info('Accuracy = %f' % accuracy)

def preprocess(args, X, T):
    d = args.d
    X_normal, std = Preprocessor().normalize(X)
    if args.pre == 'pca':
        X_phi, phi = Preprocessor().pca(X_normal, k=d)
    elif args.pre == 'lda':
        X_phi, phi = Preprocessor().lda(X_normal, T, d=d)
    else:
        X_phi = X
        phi = np.ones(d)
    bias = np.ones(len(X))[:, np.newaxis]
    X_phi = np.hstack((bias, X_phi))
    
    return X_phi, phi, std

def preprocess_test(X, std, phi):
    X_normal = X / std
    X_phi = phi.transform(X_normal)
    bias = np.ones(len(X))[:, np.newaxis]
    X_phi = np.hstack((bias, X_phi))
    return X_phi

def load_dataset_with_class(args):
    X = []
    Y = []

    # Load datasets class by class
    datasets = args.X.split(',')
    num_classes = len(datasets)
    parts = np.zeros([num_classes, 2], dtype=np.int32)
    count = 0
    for i in xrange(num_classes):
        x = np.matrix(np.load(datasets[i]), dtype=np.float32)
        y = np.tile(one_hot(num_classes, i), [len(x), 1]).astype(dtype=np.float32)
        parts[i, 0] = count
        parts[i, 1] = len(x)
        X.append(x)
        Y.append(y)
        count += len(x)
        logging.info('Load %d data for class %d from %s' % (len(x), i, datasets[i]))
    X = np.asarray(np.concatenate(X))
    Y = np.asarray(np.concatenate(Y))

    return X, Y, parts

def plot(args):
    assert args.load != None
    model, std, phi = load(args)
    sess = None

    # for plotting training data    
    x_, t_, _ = load_dataset_with_class(args)
    x_phi = preprocess_test(x_, std, phi)[:, 1:]

    def func(X):    
        # X consists of feature vector components
        bias = np.ones(len(X))[:, np.newaxis]
        X_phi = np.hstack((bias, X))
        y = model.test(sess, X_phi)
        return y.argmax(axis=1)
    plot_decision_boundary(func, x_phi, t_, x_phi[:, 0].min(), x_phi[:, 1].min(), x_phi[:, 0].max(), x_phi[:, 1].max(), 0.1) 
        
def test(args):
    assert args.output != None and args.load != None
    X = []
    Y = []

    # Load datasets without class
    datasets = args.X.split(',')
    for path in datasets:
        x = np.load(path).astype(np.float32)
        X.append(x)
        logging.info('Load %d data from %s' % (len(x), path))
    X = np.asarray(np.concatenate(X))

    model, std, phi = load(args)

    X_phi = preprocess_test(X, std, phi)

    logging.info('Use model %s with %d-dim (with bias) feautre space' % (args.model, X_phi.shape[1]))
    sess = None
    K = model.n_classes
    y = model.test(sess, X_phi)
    I = np.identity(K)
    y_one_hot = np.zeros([len(X), model.n_classes], dtype=np.int32)

    logging.info('Converting to one-hot ...')
    for n in tqdm(xrange(len(X))):
        y_one_hot[n, :] = I[y[n, :].argmax()]

    logging.info('Writing result to %s ...' % args.output)
    with open(args.output, 'w') as csv_file:
        for yn in tqdm(y_one_hot):
            csv_file.write(','.join([str(yn_i) for yn_i in yn]) + '\n')

def train(args):
    X, Y, parts = load_dataset_with_class(args)
    num_samples = X.shape[0]
    num_classes = Y.shape[1]
 
    # Perform task
    if args.task == 'eval':
        assert args.load != None and args.basis != None and args.std != None
        model, std, phi = load(args)
        
        logging.info('Preprocessing %d data...' % len(X))
        X_phi = preprocess_test(X, std, phi)

        logging.info('Evaluating...')
        evaluate(args, model, X_phi, Y)
    else:    
        model = get_model(args)
        # Preprocess datasets
        logging.info('Preprocessing %d data...' % num_samples)
        X_phi, phi, std = preprocess(args, X, Y)

        # Partitioning datasets
        logging.info('Partitioning datasets...')
        X_phi_Train = []
        Y_Train = []
        X_phi_Test = []
        Y_Test = []
        
        if args.permu == 'balance':
            fracs = [float(f) for f in args.frac.split(',')]
            for k in xrange(num_classes):
                begin = parts[k, 0]
                end = begin + parts[k, 1]
                inds = range(begin, end)
                np.random.shuffle(inds)
                nk = parts[k, 1]
                frac = fracs[k]
                nk_train = int(frac * nk)
                
                X_phi_Train.append(X_phi[inds[:nk_train]])
                Y_Train.append(Y[inds[:nk_train]])
                X_phi_Test.append(X_phi[inds[nk_train:]])
                Y_Test.append(Y[inds[nk_train:]])

            X_phi_Train = np.concatenate(X_phi_Train)
            Y_Train = np.concatenate(Y_Train)    
            X_phi_Test = np.concatenate(X_phi_Test)    
            Y_Test = np.concatenate(Y_Test)
            
            # Shuffle training, testing set
            logging.info('Shuffling datasets...')
            n_train = len(X_phi_Train)
            n_test = len(X_phi_Test)
            inds_train = range(n_train)
            inds_test = range(n_test)
            np.random.shuffle(inds_train)
            np.random.shuffle(inds_test)

            X_phi_Train = X_phi_Train[inds_train]
            Y_Train = Y_Train[inds_train]
            X_phi_Test = X_phi_Test[inds_test]
            Y_Test = Y_Test[inds_test]
        else:
            FRAC = float(args.frac)
            n_train = int(float(num_samples) * FRAC)
            inds = range(num_samples)
            np.random.shuffle(inds)
            X_phi = X_phi[inds]
            Y = Y[inds]
            X_phi_Train = X_phi[:n_train]
            Y_Train = Y[:n_train]
            X_phi_Test = X_phi[n_train:]
            Y_Test = Y[n_train:]

        logging.info('Training/Testing = %d/%d' % (len(Y_Train), len(Y_Test)))
        for k in xrange(num_classes):
            logging.info('# class-%d = %d' % (k, Y_Train[Y_Train.argmax(axis=1) == k].shape[0]))

        logging.info('Use model %s with %d-dim (with bias) feautre space' % (args.model, X_phi.shape[1]))
        sess = None
        logging.info('Training...')
        model.fit(sess, X_phi_Train, Y_Train)        

        if args.task == 'validate':
            logging.info('Evaluating testing accuracy...')
            evaluate(args, model, X_phi_Test, Y_Test)
        elif args.task == 'train':
            logging.info('Evaluating training accuracy...')
            evaluate(args, model, X_phi, Y)

            logging.info('Save model to %s' % args.output)

            assert args.output != None

            output_path = args.output
            # Save model
            model.save(output_path)
            
            # Save basis
            pickle.dump(phi, open(output_path + '.basis', "wb"), True) 

            # Save std
            np.save(output_path + '_std', std)
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', help='train/test task', 
            choices=['train', 'test', 'validate', 'eval', 'plot'], type=str, default='validate')
    parser.add_argument('--X', help='data (ordered)', type=str)
    parser.add_argument('--load', help='pre-trained model path', type=str, default=None)
    parser.add_argument('--basis', help='pre-trained model basis path', type=str, default=None)
    parser.add_argument('--std', help='pre-trained model stddev path', type=str, default=None)
    parser.add_argument('--output', help='model output', type=str, default=None)
    parser.add_argument('--model', help='gen/dis model', 
            choices=['gen', 'dis', 'nn'], type=str, default='dis')
    parser.add_argument('--pre', help='gen/dis model', 
            choices=['pca', 'hist', 'lda'], type=str, default='pca')
    parser.add_argument('--permu', help='train/test task', 
            choices=['unbalance', 'balance'], type=str, default='unbalance')
    parser.add_argument('--d', help='pca dimension', type=int, default=2)
    parser.add_argument('--h', help='number of nn hidden layers', type=int, default=1)
    parser.add_argument('--h_d', help='number of node in nn hidden layer', type=str, default="32")
    parser.add_argument('--frac', help='fraction of training set', type=str, default='0.8')
    parser.add_argument('--activations', help='activation functions', type=str, default='sigmoid')
    parser.add_argument('--tolerance', help='tolerance of error rate', type=float, default=0.01)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.01)
    parser.add_argument('--epoch', help='epoch', type=int, default=20)
    parser.add_argument('--batch_size', help='batch size', type=int, default=64)
    args = parser.parse_args()

    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    if args.task == 'test':
        test(args)
    elif args.task == 'plot':
        plot(args)
    else:
        train(args)
