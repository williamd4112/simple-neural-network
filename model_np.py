import numpy as np
import logging
import random

from model_util import *

class ClassificationModel(object):
    def __init__(self, optimizer, epochs=None, batch_size=None, tolerance=None):
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.tolerance = tolerance

    def eval(self, sess, x_, t_):
        return float(np.equal(self.test(sess, x_).argmax(axis=1), t_.argmax(axis=1)).sum()) / len(t_)

    def fit(self, sess, x_, t_):
        # Initialize model fitting parameters
        self._init_model_fit_parameter(sess, x_, t_)

        # Initialize weight for each class
        self._init_weight()

        if self.optimizer == 'seq':
            self._fit_sequential(sess, x_, t_, self.tolerance, self.epochs, self.batch_size)
        else:
            self._fit(sess, x_, t_)
    
    def _init_model_fit_parameter(self, sess, x_, t_):
        self.n_classes = t_.shape[1]
        self.n_features = x_.shape[1]

    def _fit(self, sess, x_, t_):
        self._optimize(sess, x_, t_)
        acc = self.eval(sess, x_, t_)
        logging.info('Training accuracy = %f, error rate = %f' % (acc, 1.0 - acc))

    def _fit_sequential(self, sess, x_, t_, tolerance, epochs, batch_size):
        assert len(x_.shape) == 2

        N = len(x_)
        D = self.n_features
        K = self.n_classes

        # Iterative optimize
        indices = range(N)
        for epoch in xrange(epochs):
            np.random.shuffle(indices)
            for begin in xrange(0, N, batch_size):
                end = min(N, begin + batch_size)
                x_train = x_[indices[begin:end]]
                t_train = t_[indices[begin:end]]
                self._optimize(sess, x_train, t_train)
            acc = self.eval(sess, x_, t_)
            logging.info('Epoch %d Training accuracy = %f, error rate = %f' % (epoch, acc, 1.0 - acc))

            if 1.0 - acc <= tolerance:
                logging.info('Target error rate reached.')
                break

    def test(self, sess, x_):
        raise NotImplementedError()         

    def _init_weight(self):
        raise NotImplementedError()         

    def _optimize(self, sess, x_, t_):
        raise NotImplementedError()         
   

class LinearClassificationModel(ClassificationModel):
    def __init__(self, optimizer, epochs=None, batch_size=None, tolerance=None):
        super(ClassificationModel, self).__init__(optimizer, epochs, batch_size, tolerance)
        self.w = None

    def save(self, path):
        np.save(path, self.w)
        logging.info('Saving model to %s success [K = %d, M = %d]' % (path, self.w.shape[0], self.w.shape[1]))

    def load(self, path):
        self.w = np.load(path)
        self.n_classes = self.w.shape[0]
        self.n_features = self.w.shape[1]
        logging.info('Loading model from %s success [K = %d, M = %d]' % (path, self.n_classes, self.n_features))
    
    def test(self, sess, x_):
        return softmax(x_.dot(self.w.T))

    def eval(self, sess, x_, t_):
        return float(np.equal(self.test(sess, x_).argmax(axis=1), t_.argmax(axis=1)).sum()) / len(t_)

    def _init_weight(self):
        self.w = np.zeros([self.n_classes, self.n_features])

class ProbabilisticDiscriminativeModel(LinearClassificationModel):
    def __init__(self, lr=0.01, epochs=20, batch_size=128, tolerance=0.01):
        super(ProbabilisticDiscriminativeModel, self).__init__('seq', epochs, batch_size, tolerance)
        self.lr = lr

    def _optimize(self, sess, x_, t_):
        N, M = x_.shape
        K = self.n_classes
        
        x = np.asarray(x_)
        y = np.asarray(self.test(sess, x_))
        t = np.asarray(t_)

        # Calculate gradient
        grad = np.zeros([K, M])
        for j in xrange(K):
            for n in xrange(N):
                grad[j, :] += (y[n,j] - t[n,j]) * x[n]
        grad = grad.flatten()

        # Calculate Hessian matrix
        I = np.identity(K)
        H = np.zeros([K*M, K*M])
        for j in xrange(K):
            for k in xrange(K):
                D_wjk = np.zeros([M, M])
                for n in xrange(N):
                    D_wjk += y[n,k] * (I[k,j] - y[n,j]) * x.T.dot(x)
                H[j*(M):(j+1)*M, (k)*M:(k+1)*M] = D_wjk
        try:
            H_inv = np.linalg.pinv(H)
        except np.linalg.linalg.LinAlgError:
            return

        # Update weight
        w_old = self.w.flatten()
        w_new = w_old - self.lr * H_inv.dot(grad)
        w_new = w_new.reshape([K, M])
        self.w = w_new

class ProbabilisticGenerativeModel(LinearClassificationModel):
    def __init__(self):
        super(ProbabilisticGenerativeModel, self).__init__(optimizer='once')

    def _optimize(self, sess, x_, t_):
        N, M = x_.shape
        K = self.n_classes

        priors = np.zeros([K, 1])
        means = np.zeros([K, M-1])
        sigma = np.zeros([M-1, M-1])
    
        for k in xrange(K):
            x_k = x_[t_[:, k] == 1][:, 1:]
            n_k = float(len(x_k))
            priors[k] = n_k / N
            means[k] = np.mean(x_k, axis=0)
            sigma += priors[k] * ((x_k - means[k]).T.dot(x_k - means[k])) / n_k
        sigma_inv = np.asarray(np.linalg.pinv(sigma))
        
        for k in xrange(K):
            self.w[k, 1:] = (sigma_inv.dot(means[k]))
            self.w[k, 0] = (-1.0 / 2) * means[k].T.dot(sigma_inv.dot(means[k])) + np.log(priors[k])

class MultiLayerPerceptron(ClassificationModel):
    def __init__(self, n_hidden_layers=1, n_hidden_units=[32], hidden_activations=['sigmoid'], lr=0.01, epochs=10, batch_size=16, tolerance=0.0):
        super(MultiLayerPerceptron, self).__init__('seq', epochs, batch_size, tolerance)

        assert(n_hidden_layers == len(n_hidden_units))

        self.n_layers = n_hidden_layers + 1
        self.n_hidden_units = n_hidden_units
        self.lr = lr
        self.activations = self._get_activations(hidden_activations)
        
        # Forward result z
        self.layer_forward_z = [None] * (self.n_layers)

        # Forward result a
        self.layer_forward_a = [None] * (self.n_layers)

        # Backward error
        self.layer_backward_errors = [None] * (self.n_layers)

        # Backward derivative
        self.layer_backward_derivatives = [None] * (self.n_layers)
    
    def _get_activations(self, activations):
        acts = []
        for act in activations:
            if act == 'sigmoid':
                acts.append(Sigmoid())
            elif act == 'softmax':
                acts.append(Softmax())
            else:
                raise NotImplementedError()
        acts = acts + [Softmax()]
        return acts

    def _init_model_fit_parameter(self, sess, x_, t_):
        super(MultiLayerPerceptron, self)._init_model_fit_parameter(sess, x_, t_)

        # Assign units of all layers
        self.n_units = self.n_hidden_units + [self.n_classes]

    def test(self, sess, x_):
        return self._forward(sess, x_)

    def _init_layer_weight(self, shape):
        return np.random.normal(0.0, 1e-4, shape)

    def _init_weight(self):
        w = []
        n_input_unit = self.n_features
        for i in xrange(len(self.n_units)):
            n_output_unit = self.n_units[i]
            w.append(self._init_layer_weight([n_input_unit, n_output_unit]))
            n_input_unit = n_output_unit 
        self.w = w

    def _forward(self, sess, x_):
        L = self.n_layers

        input_unit = x_
        for l in xrange(self.n_layers):
            w = self.w[l]
            a = input_unit.dot(w)
            z = self.activations[l](a)
            input_unit = z
            self.layer_forward_z[l] = z
            self.layer_forward_a[l] = a
        return self.layer_forward_z[L - 1]
    
    def _backward(self, sess, x_, t_):
        L = self.n_layers

        # Calculate errors
        self.layer_backward_errors[L - 1] = self.layer_forward_z[L - 1] - t_
        for l in reversed(xrange(0, L - 1)):
            z = self.activations[l].d(self.layer_forward_a[l])
            w = self.w[l + 1]
            error = self.layer_backward_errors[l + 1]
            self.layer_backward_errors[l] = z * error.dot(w.T)

        # Calculate derivative
        for l in reversed(xrange(1, L)):
            error = self.layer_backward_errors[l]
            z = self.layer_forward_z[l - 1]
            self.layer_backward_derivatives[l] = z.T.dot(error)
        self.layer_backward_derivatives[0] = x_.T.dot(self.layer_backward_errors[0])


    def _optimize(self, sess, x_, t_):
        L = self.n_layers    

        self._forward(sess, x_)
        self._backward(sess, x_, t_)
        
        for l in xrange(L):
            self.w[l] = self.w[l] - self.lr * self.layer_backward_derivatives[l]

