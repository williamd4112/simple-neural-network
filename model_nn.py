import numpy as np

from model_util import *

def init_weight_zeros(shape):
    return np.zeros(shape, dtype=np.float32)

class Layer(object):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def _forward(self, in_):
        raise NotImplemented()

    def _backward(self, out_):
        raise NotImplemented()

class FullyConnectedLayer(Layer):
    def __init__(self, input_dim, output_dim, activation='sigmoid', weight_init='zero'):
        super(FullyConnectedLayer, self).__init__(self, (input_dim,), (output_dim,))

        # TODO: Design more weight init
        self.w = init_weight_zeros([input_dim, output_dim])

        # TODO: Design other activation function
        self.h = sigmoid
        self.h_d = d_sigmoid

        self.error = None
        self.gradient = None

    def _forward(self, in_):
        a = in_.dot(self.w)
        self.z = self.h(a)
        return self.z
    
    def _backward(self, in_, error_, z_, z_d):
        # Compute gradient
        self.gradient = error_.T.dot(z_)

        # Compute error
        self.error = z_d * (error_.dot(self.w))

        

    
