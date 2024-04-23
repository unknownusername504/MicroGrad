from micrograd.layers.layer import Layer
from micrograd.tensors.tensor import Tensor
import numpy as np


class Linear(Layer):
    def __init__(self, input_dim, output_dim, activation, bias=True):
        self.w = Tensor(np.random.randn(input_dim, output_dim), requires_grad=True)
        self.bias = bias
        if bias:
            self.b = Tensor(np.random.randn(output_dim), requires_grad=True)
        self.activation = activation

        super().__init__()
        self.parameters = [self.w, self.b]

    def __call__(self, x):
        calc = x @ self.w
        if self.bias:
            calc += self.b
        return self.activation(calc)()
