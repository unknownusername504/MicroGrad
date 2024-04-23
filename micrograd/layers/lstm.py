from micrograd.layers.layer import Layer
from micrograd.tensors.tensor import Tensor
import numpy as np


class LSTM(Layer):
    def __init__(
        self,
        num_features,
        hidden_size,
        output_dim,
        activation,
        bias=True,
    ):
        self.num_features = num_features
        self.activation = activation
        self.hidden_size = hidden_size
        self.bias = bias

        # Initialize weights and biases for the LSTM cell
        self.num_gates = 4
        self.num_layers = 1
        # Have to transpose the weights to match the input shape and perform auto grad
        self.w_ih = Tensor(
            np.random.randn(num_features, self.num_gates * hidden_size),
            requires_grad=True,
        )
        self.w_hh = Tensor(
            np.random.randn(hidden_size, self.num_gates * hidden_size),
            requires_grad=True,
        )
        if bias:
            self.b_ih = (
                Tensor(np.zeros((self.num_gates * hidden_size,)), requires_grad=True)
                if bias
                else None
            )
            self.b_hh = (
                Tensor(np.zeros((self.num_gates * hidden_size,)), requires_grad=True)
                if bias
                else None
            )

        # Initialize the weights and biases for the output layer
        self.w_ho = Tensor(np.random.randn(hidden_size, output_dim), requires_grad=True)
        if bias:
            self.b_ho = (
                Tensor(np.zeros((output_dim,)), requires_grad=True) if bias else None
            )

        super().__init__()
        self.parameters = [
            self.w_ih,
            self.w_hh,
            self.w_ho,
        ]
        if bias:
            self.parameters.extend([self.b_ih, self.b_hh, self.b_ho])

        self.h_prev = Tensor(np.zeros((self.num_layers, self.hidden_size)))
        self.c_prev = Tensor(np.zeros((self.num_features, self.hidden_size)))

    def __call__(self, x):
        # LSTM cell computations
        gi = x @ self.w_ih
        gh = self.h_prev @ self.w_hh
        if self.bias:
            gi += self.b_ih
            gh += self.b_hh

        i_f, i_i, i_c, i_o = Tensor.split(gi, self.num_gates, axis=1)
        h_f, h_i, h_c, h_o = Tensor.split(gh, self.num_gates, axis=1)

        # TODO: Replace np.tanh with function Tanh
        forgetgate = self.activation((i_f + h_f))()
        ingate = self.activation((i_i + h_i))()
        cellgate = (forgetgate * self.c_prev) + (
            ingate * Tensor(np.tanh((i_c + h_c).value))
        )
        self.c_prev = cellgate
        outgate = self.activation((i_o + h_o))()

        self.h_prev = outgate * Tensor(np.tanh(cellgate.value))

        # Output layer computations
        y_pred = self.h_prev @ self.w_ho
        if self.bias:
            y_pred += self.b_ho

        return y_pred
