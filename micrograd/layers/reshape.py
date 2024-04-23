from micrograd.layers.layer import Layer


class Reshape(Layer):
    def __init__(self, shape):
        self.shape = shape

        super().__init__()

    def __call__(self, x):
        return x.reshape(self.shape)
