from micrograd.layers.layer import Layer


class Flatten(Layer):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return x.flatten()
