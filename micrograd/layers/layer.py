class Layer:
    def __init__(self):
        self.parameters = []

    def __call__(self, _):
        raise NotImplementedError

    def get_trainable_params(self):
        return self.parameters
