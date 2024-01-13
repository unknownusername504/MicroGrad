class Tensor:
    def __init__(self, shape, value=None, requires_grad=False):
        self.shape = shape
        self.value = value
        self.requires_grad = requires_grad
        self.grad = None
        self.grad_fn = None
        self.children = {}
        self.parents = {}
