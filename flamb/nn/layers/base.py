class LayerBase:
    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        raise Exception("You need to implement forward method")
