class LayerBase:
    def __call__(self, x):
        return self.forward(x)

    def get_parameters(self):
        raise Exception("You need to implement get_parameters method") 

    def forward(self, x):
        raise Exception("You need to implement forward method")
