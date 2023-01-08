import flamb

class no_grad:
    def __init__(self):
        pass

    def __enter__(self):
        flamb.environ['compute_grad'] = False

    def __exit__(self, type, value, traceback):
        flamb.environ['compute_grad'] = True
