"""
This file contains a class allowing to use a context where gradient is not computed
"""

import flamb


class no_grad:
    """
    Context that disables gradient computation
    """

    def __init__(self):
        pass

    def __enter__(self):
        flamb.environ["grad_enabled"] = False

    def __exit__(self, type, value, traceback):
        flamb.environ["grad_enabled"] = True
