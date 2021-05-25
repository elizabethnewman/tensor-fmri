import numpy as np


def rescale(x, low=0, high=1):

    # normalize between 0 and 1
    x -= x.min()
    x /= (x.max() - x.min())

    # shift between low and high range
    x *= high - low
    x += low

    return x


def prod(a):
    n = 1
    for ai in a:
        n *= ai
    return n
