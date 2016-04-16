"""Copyright Xianming Liu, University of Illinois at Urbana-Champain

Utils functions for python data layers,
including parsing strings, sampling, etc
"""

import numpy as np

__authors__ = ['Xianming Liu (liuxianming@gmail.com)']


def parse_label(labels):
    """Parse label strings into numpy array of labels"""
    return [int(x) for x in str(labels).split(':')]


def intersect_sim(array_1, array_2):
    """Calculate the simiarity of two arrays
    by using intersection / union
    """
    sim = float(np.intersect1d(array_1, array_2).size) / \
        float(np.union1d(array_1, array_2).size)
    return sim
