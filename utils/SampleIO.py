"""Utility functions of Sample I/O for python data layer
"""

import sys
import os.path
import numpy as np
import scipy.misc
from cStringIO import StringIO

__author__ = ['Xianming Liu(liuxianming@gmail.com']


def extract_sample(img, image_mean=None, resize=-1):
    """Extract image content from image string or from file
    TAKE:
    input - either file content as string or numpy array
    image_mean - numpy array of image mean or a values of size (1,3)
    resize - to resize image, set resize > 0; otherwise, don't resize
    """
    try:
        # if input is a file name, then read image; otherwise decode_imgstr
        if type(img) is np.ndarray:
            img_data = img
        else:
            img_data = decode_imgstr(img)
        if type(resize) in [tuple, list]:
            # resize in two dimensions
            img_data = scipy.misc.imresize(img_data, (resize[0], resize[1]))
        elif resize > 0:
            img_data = scipy.misc.imresize(img_data, (resize, resize))
        img_data = img_data.astype(np.float32, copy=False)
        img_data = img_data[:, :, ::-1]
        # change channel for caffe:
        img_data = img_data.transpose(2, 0, 1)  # to CxHxW
        # substract_mean
        if image_mean is not None:
            img_data = substract_mean(img_data, image_mean)
        return img_data
    except:
        print sys.exc_info()[0], sys.exc_info()[1]
        return


def decode_imgstr(imgstr):
    img_data = scipy.misc.imread(StringIO(imgstr))
    return img_data


def substract_mean(img, image_mean):
    """Substract image mean from data sample

    image_mean is a numpy array,
    either 1 * 3 or of the same size as input image
    """
    if image_mean.ndim == 1:
        image_mean = image_mean[:, np.newaxis, np.newaxis]
    img -= image_mean
    return img
