""" Copyright @ Xianming Liu, University of Illinois at Urbana-Champaign

Class implementations for various DataManager

DataManager takes charge of data input from files,
including: LMDB, ImageList, csv file, BCF file (either memory or file mode)
and etc.

All DataManager uses load_all() function to get data and labels,
return Data and Labels, which Data is the raw format(no decompression)

For Multiple Labels problem:
self._labels is a list of string =
":".join([str(a) for a in label_.astype(int)])
labels are separated by :
"""

from bcfstore import (bcf_store_file, bcf_store_memory)
import sys
import pandas as pd
import numpy as np
import lmdb
import os.path
from exceptions import Exception
import time
import caffe
from caffe.io import caffe_pb2

__authors__ = ['Xianming Liu (liuxianming@gmail.com)']


class BCFDataManager():
    """BCFDataManager

    Read and process data from bcf file
    param in format of yaml object

    Data file: param['source']
    Label file: param['labels']

    bcf_mode: param['bcf_mode'], (FILE or MEM) default = FILE
    """
    def __init__(self, param):
        self._source_fn = param.get('source')
        self._label_fn = param.get('labels')
        # bcf_mode: either FILE or MEM, default=FILE
        self._bcf_mode = param.get('bcf_mode', 'FILE')
        if not os.path.isfile(self._source_fn) or \
           not os.path.isfile(self._label_fn):
            raise Exception("Either Source of Label file does not exist")
        else:
            if self._bcf_mode == 'MEM':
                self._bcf = bcf_store_memory(self._source_fn)
            elif self._bcf_mode == 'FILE':
                self._bcf = bcf_store_file(self._source_fn)
        self._data = []
        self._labels = []

    def load_all(self):
        """The function to load all data and labels

        Give:
        data: the list of raw data, needs to be decompressed
              (e.g., raw JPEG string)
        labels: numpy array, with each element is a string
        """
        start = time.time()
        print("Start Loading Data from BCF {}".format(
            'MEMORY' if self._bcf_mode == 'MEM' else 'FILE'))

        self._labels = np.loadtxt(self._label_fn).astype(str)

        if self._bcf.size() != self._labels.shape[0]:
            raise Exception("Number of samples in data"
                            "and labels are not equal")
        else:
            for idx in range(self._bcf.size()):
                datum_str = self._bcf.get(idx)
                self._data.append(datum_str)
        end = time.time()
        print("Loading {} samples Done: Time cost {} seconds".format(
            len(self._data), end - start))

        return self._data, self._labels


class CSVDataManager():
    """CSVDataManager

    Read and process image list / labels from csv file
    param in format of yaml object

    Data and Label file: param['source']
    [optional] root = param['root'], default None
    """
    def __init__(self, param):
        self._source_fn = param.get('source')
        self._root = param.get('root', None)
        self._header = param.get('header', None)
        if not os.path.isfile(self._source_fn):
            raise Exception("Source file does not exist")
        self._data = []
        self._labels = []

    def load_all(self):
        """The function to load all data and labels

        Give:
        data: the list of raw data, needs to be decompressed
              (e.g., raw JPEG string)
        labels: numpy array of string, to support multiple label
        """
        start = time.time()
        print("Start Loading Data from CSV File {}".format(
            self._source_fn))
        # split csv using both space, tab, or comma
        sep = '[\s,]+'
        try:
            df = pd.read_csv(self._source_fn, sep=sep, engine='python',
                             header=self._header)
            print("Totally {} rows loaded to parse...".format(
                len(df.index)
            ))
            # parse df to get image file name and label
            for ln in df.iterrows():
                # for each row, the first column is file name, then labels
                fn_ = ln[1][0]
                if self._root:
                    fn_ = os.path.join(self._root, fn_)
                if not os.path.exists(fn_):
                    print("File {} does not exist, skip".format(fn_))
                    continue
                # read labels: the first column is image file name
                # and others are labels (one or more)
                label_ = ln[1][1:].values
                if len(label_) == 1:
                    label_ = label_[0]
                else:
                    label_ = ":".join([str(x) for x in label_.astype(int)])
                self._labels.append(str(label_))
                # open file as binary and read in
                with open(fn_, 'rb') as image_fp:
                    datum_str_ = image_fp.read()
                    self._data.append(datum_str_)
        except:
            print sys.exc_info()[1], fn_
            raise Exception("Error in Parsing input file")
        end = time.time()
        self._labels = np.array(self._labels)
        print("Loading {} samples Done: Time cost {} seconds".format(
            len(self._data), end - start))

        return self._data, self._labels


class LMDBDataManager():
    """LMDBDataManager

    Read and process images / labels from LMDB database
    param in format of yaml object

    Data and Label file: param['source']
    if there param['label'], then read labels from a separate lmdb
    """
    def __init__(self, param):
        self._source_fn = param.get('source')
        if not os.path.isfile(self._source_fn):
            raise Exception("Source file does not exist")
        self._label_fn = param.get('labels', None)
        self._data = []
        self._labels = []

    def load_all(self):
        """The function to load all data and labels

        Give:
        data: the list of raw data, needs to be decompressed
              (e.g., raw JPEG string)
        labels: 0-based labels, in format of numpy array
        """
        start = time.time()
        print("Start Loading Data from CSV File {}".format(
            self._source_fn))
        try:
            db_ = lmdb.open(self._source_fn)
            data_cursor_ = db_.begin().cursor()
            if self._label_fn:
                label_db_ = lmdb.open(self._label_fn)
                label_cursor_ = label_db_.begin().cursor()
            # begin reading data
            if self._label_fn:
                label_cursor_.first()
            while data_cursor_.next():
                value_str = data_cursor_.value()
                datum_ = caffe_pb2.Datum()
                datum_.ParseFromString(value_str)
                self._data.append(datum_.data)
                if self._label_fn:
                    label_cursor_.next()
                    label_datum_ = caffe_pb2.Datum()
                    label_datum_.ParseFromString(label_cursor_.value())
                    label_ = caffe.io.datum_to_array(label_datum_)
                    label_ = ":".join([str(x) for x in label_.astype(int)])
                else:
                    label_ = str(datum_.label)
                self._labels.appen(label_)
            # close all db
            db_.close()
            if self._label_fn:
                label_db_.close()
        except:
            raise Exception("Error in Parsing input file")
        end = time.time()
        self._labels = np.array(self._labels)
        print("Loading {} samples Done: Time cost {} seconds".format(
            len(self._data), end - start))

        return self._data, self._labels
