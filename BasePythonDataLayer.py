"""Copyright @ Xianming Liu, University of Illinois at Urbana-Champaign

Implementation of BasePythonDataLayer

Basic operation is to load all compressed version of images into memory
and decomress a particular image when access
Input of the layer is maintained by DataManager Class Instance
"""
import caffe
from caffe.io import caffe_pb2
import numpy as np
import yaml
import random
from utils.DataManager import (BCFDataManager,
                               CSVDataManager,
                               LMDBDataManager)
from utils.SampleIO import extract_sample

__authors__ = ['Xianming Liu (liuxianming@gmail.com)']


class BasePythonDataLayer(caffe.Layer):
    """Base Class for all python data layers

    Data are stored in self._data,
    and labels are stored at self._label

    It first read a compressed / non-compressed datum from self._data,
    then decompress / preprocess into the data used for caffe

    preload_db() relies on the implemenation of DataManager,
    it handles various types of data source including BCF, CSV file, and LMDB

    There are following functions to implement:
    1. type(self)
    2. get_next_minibatch(self)

    Before implementation of each class, call corresponding method from super:
    super(CLASSNAME, self).preload_db() for example

    Private data type:
    self._data: list of compressed image binary strings
    self._label: numpy array of size (n_samples, 1),
                 to facilitate fast selection when sampling
    self._source_type: type of DataManager used for loading data,
          Including: CSV, BCF, LMDB
          plain text files could be parsed by CSVDataManager
    """

    def setup(self, bottom, top):
        layer_params = yaml.load(self.param_str)
        self._layer_params = layer_params
        # default batch_size = 256
        self._batch_size = int(layer_params.get('batch_size', 256))
        self._resize = layer_params.get('resize', -1)
        self._mean_file = layer_params.get('mean_file', None)
        self._source_type = layer_params.get('source_type', 'CSV')
        self._shuffle = layer_params.get('shuffle', False)
        # read image_mean from file and preload all data into memory
        # will read either file or array into self._mean
        self.set_mean()
        self.preload_db()
        self._compressed = self._layer_params.get('compressed', True)
        if not self._compressed:
            self.decompress_data()

    def decompress_data(self):
        print("Decompressing all data...")
        for i in range(self._sample_count):
            self._data[i] = extract_sample(
                self._data[i], self._mean, self._resize)

    def preload_db(self):
        """Read all images in and all labels

        Implemenation relies on DataManager Classes
        """
        print("Preloading Data...")
        if self._source_type == 'BCF':
            self._data_manager = BCFDataManager(self._layer_params)
        elif self._source_type == 'CSV':
            self._data_manager = CSVDataManager(self._layer_params)
        elif self._source_type == 'LMDB':
            self._data_manager = LMDBDataManager(self._layer_params)
        # read all data
        self._data, self._label = self._data_manager.load_all()
        self._sample_count = len(self._data)
        if self._shuffle:
            self.shuffle()

    def data(self):
        return self._data

    def labels(self):
        return self._label

    def set_mean(self):
        if self._mean_file:
            if type(self._mean_file) is str:
                # read image mean from file
                try:
                    # if it is a pickle file
                    self._mean = np.load(self._mean_file)
                except (IOError):
                    blob = caffe_pb2.BlobProto()
                    blob_str = open(self._mean_file, 'rb').read()
                    blob.ParseFromString(blob_str)
                    self._mean = np.array(caffe.io.blobproto_to_array(blob))[0]
            else:
                self._mean = self._mean_file
        else:
            self._mean = None

    def type(self):
        return "BasePythonDataLayer"

    def shuffle(self):
        """Shuffle all samples and their labels"""
        shuffled_data_ = list(zip(self._data, self._label))
        random.shuffle(shuffled_data_)
        self._data, self._label = zip(*shuffled_data_)
        self._data = list(self._data)
        self._label = list(self._label)

    def get_next_minibatch(self):
        """Generate next mini-batch

        The return value is array of numpy array: [data, label]
        Reshape funcion will be called based on resutls of this function

        Needs to implement in each class
        """
        pass

    def forward(self, bottom, top):
        blob = self.get_next_minibatch()
        for i in range(len(blob)):
            top[i].data[...] = blob[i].astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        pass

    def reshape(self, bottom, top):
        blob = self.get_next_minibatch()
        for i in range(len(blob)):
            top[i].reshape(*(blob[i].shape))
