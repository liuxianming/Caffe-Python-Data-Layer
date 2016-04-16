"""Copyright@Xianming Liu, University of Illinois at Urbana, Champaign

Implementation of Triplet Data Layer


"""
import atexit
import numpy as np
from BasePythonDataLayer import BasePythonDataLayer
from multiprocessing import (Process, Pipe)
from utils.SampleIO import extract_sample
from TripletSampler import TripletSampler

__authors__ = ['Xianming Liu(liuxianming@gmail.com)']


class TripletDataLayer(BasePythonDataLayer):
    """Triplet Data Layer:
    Provide data batches for Triplet Network (using ranking loss)

    Data: 3 * batch_size * channels * width * height
          anchor image, positive and negative ones
    Label: Relative Similarity (Optional)

    Implemenation is based on BasePythonDataLayer,
    need to implement:
    1. get_next_minibatch(self) function
    2. sampleing functions for randomly sampling and guided sampling
    """

    def setup(self, bottom, top):
        # setup functions from super class
        super(TripletDataLayer, self).setup(bottom, top)
        print("Using Triplet Python Data Layer")
        # prefetch or not: default = False
        self._sampling_type = self._layer_params.get('type', 'RANDOM')
        self._prefetch = self._layer_params.get('prefetch', False)
        """Construct kwargs:
        possible fields:
        k - number of candidates when hard negative sampling
        m - similarity graph filename for hard negative sampling
        n - number of iterations before hard negative sampling
        """
        kwargs = {}
        for key, value in self._layer_params.iteritems():
            if key.lower() in ['k', 'm', 'n']:
                kwargs[key.lower()] = value
        if self._prefetch:
            # using prefetch to generate mini-batches
            self._conn, conn = Pipe()
            self._prefetch_process = TripletPrefetcher(
                conn,
                self._label, self._data,
                self._mean, self._resize, self._batch_size,
                self._sampling_type, **kwargs
            )
            print("Start Prefetching Process...")
            self._prefetch_process.start()

            def cleanup():
                print("Terminating Prefetching Processs...")
                self._prefetch_process.terminate()
                self._prefetch_process.join()
                self._conn.close()
            atexit.register(cleanup)
        else:
            self._sampler = TripletSampler(
                self._sampling_type, self._label, **kwargs)
        self.reshape(bottom, top)

    def get_a_datum(self):
        """Get a datum:

        Sampling -> decode images -> stack numpy array
        """
        sample = self._sampler.sample()
        if self._compressed:
            datum_ = [
                extract_sample(self._data[id], self._mean, self._resize) for
                id in sample[:3]]
        else:
            datum_ = [self._data[id] for id in sample[:3]]
        if len(sample) == 4:
            datum_.append(sample[-1])
        return datum_

    def get_next_minibatch(self):
        if self._prefetch:
            # get mini-batch from prefetcher
            batch = self._conn.recv()
        else:
            # generate using in-thread functions
            data = []
            p_data = []
            n_data = []
            label = []
            for i in range(self._batch_size):
                datum_ = self.get_a_datum()
                data.append(datum_[0])
                p_data.append(datum_[1])
                n_data.append(datum_[2])
                if len(datum_) == 4:
                    # datum and label / margin
                    label.append(datum_[-1])
            batch = [np.array(data),
                     np.array(p_data),
                     np.array(n_data)]
            if len(label):
                label = np.array(label).reshape(self._batch_size, 1, 1, 1)
                batch.append(label)
        return batch


class TripletPrefetcher(Process):
    """TripletPrefetcher:

    Use a separate process to sample triplets,
    following the same function implementations as TripletDataLayer
    """
    def __init__(self, conn, labels, data,
                 mean, resize, batch_size,
                 # samping related parameters
                 sampling_type, **kwargs):
        super(TripletPrefetcher, self).__init__()
        self._conn = conn
        self._labels = labels
        self._data = data
        if type(self._data[0]) is not str:
            self._compressed = False
        else:
            self._compressed = True

        self._batch_size = batch_size
        self._mean = mean
        self._resize = resize
        self._sampling_type = sampling_type
        # kwargs is a dictionary related with sampling
        self._sampler = TripletSampler(
            self._sampling_type, self._labels, **kwargs)

    def type(self):
        return "TripletPrefetcher"

    def get_a_datum(self):
        """Get a datum:

        Sampling -> decode images -> stack numpy array
        """
        sample = self._sampler.sample()
        if self._compressed:
            datum_ = [
                extract_sample(self._data[id], self._mean, self._resize) for
                id in sample[:3]]
        else:
            datum_ = [self._data[id] for id in sample[:3]]
        if len(sample) == 4:
            datum_.append(sample[-1])
        return datum_

    def get_next_minibatch(self):
        # generate using in-thread functions
        data = []
        p_data = []
        n_data = []
        label = []
        for i in range(self._batch_size):
            datum_ = self.get_a_datum()
            # print(len(datum_), ":".join([str(x.shape) for x in datum_]))
            data.append(datum_[0])
            p_data.append(datum_[1])
            n_data.append(datum_[2])
            if len(datum_) == 4:
                # datum and label / margin
                label.append(datum_[-1])
        batch = [np.array(data),
                 np.array(p_data),
                 np.array(n_data)]
        if len(label):
            label = np.array(label).reshape(self._batch_size, 1, 1, 1)
            batch.append(label)
        return batch

    def run(self):
        print("Prefetcher Started...")
        while True:
            batch = self.get_next_minibatch()
            self._conn.send(batch)
