"""Copyright @ Xianming Liu, University of Illinois at Urbana-Champaign

MultiLabelLayer, which deals with the multiple labels input,
Outputs include both data and label numpy arrays,
For the labels, the output will be a probabilistic array or binary array
"""

import numpy as np
from BasePythonDataLayer import BasePythonDataLayer
from utils.SampleIO import extract_sample
from utils.util import parse_label

__author__ = ['Xianming Liu(liuxianming@gmail.com)']


class MultiLabelLayer(BasePythonDataLayer):
    """Implementation of Data Layer for MultiLabel raw data
    """

    def setup(self, bottom, top):
        super(MultiLabelLayer, self).setup(bottom, top)
        self._multilabel = self._layer_params.get('multilabel', False)
        if not self._multilabel:
            self._label_dim = 1
        else:
            self._label_dim = self._layer_params.get('label dim', None)
            if not self._label_dim:
                # try to estimate the dimension of labels
                self.calculate_label_dim()
        self._cur = 0

    def calculate_label_dim(self):
        """Calculate the dimension of labels
        by calculating the lenth of label set
        """
        all_labels = []
        for label_str in self._label:
            label = parse_label(label_str)
            all_labels += label
        all_labels = set(all_labels)
        self._label_dim = len(all_labels)

    def get_a_datum(self):
        if self._compressed:
            datum = extract_sample(
                self._data[self._cur], self._mean, self._resize)
        else:
            datum = self._data[self._cur]
        # start parsing labels
        label_elems = parse_label(self._label[self._cur])
        label = np.zeros(self._label_dim)
        if not self._multilabel:
            label[0] = label_elems[0]
        else:
            for i in label_elems:
                label[i] = 1
        self._cur = (self._cur + 1) % self._sample_count
        return datum, label

    def get_next_minibatch(self):
        data = []
        labels = []
        for i in range(self._batch_size):
            datum, label = self.get_a_datum()
            data.append(datum)
            labels.append(label)
        batch = [
            np.array(data),
            np.array(labels).reshape(self._batch_size, self._label_dim, 1, 1)
        ]
        return batch
