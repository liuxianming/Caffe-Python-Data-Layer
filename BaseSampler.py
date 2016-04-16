"""Copyright @ Xianming Liu, University of Illinois at Urbana-Champaign
Base Sampler class definintion

Implement __init__, build_index, sample() functions
"""

from utils.util import parse_label

__author__ = ['Xianming Liu(liuxianming@gmail.com)']


class BaseSampler(object):
    def __init__(self, labels):
        """self._funcdict is preserved to customrized sampling functions
        """
        self._labels = labels
        self._funcdict = dict()
        self._build_index()

    def _build_index(self):
        """Build Index to randomly fetch samples from data

        The index is in the format of python dict
        {label: [list of sample id]}
        """
        self._sample_count = len(self._labels)
        self._index = dict()
        for id in range(self._sample_count):
            # parse label and insert into self._index
            labels_ = parse_label(self._labels[id])
            for label_ in labels_:
                if label_ in self._index.keys():
                    self._index[label_].append(id)
                else:
                    self._index[label_] = [id]

    def sample(self):
        """Function to run sampling

        Give a triplet of sample ids and corresponding distance

        The type of sampling function called is determined by
        self._sampling_type
        """
        self._iteration += 1
        return self._funcdict[self._sampling_type]()
