"""Copyright @ Xianming Liu, University of Illinois at Urbana-Champaign

Implementation of TripletSampler Class based on BaseSampler
"""
from utils.util import (parse_label, intersect_sim)
from BaseSampler import BaseSampler
import numpy as np

__author__ = ['Xianming Liu(liuxianming@gmail.com']


class TripletSampler(BaseSampler):
    """Class TripletSampler

    Collection of sampling functions for TripletDataLayer
    output will be:
    (anchor, positive, negative, distance) or (a, p, n)

    Run function sample() to generate a triplet and the corresponding distance

    Type of sampling methods:
    1. random: Randomly sampling based on the assumption of
               single label each sample. Output will contain no similarity
               - RANDOM
    2. random_multilabel: Randomly sampling based on the assumption that
               each sample has multiple labels. Output will contain
               a similarity score (or distance of similarities) of the triplet
               - RANDOM_MULTILABEL
    3. Multiple Label Hard Negative Sampling: hard negative sampling based on
               the multiple labels assumption. Sample k negatives and then pick
               the one with largest distance (minimal label cooccurance) with
               the anchor sample.
               - HARD_MULTILABEL
    4. Hard Negative Mining based on pre-calculated similarity graph:
    (modified version of hard negative sampling from paper:
    Learning Fine-grained Image Similarity with Deep Ranking, CVPR 2014):
               Hard negative mining based on pre-calculated similarity graph.
               Need to pre-compute the graph offline and do sampling online.
               The input will be a adjacent matrix of all nodes, in order to
               load all the graph to memory.
               - HARD
    -
    """
    def __init__(self, sampling_type, labels, **kwargs):
        super(TripletSampler, self).__init__(labels)
        self._sampling_type = sampling_type.upper()
        """set other attributes

        including
            k - number of candidates to consider when hard negative mining
            m - precomputed similarity graph adjacant matrix
                (in the format of dict)
            n - the number of iterations before hard negative sampling.
                Run n iterations of randomly sampling then do hard sampling
        """
        if kwargs:
            for key, value in kwargs.iteritems():
                if key.lower() in ['k', 'm', 'n']:
                    self.__setattr__('_{}'.format(key.lower()), value)
            print("Set attributes done")
        self._iteration = 0  # counter
        # define function dictionar
        self._funcdict = {
            'RANDOM': self.random_sampling,
            'RANDOM_MULTILABEL': self.random_multilabel,
            'HARD_MULTILABEL': self.hard_negative_multilabel,
            'HARD': self.hard_negative_graph
        }

    def random_sampling(self):
        """Random Sampling of Triplets
        """
        anchor_class_id, negative_class_id = np.random.choice(
            self._index.keys(), 2)
        anchor_id, positive_id = np.random.choice(
            self._index[anchor_class_id], 2)
        negative_id = np.random.choice(
            self._index[negative_class_id])
        return (anchor_id, positive_id, negative_id)

    def random_multilabel(self):
        """Random Sampling under the assumption of multilabels

        All are similar to random sampling
        the difference is to involve a distance ofsimilarity measurements
        in addition.
        Or equvilent, as a margin of anchor sample
        between positive and negative
        """
        anchor_id, positive_id, negative_id = self.random_sampling()
        # calculate the distance of similarity score / margin
        anchor_label = parse_label(self._labels[anchor_id])
        positive_label = parse_label(self._labels[positive_id])
        negative_label = parse_label(self._labels[negative_id])
        p_sim = intersect_sim(anchor_label, positive_label)
        n_sim = intersect_sim(anchor_label, negative_label)
        margin = p_sim - n_sim
        return (anchor_id, positive_id, negative_id, margin)

    def hard_negative_multilabel(self):
        """Hard Negative Sampling based on multilabel assumption

        Search the negative sample with largest distance (smallest sim)
        with the anchor within self._k negative samplels
        """
        # During early iterations of sampling, use random sampling instead
        if self._iteration <= self._n:
            return self.random_multilabel()

        anchor_class_id, negative_class_id = np.random.choice(
            self._index.keys(), 2)
        anchor_id, positive_id = np.random.choice(
            self._index[anchor_class_id], 2)
        negative_ids = np.random.choice(
            self._index[negative_class_id], self._k)
        # calcualte the smallest simlarity one with negatives
        anchor_label = parse_label(self._labels[anchor_id])
        positive_label = parse_label(self._labels[positive_id])
        negative_labels = [parse_label(self._labels[negative_id]) for
                           negative_id in negative_ids]
        p_sim = intersect_sim(anchor_label, positive_label)
        n_sims = np.array(
            [intersect_sim(anchor_label, negative_label) for
             negative_label in negative_labels])
        min_sim_id = np.argmin(n_sims)
        negative_id = negative_ids[min_sim_id]
        n_sim = n_sims[min_sim_id]
        margin = p_sim - n_sim
        return (anchor_id, positive_id, negative_id, margin)

    def hard_negative_graph(self):
        """Implementation of hard negative sampling based on pre-computed
        similarity graph
        Similar to implementations in paper
        Learning Fine-grained Image Similarity with Deep Ranking, CVPR 2014)

        Need to implement

        simiarity graph is given by self._m
        """
        pass
