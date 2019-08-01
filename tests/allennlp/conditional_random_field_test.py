# pylint: disable=no-self-use,invalid-name
import itertools
import math
import unittest

import numpy as np

from pytest import approx, raises
from tensorflow.python.keras import initializers, Sequential
from tensorflow.python.keras import layers
import tensorflow as tf

from tf_crf_layer.layer import CRF
from tf_crf_layer.crf_helper import allowed_transitions
from tf_crf_layer.exceptions import ConfigurationError
from tf_crf_layer.loss import crf_loss

from tests.MockedMasking import MockMasking


class TestConditionalRandomField(unittest.TestCase):
    def setUp(self):
        super().setUp()

        self.logits = np.array([
                [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
                [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
        ])
        self.tags = np.array([
                [2, 3, 4],
                [3, 2, 2]
        ])

        self.transitions = np.array([
                [0.1, 0.2, 0.3, 0.4, 0.5],
                [0.8, 0.3, 0.1, 0.7, 0.9],
                [-0.3, 2.1, -5.6, 3.4, 4.0],
                [0.2, 0.4, 0.6, -0.3, -0.4],
                [1.0, 1.0, 1.0, 1.0, 1.0]
        ])

        self.transitions_from_start = np.array([0.1, 0.2, 0.3, 0.4, 0.6])
        self.transitions_to_end = np.array([-0.1, -0.2, 0.3, -0.4, -0.4])

        # Use the CRF Module with fixed transitions to compute the log_likelihood
        self.crf = CRF(
            units=5,
            use_kernel=False,  # disable kernel transform
            chain_initializer=initializers.Constant(self.transitions),
            left_boundary_initializer=initializers.Constant(self.transitions_from_start),
            right_boundary_initializer=initializers.Constant(self.transitions_to_end)
        )

    def score(self, logits, tags):
        """
        Computes the likelihood score for the given sequence of tags,
        given the provided logits (and the transition weights in the CRF model)
        """
        # Start with transitions from START and to END
        total = self.transitions_from_start[tags[0]] + self.transitions_to_end[tags[-1]]
        # Add in all the intermediate transitions
        for tag, next_tag in zip(tags, tags[1:]):
            total += self.transitions[tag, next_tag]
        # Add in the logits for the observed tags
        for logit, tag in zip(logits, tags):
            total += logit[tag]
        return total

    # def test_forward_works_without_mask(self):
    #     log_likelihood = self.crf(self.logits, self.tags).item()
    #
    #     # Now compute the log-likelihood manually
    #     manual_log_likelihood = 0.0
    #
    #     # For each instance, manually compute the numerator
    #     # (which is just the score for the logits and actual tags)
    #     # and the denominator
    #     # (which is the log-sum-exp of the scores for the logits across all possible tags)
    #     for logits_i, tags_i in zip(self.logits, self.tags):
    #         numerator = self.score(logits_i.detach(), tags_i.detach())
    #         all_scores = [self.score(logits_i.detach(), tags_j)
    #                       for tags_j in itertools.product(range(5), repeat=3)]
    #         denominator = math.log(sum(math.exp(score) for score in all_scores))
    #         # And include them in the manual calculation.
    #         manual_log_likelihood += numerator - denominator
    #
    #     # The manually computed log likelihood should equal the result of crf.forward.
    #     assert manual_log_likelihood.item() == approx(log_likelihood)

    def test_forward_works_without_mask(self):
        model = Sequential()
        model.add(layers.Input(shape=(3, 5)))
        model.add(self.crf)
        model.compile('adam', loss=crf_loss)
        model.summary()

        log_likelihood = model.train_on_batch(self.logits, self.tags)

        def compute_log_likelihood():
            # Now compute the log-likelihood manually
            manual_log_likelihood = 0.0

            # For each instance, manually compute the numerator
            # (which is just the score for the logits and actual tags)
            # and the denominator
            # (which is the log-sum-exp of the scores for the logits across all possible tags)
            for logits_i, tags_i in zip(self.logits, self.tags):
                numerator = self.score(logits_i, tags_i)
                all_scores = [self.score(logits_i, tags_j)
                              for tags_j in itertools.product(range(5), repeat=3)]
                denominator = math.log(sum(math.exp(score) for score in all_scores))
                # And include them in the manual calculation.
                manual_log_likelihood += numerator - denominator

            return manual_log_likelihood

        # The manually computed log likelihood should equal the result of crf.forward.
        expected_log_likelihood = compute_log_likelihood()
        unbatched_log_likelihood = -2 * log_likelihood
        assert expected_log_likelihood == approx(unbatched_log_likelihood)

    def test_forward_works_with_mask(self):
        # Use a non-trivial mask
        mask = np.array([
                [1, 1, 1],
                [1, 1, 0]
        ])

        model = Sequential()
        model.add(layers.Input(shape=(3, 5)))
        model.add(MockMasking(mask_shape=(2, 3), mask_value=mask))
        model.add(self.crf)
        model.compile('adam', loss=crf_loss)
        model.summary()

        log_likelihood = model.train_on_batch(self.logits, self.tags)

        def compute_log_likelihood():
            # Now compute the log-likelihood manually
            manual_log_likelihood = 0.0

            # For each instance, manually compute the numerator
            #   (which is just the score for the logits and actual tags)
            # and the denominator
            #   (which is the log-sum-exp of the scores for the logits across all possible tags)
            for logits_i, tags_i, mask_i in zip(self.logits, self.tags, mask):
                # Find the sequence length for this input and only look at that much of each sequence.
                sequence_length = np.sum(mask_i)
                logits_i = logits_i[:sequence_length]
                tags_i = tags_i[:sequence_length]

                numerator = self.score(logits_i, tags_i)
                all_scores = [self.score(logits_i, tags_j)
                              for tags_j in itertools.product(range(5), repeat=sequence_length)]
                denominator = math.log(sum(math.exp(score) for score in all_scores))
                # And include them in the manual calculation.
                manual_log_likelihood += numerator - denominator

            return manual_log_likelihood

        exepected_log_likelihood = compute_log_likelihood()

        # The manually computed log likelihood should equal the result of crf.forward.
        unbatched_log_likelihood = -2 * log_likelihood

        assert exepected_log_likelihood == approx(unbatched_log_likelihood)


    def test_viterbi_tags(self):
        mask = np.array([
                [1, 1, 1],
                [1, 1, 0]
        ])

        model = Sequential()
        model.add(layers.Input(shape=(3, 5)))
        model.add(MockMasking(mask_shape=(2, 3), mask_value=mask))
        model.add(self.crf)
        model.compile('adam', loss=crf_loss)
        model.summary()

        # Separate the tags and scores.
        viterbi_tags = model.predict(self.logits)

        # Check that the viterbi tags are what I think they should be.
        assert viterbi_tags == [
                [2, 4, 3],
                [4, 2]
        ]

        # We can also iterate over all possible tag sequences and use self.score
        # to check the likelihood of each. The most likely sequence should be the
        # same as what we get from viterbi_tags.
        most_likely_tags = []
        best_scores = []

        for logit, mas in zip(self.logits, mask):
            sequence_length = np.sum(mas.detach())
            most_likely, most_likelihood = None, -float('inf')
            for tags in itertools.product(range(5), repeat=sequence_length):
                score = self.score(logit.data, tags)
                if score > most_likelihood:
                    most_likely, most_likelihood = tags, score
            # Convert tuple to list; otherwise == complains.
            most_likely_tags.append(list(most_likely))
            best_scores.append(most_likelihood)

        assert viterbi_tags == most_likely_tags

    def test_constrained_viterbi_tags(self):
        constraints = {(0, 0), (0, 1),
                       (1, 1), (1, 2),
                       (2, 2), (2, 3),
                       (3, 3), (3, 4),
                       (4, 4), (4, 0)}

        # Add the transitions to the end tag
        # and from the start tag.
        for i in range(5):
            constraints.add((5, i))
            constraints.add((i, 6))

        mask = np.array([
                [1, 1, 1],
                [1, 1, 0]
        ])

        crf = CRF(
            units=5,
            use_kernel=False,  # disable kernel transform
            chain_initializer=initializers.Constant(self.transitions),
            left_boundary_initializer=initializers.Constant(self.transitions_from_start),
            right_boundary_initializer=initializers.Constant(self.transitions_to_end),
            transition_constraint=constraints
        )

        model = Sequential()
        model.add(layers.Input(shape=(3, 5)))
        model.add(MockMasking(mask_shape=(2, 3), mask_value=mask))
        model.add(crf)
        model.compile('adam', loss=crf_loss)
        model.summary()

        for layer in model.layers:
            print(layer.get_config())
            print(dict(zip(layer.weights, layer.get_weights())))

        # Get just the tags from each tuple of (tags, score).
        viterbi_tags = model.predict(self.logits)

        # Now the tags should respect the constraints
        expected_tags = [
            [2, 3, 3],
            [2, 3, 0]
        ]

        # if constrain not work it should be:
        # [
        #     [2, 4, 3],
        #     [2, 3, 0]
        # ]

        assert (viterbi_tags == expected_tags).all()
