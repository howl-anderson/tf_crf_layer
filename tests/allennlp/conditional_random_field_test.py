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
from tf_crf_layer.loss import crf_loss, ConditionalRandomFieldLoss

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
            use_boundary=True,
            left_boundary_initializer=initializers.Constant(self.transitions_from_start),
            right_boundary_initializer=initializers.Constant(self.transitions_to_end),
            name="crf_layer"
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
        crf_loss_instance = ConditionalRandomFieldLoss()

        model = Sequential()
        model.add(layers.Input(shape=(3, 5)))
        model.add(self.crf)
        model.compile('adam', loss={"crf_layer": crf_loss_instance})
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

        crf_loss_instance = ConditionalRandomFieldLoss()

        model = Sequential()
        model.add(layers.Input(shape=(3, 5)))
        model.add(MockMasking(mask_shape=(2, 3), mask_value=mask))
        model.add(self.crf)
        model.compile('adam', loss={"crf_layer": crf_loss_instance})
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

        crf_loss_instance = ConditionalRandomFieldLoss()

        model = Sequential()
        model.add(layers.Input(shape=(3, 5)))
        model.add(MockMasking(mask_shape=(2, 3), mask_value=mask))
        model.add(self.crf)
        model.compile('adam', loss={"crf_layer": crf_loss_instance})
        model.summary()

        # Separate the tags and scores.
        viterbi_tags = model.predict(self.logits)

        # Check that the viterbi tags are what I think they should be.

        expected_viterbi_tags = [
                [2, 4, 3],
                [4, 2, 0]
        ]

        # test assert
        np.testing.assert_equal(viterbi_tags, expected_viterbi_tags)

        # We can also iterate over all possible tag sequences and use self.score
        # to check the likelihood of each. The most likely sequence should be the
        # same as what we get from viterbi_tags.
        most_likely_tags = []
        best_scores = []

        for logit, mas in zip(self.logits, mask):
            sequence_length = np.sum(mas)
            most_likely, most_likelihood = None, -float('inf')
            for tags in itertools.product(range(5), repeat=sequence_length):
                score = self.score(logit, tags)
                if score > most_likelihood:
                    # padding tags to sequence length
                    tag_len_diff = 3 - len(tags)
                    if tag_len_diff:
                        tags = list(tags) + [0] * tag_len_diff

                    most_likely, most_likelihood = tags, score
            # Convert tuple to list; otherwise == complains.
            most_likely_tags.append(list(most_likely))
            best_scores.append(most_likelihood)

        # test assert
        np.testing.assert_equal(viterbi_tags, most_likely_tags)
        # No such viterbi score from current CRF implement
        # assert viterbi_scores == best_scores

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
            use_boundary=True,
            left_boundary_initializer=initializers.Constant(self.transitions_from_start),
            right_boundary_initializer=initializers.Constant(self.transitions_to_end),
            transition_constraint=constraints,
            name="crf_layer"
        )

        crf_loss_instance = ConditionalRandomFieldLoss()

        model = Sequential()
        model.add(layers.Input(shape=(3, 5)))
        model.add(MockMasking(mask_shape=(2, 3), mask_value=mask))
        model.add(crf)
        model.compile('adam', loss={"crf_layer": crf_loss_instance})
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

        # test assert
        np.testing.assert_equal(viterbi_tags, expected_tags)

    def test_unmasked_constrained_viterbi_tags(self):
        # TODO: using BILUO tag scheme instead of BIO.
        #       So that, transition from tags to end can be tested.

        raw_constraints = np.array([
            #     O     B-X    I-X    B-Y    I-Y  start   end
            [     1,     1,     0,     1,     0,    0,     1],  # O
            [     1,     1,     1,     1,     0,    0,     1],  # B-X
            [     1,     1,     1,     1,     0,    0,     1],  # I-X
            [     1,     1,     0,     1,     1,    0,     1],  # B-Y
            [     1,     1,     0,     1,     1,    0,     1],  # I-Y
            [     1,     1,     0,     1,     0,    0,     0],  # start
            [     0,     0,     0,     0,     0,    0,     0],  # end
        ])

        constraints = np.argwhere(raw_constraints > 0).tolist()

        # transitions = np.array([
        #     #     O     B-X    I-X    B-Y    I-Y
        #     [    0.1,   0.2,   0.3,   0.4,   0.5],  # O
        #     [    0.8,   0.3,   0.1,   0.7,   0.9],  # B-X
        #     [   -0.3,   2.1,  -5.6,   3.4,   4.0],  # I-X
        #     [    0.2,   0.4,   0.6,  -0.3,  -0.4],  # B-Y
        #     [    1.0,   1.0,   1.0,   1.0,   1.0]   # I-Y
        # ])

        transitions = np.ones([5, 5])

        # transitions_from_start = np.array(
        #     #     O     B-X    I-X    B-Y    I-Y
        #     [    0.1,   0.2,   0.3,   0.4,   0.6]  # start
        # )

        transitions_from_start = np.ones(5)

        # transitions_to_end = np.array(
        #     [
        #     #    end
        #         -0.1,  # O
        #         -0.2,  # B-X
        #          0.3,  # I-X
        #         -0.4,  # B-Y
        #         -0.4   # I-Y
        #     ]
        # )

        transitions_to_end = np.ones(5)

        logits = np.array([
            [
            # constraint transition from start to tags
            #     O     B-X    I-X    B-Y    I-Y
                [ 0.,    .1,   1.,     0.,   0.],
                [ 0.,    0.,   1.,     0.,   0.],
                [ 0.,    0.,   1.,     0.,   0.]
            ],
            [
            # constraint transition from tags to tags
            #     O     B-X    I-X    B-Y    I-Y
                [ 0.,    1.,   0.,     0.,   0.],
                [ 0.,    0.,   .1,     1.,   0.],
                [ 0.,    0.,   1.,     0.,   0.]
            ]
        ])

        crf = CRF(
            units=5,
            use_kernel=False,  # disable kernel transform
            chain_initializer=initializers.Constant(transitions),
            use_boundary=True,
            left_boundary_initializer=initializers.Constant(transitions_from_start),
            right_boundary_initializer=initializers.Constant(transitions_to_end),
            transition_constraint=constraints,
            name="crf_layer"
        )

        crf_loss_instance = ConditionalRandomFieldLoss()

        model = Sequential()
        model.add(layers.Input(shape=(3, 5)))
        model.add(crf)
        model.compile('adam', loss={"crf_layer": crf_loss_instance})
        model.summary()

        for layer in model.layers:
            print(layer.get_config())
            print(dict(zip(layer.weights, layer.get_weights())))

        # Get just the tags from each tuple of (tags, score).
        viterbi_tags = model.predict(logits)

        # Now the tags should respect the constraints
        expected_tags = [
            [1, 2, 2],  # B-X  I-X  I-X
            [1, 2, 2]   # B-X  I-X  I-X
        ]

        # if constrain not work it should be:
        # [
        #     [2, 4, 3],
        #     [2, 3, 0]
        # ]

        # test assert
        np.testing.assert_equal(viterbi_tags, expected_tags)

    def test_masked_viterbi_decode(self):
        transitions = np.ones([5, 5])
        transitions_from_start = np.ones(5)
        transitions_to_end = np.ones(5)

        logits = np.array([
            [
            #     O     B-X    I-X    B-Y    I-Y
                [ 0.,    1.,   0.,     0.,   0.],
                [ 0.,    0.,   1.,     0.,   0.],
                [ 0.,    0.,   1.,     0.,   0.]
            ],
            [
            #     O     B-X    I-X    B-Y    I-Y
                [ 0.,    1.,   0.,     0.,   0.],
                [ 0.,    1.,   0.,     0.,   0.],
                [ 0.,    1.,   0.,     0.,   0.]
            ]
        ])

        # TODO: this test case is right padding mask only
        #       due to the underline crf function only support sequence length
        mask = np.array([
                [1, 1, 0],
                [1, 1, 0]
        ])

        crf = CRF(
            units=5,
            use_kernel=False,  # disable kernel transform
            chain_initializer=initializers.Constant(transitions),
            use_boundary=True,
            left_boundary_initializer=initializers.Constant(transitions_from_start),
            right_boundary_initializer=initializers.Constant(transitions_to_end),
            name="crf_layer"
        )

        crf_loss_instance = ConditionalRandomFieldLoss()

        model = Sequential()
        model.add(layers.Input(shape=(3, 5)))
        model.add(MockMasking(mask_shape=(2, 3), mask_value=mask))
        model.add(crf)
        model.compile('adam', loss={"crf_layer": crf_loss_instance})
        model.summary()

        for layer in model.layers:
            print(layer.get_config())
            print(dict(zip(layer.weights, layer.get_weights())))

        # Get just the tags from each tuple of (tags, score).
        viterbi_tags = model.predict(logits)

        # Now the tags should respect the constraints
        expected_tags = [
            [1, 2, 0],  # B-X  I-X  NA
            [1, 1, 0]   # B-X  B-X  NA
        ]

        # if constrain not work it should be:
        # [
        #     [2, 4, 3],
        #     [2, 3, 0]
        # ]

        # test assert
        np.testing.assert_equal(viterbi_tags, expected_tags)
