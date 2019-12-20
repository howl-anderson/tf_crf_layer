# pylint: disable=no-self-use,invalid-name
import itertools
import math
import unittest

import numpy as np
import pytest

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
            # left_boundary_initializer=initializers.Constant(self.transitions_from_start),
            # right_boundary_initializer=initializers.Constant(self.transitions_to_end),
            name="crf_layer"
        )
        self.crf.left_boundary = self.crf.add_weight(
            shape=(self.crf.units,),
            name="left_boundary",
            initializer=initializers.Constant(self.transitions_from_start),
        )
        self.crf.right_boundary = self.crf.add_weight(
            shape=(self.crf.units,),
            name="right_boundary",
            initializer=initializers.Constant(self.transitions_to_end),
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




    @pytest.mark.skip("constrain is not supported yet")
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
            # left_boundary_initializer=initializers.Constant(self.transitions_from_start),
            # right_boundary_initializer=initializers.Constant(self.transitions_to_end),
            transition_constraint=constraints,
            name="crf_layer"
        )
        crf.left_boundary = crf.add_weight(
            shape=(5,),
            name="left_boundary",
            initializer=initializers.Constant(self.transitions_from_start),
        )
        crf.right_boundary = crf.add_weight(
            shape=(5,),
            name="right_boundary",
            initializer=initializers.Constant(self.transitions_to_end),
        )


        crf_loss_instance = ConditionalRandomFieldLoss()

        model = Sequential()
        model.add(layers.Input(shape=(3, 5)))
        model.add(MockMasking(mask_shape=(2, 3), mask_value=mask))
        model.add(crf)
        model.compile('adam', loss={"crf_layer": crf_loss_instance})

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

    @pytest.mark.skip("constrain is not supported yet")
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
            # left_boundary_initializer=initializers.Constant(transitions_from_start),
            # right_boundary_initializer=initializers.Constant(transitions_to_end),
            transition_constraint=constraints,
            name="crf_layer"
        )
        crf.left_boundary = crf.add_weight(
            shape=(5,),
            name="left_boundary",
            initializer=initializers.Constant(self.transitions_from_start),
        )
        crf.right_boundary = crf.add_weight(
            shape=(5,),
            name="right_boundary",
            initializer=initializers.Constant(self.transitions_to_end),
        )

        crf_loss_instance = ConditionalRandomFieldLoss()

        model = Sequential()
        model.add(layers.Input(shape=(3, 5)))
        model.add(crf)
        model.compile('adam', loss={"crf_layer": crf_loss_instance})

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

def test_masked_viterbi_decode():
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
        # left_boundary_initializer=initializers.Constant(transitions_from_start),
        # right_boundary_initializer=initializers.Constant(transitions_to_end),
        name="crf_layer"
    )

    crf_loss_instance = ConditionalRandomFieldLoss()

    model = Sequential()
    model.add(layers.Input(shape=(3, 5)))
    model.add(MockMasking(mask_shape=(2, 3), mask_value=mask))
    model.add(crf)
    model.compile('adam', loss={"crf_layer": crf_loss_instance})

    # for layer in model.layers:
    #     print(layer.get_config())
    #     print(dict(zip(layer.weights, layer.get_weights())))

    # Get just the tags from each tuple of (tags, score).
    result = model.predict(logits)

    # Now the tags should respect the constraints
    expected = [
        [1, 2, 0],  # B-X  I-X  NA
        [1, 1, 0]   # B-X  B-X  NA
    ]

    # if constrain not work it should be:
    # [
    #     [2, 4, 3],
    #     [2, 3, 0]
    # ]

    # test assert
    np.testing.assert_equal(result, expected)


def test_viterbi_tags(numpy_crf):
    logits = np.array([
        [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
        [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
    ])
    transitions = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.8, 0.3, 0.1, 0.7, 0.9],
        [-0.3, 2.1, -5.6, 3.4, 4.0],
        [0.2, 0.4, 0.6, -0.3, -0.4],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ])

    boundary_transitions = np.array([0.1, 0.2, 0.3, 0.4, 0.6])

    # Use the CRF Module with fixed transitions to compute the log_likelihood
    crf = CRF(
        units=5,
        use_kernel=False,  # disable kernel transform
        chain_initializer=initializers.Constant(transitions),
        use_boundary=True,
        boundary_initializer=initializers.Constant(boundary_transitions),
        name="crf_layer"
    )
    mask = np.array([
            [1, 1, 1],
            [1, 1, 0]
    ])

    crf_loss_instance = ConditionalRandomFieldLoss()

    model = Sequential()
    model.add(layers.Input(shape=(3, 5)))
    model.add(MockMasking(mask_shape=(2, 3), mask_value=mask))
    model.add(crf)
    model.compile('adam', loss={"crf_layer": crf_loss_instance})

    # Separate the tags and scores.
    result = model.predict(logits)

    numpy_crf_instance = numpy_crf(logits, mask, transitions, boundary_transitions, boundary_transitions)
    expected, _ = numpy_crf_instance.decode()

    np.testing.assert_equal(result, expected)


def test_forward_works_without_mask(numpy_crf):
    logits = np.array([
        [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
        [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
    ])
    transitions = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.8, 0.3, 0.1, 0.7, 0.9],
        [-0.3, 2.1, -5.6, 3.4, 4.0],
        [0.2, 0.4, 0.6, -0.3, -0.4],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ])

    boundary_transitions = np.array([0.1, 0.2, 0.3, 0.4, 0.6])

    tags = np.array([
            [2, 3, 4],
            [3, 2, 2]
    ])

    # Use the CRF Module with fixed transitions to compute the log_likelihood
    crf = CRF(
        units=5,
        use_kernel=False,  # disable kernel transform
        chain_initializer=initializers.Constant(transitions),
        use_boundary=True,
        boundary_initializer=initializers.Constant(boundary_transitions),
        name="crf_layer"
    )

    crf_loss_instance = ConditionalRandomFieldLoss()

    model = Sequential()
    model.add(layers.Input(shape=(3, 5)))
    model.add(crf)
    model.compile('adam', loss={"crf_layer": crf_loss_instance})

    result = model.train_on_batch(logits, tags)

    numpy_crf_instance = numpy_crf(logits, None, transitions, boundary_transitions, boundary_transitions)
    expected = numpy_crf_instance.compute_log_likehood(tags) / -2

    assert result == approx(expected)


@pytest.mark.skip("fixme")
def test_forward_works_with_mask(numpy_crf):
    logits = np.array([
        [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
        [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
    ])
    transitions = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.8, 0.3, 0.1, 0.7, 0.9],
        [-0.3, 2.1, -5.6, 3.4, 4.0],
        [0.2, 0.4, 0.6, -0.3, -0.4],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ])

    boundary_transitions = np.array([0.1, 0.2, 0.3, 0.4, 0.6])

    tags = np.array([
            [2, 3, 4],
            [3, 2, 2]
    ])

    # Use the CRF Module with fixed transitions to compute the log_likelihood
    crf = CRF(
        units=5,
        use_kernel=False,  # disable kernel transform
        chain_initializer=initializers.Constant(transitions),
        use_boundary=True,
        boundary_initializer=initializers.Constant(boundary_transitions),
        name="crf_layer"
    )
    # Use a non-trivial mask
    mask = np.array([
            [1, 1, 1],
            [1, 1, 0]
    ])

    crf_loss_instance = ConditionalRandomFieldLoss()

    model = Sequential()
    model.add(layers.Input(shape=(3, 5)))
    model.add(MockMasking(mask_shape=(2, 3), mask_value=mask))
    model.add(crf)
    model.compile('adam', loss={"crf_layer": crf_loss_instance})

    result = model.train_on_batch(logits, tags)


    numpy_crf_instance = numpy_crf(logits, mask, transitions, boundary_transitions, boundary_transitions)
    expected = numpy_crf_instance.compute_log_likehood(tags) / -2

    assert result == approx(expected)
