import unittest

import numpy as np
from tensorflow.python.keras import Sequential, initializers, models
from tensorflow.python.keras import layers

from tests.MockedMasking import MockMasking
from tf_crf_layer.layer import CRF
from tf_crf_layer.numpy_crf import CRF as NPCRF
from tf_crf_layer.loss import crf_loss


class TestDynamicTransitionConstraint(unittest.TestCase):
    def setUp(self):
        super().setUp()
        # n (number of tags): 5
        # B (batch size): 2
        # T (sequence length): 3
        # C (number of intent): 2

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

        self.transition_constraint_matrix = np.array(
            [
                [1] * 30 + [0] * 19,  # [1] * ((5 + 2) * (5 + 2)),
                [1] * ((5 + 2) * (5 + 2))
            ]
        )

        self.dynamic_constraint = np.array([
            [0, 1],
            [1, 0]
        ])

        self.mask = np.array([[1, 1, 1], [1, 1, 0]])

        self.crf = CRF(
            units=5,
            use_kernel=False,  # disable kernel transform
            chain_initializer=initializers.Constant(self.transitions),
            use_boundary=True,
            left_boundary_initializer=initializers.Constant(self.transitions_from_start),
            right_boundary_initializer=initializers.Constant(self.transitions_to_end),
            transition_constraint_matrix=self.transition_constraint_matrix
        )

        self.np_crf = NPCRF(
            self.transitions,
            self.transitions_from_start,
            self.transitions_to_end
        )

    def test_decode_without_mask(self):
        raw_logits_input = layers.Input(shape=(3, 5))
        logits_input = raw_logits_input
        dynamic_constraint_input = layers.Input(shape=(2,))

        output_layer = self.crf([logits_input, dynamic_constraint_input])

        model = models.Model([raw_logits_input, dynamic_constraint_input], output_layer)
        model.summary()

        out = model.predict([self.logits, self.dynamic_constraint])

        # compute expected value
        rolled_mask = self.dynamic_constraint * self.transition_constraint_matrix
        mask = np.sum(rolled_mask, axis=1)
        expect_decode = self.np_crf.viterbi_decode(self.logits, mask)

        assert out.shape == (2, 3)
        print(out)

    def test_decode_with_mask(self):
        raw_logits_input = layers.Input(shape=(3, 5))
        # mask_input = MockMasking(mask_shape=(2, 3), mask_value=mask)
        # logits_input = mask_input(raw_logits_input)
        logits_input = raw_logits_input
        dynamic_constraint_input = layers.Input(shape=(2,))

        output_layer = self.crf([logits_input, dynamic_constraint_input])

        model = models.Model([raw_logits_input, dynamic_constraint_input], output_layer)
        model.summary()

        out = model.predict([self.logits, self.dynamic_constraint])

        assert out.shape == (2, 3)
        print(out)