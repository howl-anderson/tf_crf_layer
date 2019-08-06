import numpy as np
from tensorflow.python.keras import Sequential, initializers, models
from tensorflow.python.keras import layers

from tests.MockedMasking import MockMasking
from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import crf_loss


def test_dynamic_transition_constraint():
    # n (number of tags): 5
    # B (batch size): 2
    # T (sequence length): 3
    # C (number of intent): 2

    logits = np.array([
        [[0, 0, .5, .5, .2], [0, 0, .3, .3, .1], [0, 0, .9, 10, 1]],
        [[0, 0, .2, .5, .2], [0, 0, 3, .3, .1], [0, 0, .9, 1, 1]],
    ])
    tags = np.array([
        [2, 3, 4],
        [3, 2, 2]
    ])

    transitions = np.array([
        [0.1, 0.2, 0.3, 0.4, 0.5],
        [0.8, 0.3, 0.1, 0.7, 0.9],
        [-0.3, 2.1, -5.6, 3.4, 4.0],
        [0.2, 0.4, 0.6, -0.3, -0.4],
        [1.0, 1.0, 1.0, 1.0, 1.0]
    ])

    transitions_from_start = np.array([0.1, 0.2, 0.3, 0.4, 0.6])
    transitions_to_end = np.array([-0.1, -0.2, 0.3, -0.4, -0.4])

    transition_constraint_matrix = np.array(
        [
            [1] * 30 + [0] * 19,  # [1] * ((5 + 2) * (5 + 2)),
            [1] * ((5 + 2) * (5 + 2))
        ]
    )

    dynamic_constraint = np.array([
        [0, 1],
        [1, 0]
    ])

    mask = np.array([[1, 1, 1], [1, 1, 0]])

    raw_logits_input = layers.Input(shape=(3, 5))
    # mask_input = MockMasking(mask_shape=(2, 3), mask_value=mask)
    # logits_input = mask_input(raw_logits_input)
    logits_input = raw_logits_input
    dynamic_constraint_input = layers.Input(shape=(2,))

    crf = CRF(
        units=5,
        use_kernel=False,  # disable kernel transform
        chain_initializer=initializers.Constant(transitions),
        left_boundary_initializer=initializers.Constant(
            transitions_from_start),
        right_boundary_initializer=initializers.Constant(
            transitions_to_end),
        transition_constraint_matrix=transition_constraint_matrix
    )

    output_layer = crf([logits_input, dynamic_constraint_input])

    model = models.Model([raw_logits_input, dynamic_constraint_input], output_layer)
    model.summary()

    out = model.predict([logits, dynamic_constraint])

    assert out.shape == (2, 3)
    print(out)
