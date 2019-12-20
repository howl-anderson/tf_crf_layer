import numpy as np
import tensorflow as tf

from tf_crf_layer.layer import CRF


def test_crf_add_boundary_energy_with_no_mask():
    energy = tf.constant(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=tf.float32,
    )

    mask = None
    start = tf.constant([1, 1, 1, 1, 1], dtype=tf.float32)
    end = tf.constant([-1, -1, -1, -1, -1], dtype=tf.float32)

    crf = CRF(None)
    new_energy_tensor = crf.add_boundary_energy(energy, mask, start, end)

    with tf.Session() as sess:
        result = sess.run(new_energy_tensor)

    expected = np.array(
        [
            [
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1],
            ],
            [
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1],
            ],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_crf_add_boundary_energy_with_mask():
    energy = tf.constant(
        [
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
            ],
        ],
        dtype=tf.float32,
    )

    mask = tf.constant(
        [
            [0, 1, 1, 1, 1],  # pre padding
            [1, 1, 1, 1, 0],  # post padding
            [0, 1, 1, 1, 0],  # pre and post padding
        ]
    )
    start = tf.constant([1, 1, 1, 1, 1], dtype=tf.float32)
    end = tf.constant([-1, -1, -1, -1, -1], dtype=tf.float32)

    crf = CRF(None)
    new_energy_tensor = crf.add_boundary_energy(energy, mask, start, end)

    with tf.Session() as sess:
        result = sess.run(new_energy_tensor)

    expected = np.array(
        [
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1],
            ],
            [
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1],
                [0, 0, 0, 0, 0],
            ],
            [
                [0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1],
                [0, 0, 0, 0, 0],
                [-1, -1, -1, -1, -1],
                [0, 0, 0, 0, 0],
            ],
        ]
    )

    np.testing.assert_array_equal(result, expected)
