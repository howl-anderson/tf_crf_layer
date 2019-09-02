import numpy as np
import tensorflow as tf

from tf_crf_layer.layer import CRF


def test_crf_add_boundary_energy_with_no_mask():
    sess = tf.InteractiveSession()
    sess.as_default()

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
    new_energy_tensor = crf.compute_energy(start, end, energy, mask)

    with tf.Session() as sess:
        new_energy = sess.run(new_energy_tensor)

    expected_energy = np.array(
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

    np.testing.assert_array_equal(new_energy, expected_energy)


def test_crf_add_boundary_energy_with_mask():
    sess = tf.InteractiveSession()
    sess.as_default()

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
    new_energy_tensor = crf.compute_energy(start, end, energy, mask)

    with tf.Session() as sess:
        new_energy = sess.run(new_energy_tensor)

    expected_energy = np.array(
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

    np.testing.assert_array_equal(new_energy, expected_energy)


if __name__ == "__main__":
    test_crf_add_boundary_energy_with_mask()
