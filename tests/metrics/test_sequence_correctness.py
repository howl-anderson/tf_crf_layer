import tensorflow as tf
import numpy as np

from tf_crf_layer.metrics.sequence_correctness import SequenceCorrectness


class SequenceCorrectnessTest(tf.test.TestCase):
    def test_config(self):
        # default
        sc = SequenceCorrectness()
        self.assertEqual(sc.name, 'sequence_correctness')
        self.assertEqual(sc.dtype, tf.float32)

        # Check save and restore config
        sc = SequenceCorrectness.from_config(sc.get_config())
        self.assertEqual(sc.name, 'sequence_correctness')
        self.assertEqual(sc.dtype, tf.float32)

    def test_one_time_result(self):
        with self.session():
            sc = SequenceCorrectness()

            self.evaluate(tf.compat.v1.variables_initializer(sc.variables))

            # fmt: off
            y_true = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0]
            ]
            )
            y_pred = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 0],
                [1, 1, 1]
            ]
            )
            # fmt: on

            update_op = sc.update_state(y_true, y_pred)
            self.evaluate(update_op)

            result = self.evaluate(sc.result())
            self.assertAlmostEqual(result, 0.75)

    def test_multiply_time_result(self):
        with self.session():
            sc = SequenceCorrectness()

            self.evaluate(tf.compat.v1.variables_initializer(sc.variables))

            # fmt: off
            y_true = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 0],
                [0, 0, 0]
            ]
            )
            y_pred = np.array(
            [
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 0],
                [1, 1, 1]
            ]
            )
            # fmt: on

            update_op = sc.update_state(y_true, y_pred)
            self.evaluate(update_op)  # total:4, count: 3

            # fmt: off
            y_true = np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
                [1, 0, 0]
            ]
            )
            y_pred = np.array(
            [
                [1, 0, 0],
                [1, 1, 0],
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 0],
                [1, 0, 0]
            ]
            )
            # fmt: on

            update_op = sc.update_state(y_true, y_pred)
            self.evaluate(update_op)  # total:10, count: 9

            result = self.evaluate(sc.result())
            self.assertAlmostEqual(result, 0.9)
