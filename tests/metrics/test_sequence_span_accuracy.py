import tensorflow as tf
import numpy as np

from tf_crf_layer.metrics import SequenceSpanAccuracy


class SequenceSpanAccuracyTest(tf.test.TestCase):
    def test_config(self):
        # default
        sc = SequenceSpanAccuracy()
        self.assertEqual(sc.name, 'sequence_span_accuracy')
        self.assertEqual(sc.dtype, tf.float32)

        # Check save and restore config
        sc = SequenceSpanAccuracy.from_config(sc.get_config())
        self.assertEqual(sc.name, 'sequence_span_accuracy')
        self.assertEqual(sc.dtype, tf.float32)

    def test_one_time_result(self):
        with self.session():
            spa = SequenceSpanAccuracy()

            self.evaluate(tf.compat.v1.variables_initializer(spa.variables))

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

            update_op = spa.update_state(y_true, y_pred)
            self.evaluate(update_op)  # total:12, count: 9

            result = self.evaluate(spa.result())
            self.assertAlmostEqual(result, 0.75)

    def test_one_time_result_with_sample_weight(self):
        with self.session():
            spa = SequenceSpanAccuracy()

            self.evaluate(tf.compat.v1.variables_initializer(spa.variables))

            # fmt: off
            y_true = np.array(
            [
                [1, 1, 1],
                [1, 1, 0],
                [0, 1, 1],
                [0, 1, 0],
            ]
            )
            y_pred = np.array(
            [
                [1, 0, 1],
                [1, 0, 0],
                [0, 0, 1],
                [0, 1, 0],
            ]
            )
            sample_weight = np.array(
                [
                    [1, 1, 1],
                    [1, 1, 0],
                    [0, 1, 1],
                    [0, 1, 0],
                ]
            )
            # fmt: on

            update_op = spa.update_state(y_true, y_pred, sample_weight)
            self.evaluate(update_op)  # total: 8, count: 5

            result = self.evaluate(spa.result())

            # wrong result should be: 8 / 12
            self.assertAlmostEqual(result, 5/8)

    def test_multiply_time_result(self):
        with self.session():
            sc = SequenceSpanAccuracy()

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
            self.evaluate(update_op)  # total:12, count: 9

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
            self.evaluate(update_op)  # total:12+18=30, count: 9+18=27

            result = self.evaluate(sc.result())
            self.assertAlmostEqual(result, 0.9)
