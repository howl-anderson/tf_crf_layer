import tensorflow as tf
from tensorflow.python.keras.metrics import Mean
from tensorflow.python.keras import backend as K

# for future reference:
# B: batch size
# M: intent number
# n: where n is the tag set number
# n+2: n plus start and end tag
# N: n * n
# T: sequence length
# F: feature number


class SequenceSpanAccuracy(Mean):
    def __init__(self, name="sequence_span_accuracy", dtype=None):
        """Creates a `CategoricalAccuracy` instance.
        Args:
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(SequenceSpanAccuracy, self).__init__(
            name=name,
            dtype=dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        """

        :param y_true: Tensor shape: (B, T)
        :param y_pred: Tensor shape (B, T)
        :param sample_weight: None or Tensor shape (B, T)
        :return:
        """
        value = K.equal(y_pred, y_true)  # shape: (B, T)

        # # # DEBUG: output training value
        print_op = tf.print(sample_weight)
        with tf.control_dependencies([print_op]):
            value = tf.identity(value)

        super(SequenceSpanAccuracy, self).update_state(value, sample_weight)
