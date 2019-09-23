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
from tf_crf_layer.keras_utils import register_keras_custom_object


@register_keras_custom_object
class SequenceCorrectness(Mean):
    def __init__(self, name="sequence_correctness", dtype=None):
        """Creates a `CategoricalAccuracy` instance.
        Args:
          name: (Optional) string name of the metric instance.
          dtype: (Optional) data type of the metric result.
        """
        super(SequenceCorrectness, self).__init__(
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
        judge = K.equal(y_pred, y_true)  # shape: (B, T)
        value = K.all(judge, axis=1, keepdims=False)  # shape: (B, )

        # # # DEBUG: output training value
        # print_op = tf.print(value)
        # with tf.control_dependencies([print_op]):
        #     value = tf.identity(value)

        super(SequenceCorrectness, self).update_state(value)
