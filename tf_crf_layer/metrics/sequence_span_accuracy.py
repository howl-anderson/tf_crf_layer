import tensorflow as tf
from tensorflow.python.keras.metrics import Mean
from tensorflow.python.keras import backend as K

from tf_crf_layer.keras_utils import register_keras_custom_object


# for future reference:
# B: batch size
# M: intent number
# n: where n is the tag set number
# n+2: n plus start and end tag
# N: n * n
# T: sequence length
# F: feature number


@register_keras_custom_object
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

    def update_state(self, y_true, y_pred, *args, **kwargs):
        """

        :param y_true: Tensor shape: (B, T)
        :param y_pred: Tensor shape (B, T)
        :param sample_weight: None or Tensor shape (B, T)
        :return:
        """
        value = K.equal(y_pred, y_true)  # shape: (B, T)

        # # # DEBUG: output training value
        # print_op = tf.print(mask)
        # with tf.control_dependencies([print_op]):
        #     value = tf.identity(value)

        super(SequenceSpanAccuracy, self).update_state(value, *args, **kwargs)


def get_mask_from_keras_tensor(y_pred):
    layer, node_index = y_pred._keras_history[:2]
    mask = layer.output_mask

    return mask


@register_keras_custom_object
def sequence_span_accuracy(y_true, y_pred):
    """

    :param y_true: Tensor shape: (B, T)
    :param y_pred: Tensor shape (B, T)
    :return:
    """
    judge = K.cast(K.equal(y_pred, y_true), K.floatx())  # shape: (B, T)
    mask = get_mask_from_keras_tensor(y_pred)  # shape: (B, T)
    if mask is None:
        result = K.mean(judge)
        return result
    else:
        mask = K.cast(mask, K.floatx())
        result = K.sum(judge * mask) / K.sum(mask)
        return result
