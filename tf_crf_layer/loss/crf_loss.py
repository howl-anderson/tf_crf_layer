import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.losses import sparse_categorical_crossentropy, \
    categorical_crossentropy
from tensorflow.python.keras.utils.losses_utils import ReductionV2

from tf_crf_layer import keras_utils
from tf_crf_layer.keras_utils import register_keras_custom_object


def crf_nll(y_true, y_pred):
    crf, idx = y_pred._keras_history[:2]
#     if crf._outbound_nodes:
#         raise TypeError('When learn_model="join", CRF must be the last layer.')
#     if crf.sparse_target:
#         y_true = tf.one_hot(tf.cast(y_true[:, :, 0], 'int32'), crf.units)

    # node = crf._inbound_nodes[idx]
    #
    # print(node)
    # print(node.input_tensors)

    # X = node.input_tensors[0]
    # mask = node.input_tensors[1]

    # nloglik = crf.get_negative_log_likelihood(y_pred, y_true, mask)
    nloglik = crf.get_negative_log_likelihood(y_true)  # shape: (batch_size, )

    return nloglik


@register_keras_custom_object
def crf_loss(y_true, y_pred):
    crf, idx = y_pred._keras_history[:2]
    print(crf, idx)

    return crf_nll(y_true, y_pred)


# @register_keras_custom_object
# class ConditionalRandomFieldLoss(tf.keras.losses.Loss):
#     def __init__(self, name=None):
#         super(ConditionalRandomFieldLoss, self).__init__(reduction=ReductionV2.SUM_OVER_BATCH_SIZE, name=name)
#
#     def get_config(self):
#         config = {}
#         base_config = super(ConditionalRandomFieldLoss, self).get_config()
#         return dict(list(base_config.items()) + list(config.items()))
#
#     def call(self, y_true, y_pred):
#         loss_vector = crf_loss(y_true, y_pred)
#
#         two_dim_loss = K.expand_dims(loss_vector)
#
#         mask = K.ones_like(y_true)
#
#         full_shape_loss = tf.math.multiply(two_dim_loss, mask)
#
#         return full_shape_loss


@register_keras_custom_object
class ConditionalRandomFieldLoss(object):
    def get_config(self):
        return {}

    def __call__(self, y_true, y_pred, sample_weight=None):
        loss_vector = crf_loss(y_true, y_pred)

        return K.mean(loss_vector)
