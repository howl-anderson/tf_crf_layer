import tensorflow as tf
from tensorflow.python.keras.losses import sparse_categorical_crossentropy, \
    categorical_crossentropy

from tf_crf_layer import keras_utils


def crf_nll(y_true, y_pred):
    crf, idx = y_pred._keras_history[:2]
#     if crf._outbound_nodes:
#         raise TypeError('When learn_model="join", CRF must be the last layer.')
#     if crf.sparse_target:
#         y_true = tf.one_hot(tf.cast(y_true[:, :, 0], 'int32'), crf.units)

    node = crf._inbound_nodes[idx]

    print(node)
    print(node.input_tensors)

    # X = node.input_tensors[0]
    # mask = node.input_tensors[1]

    # nloglik = crf.get_negative_log_likelihood(y_pred, y_true, mask)
    nloglik = crf.get_negative_log_likelihood(y_true)

    return nloglik


def crf_loss(y_true, y_pred):
    crf, idx = y_pred._keras_history[:2]
    print(crf, idx)

    if crf.learn_mode == 'join':
        return crf_nll(y_true, y_pred)
    else:
        if crf.sparse_target:
            return sparse_categorical_crossentropy(y_true, y_pred)
        else:
            return categorical_crossentropy(y_true, y_pred)