import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding

import pytest

from tf_crf_layer.layer import CRF
from tf_crf_layer.loss import crf_loss, ConditionalRandomFieldLoss
from tf_crf_layer.metrics import crf_viterbi_accuracy


@pytest.mark.skip("fixme")
def test_crf_viterbi_accuracy(get_random_data):
    nb_samples = 2
    timesteps = 10
    embedding_dim = 4
    output_dim = 5
    embedding_num = 12

    crf_loss_instance = ConditionalRandomFieldLoss()

    x, y = get_random_data(nb_samples, timesteps, x_high=embedding_num, y_high=output_dim)
    # right padding; left padding is not supported due to the tf.contrib.crf
    x[0, -4:] = 0

    # test with masking, fix length
    model = Sequential()
    model.add(Embedding(embedding_num, embedding_dim, input_length=timesteps, mask_zero=True))
    model.add(CRF(output_dim, name="crf_layer"))
    model.compile(optimizer='rmsprop', loss={"crf_layer": crf_loss_instance}, metrics=[crf_viterbi_accuracy])

    model.fit(x, y, epochs=1, batch_size=10)

    # test viterbi_acc
    y_pred = model.predict(x)
    _, v_acc = model.evaluate(x, y)
    np_acc = (y_pred[x > 0] == y[x > 0]).astype('float32').mean()
    print(v_acc, np_acc)
    assert np.abs(v_acc - np_acc) < 1e-4


if __name__ == "__main__":
    test_crf_viterbi_accuracy()
