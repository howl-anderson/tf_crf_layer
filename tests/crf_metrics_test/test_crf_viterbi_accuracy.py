import numpy as np
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Embedding

from layer import CRF
from loss import crf_loss
from metrics import crf_viterbi_accuracy
from tests.common import get_random_data


def test_crf_viterbi_accuracy():
    nb_samples = 2
    timesteps = 10
    embedding_dim = 4
    output_dim = 5
    embedding_num = 12

    x, y = get_random_data(nb_samples, timesteps, x_high=embedding_num, y_high=output_dim)
    x[0, -4:] = 0  # right padding; left padding is not supported due to the tf.contrib.crf

    # test with masking, fix length
    model = Sequential()
    model.add(Embedding(embedding_num, embedding_dim, input_length=timesteps, mask_zero=True))
    model.add(CRF(output_dim))
    model.compile(optimizer='rmsprop', loss=crf_loss, metrics=[crf_viterbi_accuracy])
    model.summary()
    model.fit(x, y, epochs=1, batch_size=10)

    # test viterbi_acc
    y_pred = model.predict(x)
    _, v_acc = model.evaluate(x, y)
    np_acc = (y_pred[x > 0] == y[x > 0]).astype('float32').mean()
    print(v_acc, np_acc)
    assert np.abs(v_acc - np_acc) < 1e-4
